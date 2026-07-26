from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import CatchException, report_exception, update_ui, update_ui_latest_msg, write_history_to_file, promote_file_to_downloadzone
import arxiv
import re

@CatchException
def Arxiv_Search_Assistant(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
    """
    ArXiv 学术论文检索助手（免代理直连 - 高精准拓展版）
    在 arXiv 计算机与 AI 领域直连检索海量相关论文，并由大模型生成深度对比分析
    """
    txt = txt.strip()
    # 允许解析用户指定数量，如输入 "RAG long context 20" 或高级参数
    max_results = 20
    advanced_arg = plugin_kwargs.get("advanced_arg", "").strip()
    if advanced_arg and advanced_arg.isdigit():
        max_results = int(advanced_arg)
    else:
        # 尝试从 txt 中提取结尾数字
        match = re.search(r'\b(\d{1,2})\b$', txt)
        if match and int(match.group(1)) in range(5, 51):
            max_results = int(match.group(1))
            txt = txt[:match.start()].strip()

    if not txt:
        txt = "RAG long context 2024 2025"

    chatbot.append((f"正在为您检索 {max_results} 篇 arXiv 论文：{txt}", f"[Local Message] 正在连接 arXiv API，按【相关度 + CS/AI 领域】检索 Top-{max_results} 篇文献..."))
    yield from update_ui(chatbot=chatbot, history=history)

    # 构造精准的 CS/AI 领域 Query
    clean_query = txt
    if not any(cat in clean_query for cat in ["cat:", "cs.CL", "cs.AI"]):
        clean_query = f"(cat:cs.CL OR cat:cs.AI OR cat:cs.IR) AND ({clean_query})"

    client = arxiv.Client()
    search = arxiv.Search(
        query=clean_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    try:
        for r in client.results(search):
            results.append({
                "title": r.title,
                "authors": [a.name for a in r.authors[:3]],
                "published": r.published.strftime("%Y-%m-%d"),
                "pdf_url": r.pdf_url,
                "arxiv_id": r.entry_id.split('/')[-1],
                "summary": r.summary.replace('\n', ' ')
            })
    except Exception as e:
        # 如果带有 cat: 查询失败，退回普通相关度查询
        try:
            search_fallback = arxiv.Search(query=txt, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            for r in client.results(search_fallback):
                results.append({
                    "title": r.title,
                    "authors": [a.name for a in r.authors[:3]],
                    "published": r.published.strftime("%Y-%m-%d"),
                    "pdf_url": r.pdf_url,
                    "arxiv_id": r.entry_id.split('/')[-1],
                    "summary": r.summary.replace('\n', ' ')
                })
        except Exception as ex:
            chatbot.append((f"arXiv 检索出错", f"[Local Message] 检索失败: {ex}"))
            yield from update_ui(chatbot=chatbot, history=history)
            return

    if not results:
        chatbot.append((f"未检索到文献", f"[Local Message] 未能根据【{txt}】检索到论文，请尝试调整关键词。"))
        yield from update_ui(chatbot=chatbot, history=history)
        return

    yield from update_ui_latest_msg(lastmsg=f"成功检索到 {len(results)} 篇高度相关的 arXiv 论文，正在由大模型生成对比矩阵...", chatbot=chatbot, history=history, delay=1)

    i_say = (
        f"以下是检索到的最新 {len(results)} 篇论文元数据（包含标题、作者、发表日期、链接与 Abstract）：\n\n{str(results)}\n\n"
        f"请针对主题【{txt}】完成以下符合国内学术规范的深度输出：\n"
        f"【硬性配比要求】：在归纳整理的 {len(results)} 篇论文中，必须以**中文学术论文/中文编译论文为主**（包含中文标题翻译、中文摘要、中文作者），**英文原版论文保留最多不超过 3 篇**，其余全部呈现为符合国内学术综述规范的中文文献格式。\n\n"
        f"1. **海量论文对比大表**（包含：序号、论文中文/英文标题、发表时间、arXiv/DOI 链接、核心方法与技术亮点）；\n"
        f"2. **技术路线分类归纳**（将这 {len(results)} 篇论文划分为 3-4 个主要技术分支/分类，分析各分支的解决思路与代表性工作）；\n"
        f"3. **优缺点与适用场景对比**；\n"
        f"4. **针对【{txt}】研究的最新洞察与结论**；\n"
        f"5. **参考文献著录列表（依据 GB/T 7714-2015 格式）**：按国标格式输出所有 {len(results)} 篇论文的标准著录清单（中文论文按中文国标格式，英文论文按英文国标格式，例如：[1] 张三, 李四, 王五. 检索增强生成与长上下文大模型优化综述[J/OL]. 计算机学报, 2024: 1-15...）。"
    )

    gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
        inputs=i_say,
        inputs_show_user=f"检索并归纳对比 {len(results)} 篇论文（中文为主/最多3篇英文/GB/T 7714-2015）：{txt}",
        llm_kwargs=llm_kwargs,
        chatbot=chatbot,
        history=[],
        sys_prompt="你是一位顶尖的中国计算机学会（CCF）资深学术专家。请根据传入的论文元数据，输出以中文为主的论文对比矩阵、分类综述以及严格符合 GB/T 7714-2015 国标格式的参考文献列表。"
    )

    history.extend([f"arXiv 检索({len(results)}篇): {txt}", gpt_say])
    path = write_history_to_file(history)
    promote_file_to_downloadzone(path, chatbot=chatbot)
    yield from update_ui(chatbot=chatbot, history=history, msg="完成")
