from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import CatchException, report_exception, update_ui, update_ui_latest_msg, write_history_to_file, promote_file_to_downloadzone
import arxiv

@CatchException
def Arxiv_Search_Assistant(txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
    """
    ArXiv 学术论文检索助手（免代理直连）
    自动在 arXiv 上检索最新论文，并由大模型生成对比分析表格
    """
    txt = txt.strip()
    if not txt:
        txt = "RAG long context 2024 2025"

    chatbot.append((f"正在检索 arXiv 最新论文：{txt}", "[Local Message] 正在连接 arXiv 官方 API 检索学术文献（免代理直连）..."))
    yield from update_ui(chatbot=chatbot, history=history)

    client = arxiv.Client()
    search = arxiv.Search(
        query=txt,
        max_results=8,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = []
    try:
        for r in client.results(search):
            results.append({
                "title": r.title,
                "authors": [a.name for a in r.authors[:4]],
                "published": r.published.strftime("%Y-%m-%d"),
                "pdf_url": r.pdf_url,
                "arxiv_id": r.entry_id.split('/')[-1],
                "summary": r.summary.replace('\n', ' ')
            })
    except Exception as e:
        chatbot.append((f"arXiv 检索出错", f"[Local Message] 检索失败: {e}"))
        yield from update_ui(chatbot=chatbot, history=history)
        return

    if not results:
        chatbot.append((f"未检索到文献", f"[Local Message] 未能根据【{txt}】检索到论文，请尝试调整关键词（建议使用英文组合，如 RAG long context）。"))
        yield from update_ui(chatbot=chatbot, history=history)
        return

    yield from update_ui_latest_msg(lastmsg=f"成功检索到 {len(results)} 篇相关 arXiv 论文，大模型正在归纳对比...", chatbot=chatbot, history=history, delay=1)

    i_say = (
        f"以下是从 arXiv 检索到的最新论文元数据（包含标题、作者、日期、链接和 Abstract）：\n\n{str(results)}\n\n"
        f"请针对检索主题【{txt}】完成以下输出：\n"
        f"1. **论文汇总对比表格**（包含：论文标题、发表日期、arXiv ID/链接、核心方法概述）；\n"
        f"2. **技术路线归纳与对比**（分析这些论文在方法、架构及长上下文 RAG 优化上的共同点与主要差异）；\n"
        f"3. **研究启发与建议**。"
    )

    gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(
        inputs=i_say,
        inputs_show_user=f"检索并对比分析 arXiv 论文：{txt}",
        llm_kwargs=llm_kwargs,
        chatbot=chatbot,
        history=[],
        sys_prompt="你是一位顶尖的资深学术专家。请根据传入的 arXiv 论文信息进行精细梳理，输出美观的 Markdown 表格与深度的技术对比归纳。"
    )

    history.extend([f"arXiv 检索: {txt}", gpt_say])
    path = write_history_to_file(history)
    promote_file_to_downloadzone(path, chatbot=chatbot)
    yield from update_ui(chatbot=chatbot, history=history, msg="完成")
