"""MẪU 1 — ReAct (Reason + Act).

LLM tự quyết định từng bước: Suy nghĩ → Hành động (tool) → Quan sát → lặp lại.
Không có kế hoạch trước; linh hoạt nhưng khó đoán số bước.

    intake → agent ⇄ guard_tools
               └─(hết tool call)→ verify ─✓→ END
                                     └✗→ feedback → agent (tối đa max_verify_retries) / handoff
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from harness.nodes import (HARNESS_RULES, budget_handoff, guarded_tools_node, handoff_node, intake_node,
                           over_budget, request_brief, route_after_intake, verify_node)
from shared.llm import get_llm
from shared.state import AgentState, ev
from shared.tools import ALL_TOOLS, TOOLS_BY_NAME

MODE = "ReAct"
ALLOWED_TOOLS = [t.name for t in ALL_TOOLS]
DESCRIPTION = "LLM lặp Suy nghĩ → Hành động → Quan sát, tự chọn tool ở từng bước, không lập kế hoạch trước."
DIAGRAM = {
    "nodes": [("intake", "Intake\n(trích & kiểm ràng buộc)"), ("agent", "Agent LLM\n(Thought → Action)"),
              ("guard_tools", "Guard + Tools\n(quyền/ràng buộc/duyệt)"), ("verify", "Verify\n(tiêu chí bằng code)"),
              ("feedback", "Feedback\n(báo tiêu chí trượt)"), ("handoff", "Handoff"), ("END", "END")],
    "edges": [("intake", "agent", ""), ("agent", "guard_tools", "tool_calls"), ("guard_tools", "agent", "observation"),
              ("agent", "verify", "trả lời cuối"), ("verify", "END", "✓"), ("verify", "feedback", "✗"),
              ("feedback", "agent", ""), ("intake", "handoff", ""), ("agent", "handoff", ""),
              ("guard_tools", "handoff", ""), ("verify", "handoff", "hết lượt")],
}

PROMPT = (
    "Bạn là agent đặt vé máy bay theo mẫu ReAct. Ở MỖI lượt, viết 1-2 câu bắt đầu bằng 'Suy nghĩ:' "
    "giải thích bạn định làm gì và vì sao, rồi gọi ĐÚNG MỘT tool. Đọc kết quả rồi lặp lại. "
    "Khi đã xuất vé xong, trả lời tổng kết ngắn và KHÔNG gọi tool nữa.\n"
)


def agent_node(state: AgentState) -> dict:
    if over_budget(state):
        return {**budget_handoff(state), "events": [ev("agent", "error", "Hết ngân sách bước", None)]}
    llm = get_llm().bind_tools([TOOLS_BY_NAME[n] for n in state["allowed_tools"]])
    new = [] if state.get("messages") else [HumanMessage("Hãy đặt vé đúng theo yêu cầu trên.")]
    sys = SystemMessage(PROMPT + request_brief(state) + "\n" + HARNESS_RULES)
    ai = llm.invoke([sys, *(state.get("messages") or []), *new])
    events = []
    if ai.content:
        events.append(ev("agent", "thought", "Suy nghĩ", ai.content))
    for c in ai.tool_calls:
        events.append(ev("agent", "action", f"Hành động: {c['name']}", c["args"]))
    if not ai.tool_calls:
        events.append(ev("agent", "info", "Agent cho rằng đã xong → chuyển Verify", None))
    return {"messages": [*new, ai], "llm_calls": state.get("llm_calls", 0) + 1, "events": events}


def feedback_node(state: AgentState) -> dict:
    failed = state["completion"]["failed"]
    text = ("HARNESS: Chưa đạt tiêu chí hoàn thành:\n" + "\n".join(f"- {c['id']} {c['criterion']}" for c in failed)
            + "\nHãy tiếp tục xử lý cho đạt, hoặc gọi request_handoff nếu không thể.")
    return {"messages": [HumanMessage(text)], "verify_retries": state.get("verify_retries", 0) + 1,
            "events": [ev("feedback", "guard", "Trả tiêu chí trượt về cho agent", text)]}


def route_agent(state: AgentState) -> str:
    if state.get("handoff_reason"):
        return "handoff"
    return "guard_tools" if state["messages"][-1].tool_calls else "verify"


def route_tools(state: AgentState) -> str:
    return "handoff" if state.get("handoff_reason") else "agent"


def route_verify(state: AgentState) -> str:
    if state["completion"]["done"]:
        return END
    if state.get("verify_retries", 0) < state["policy"]["limits"]["max_verify_retries"]:
        return "feedback"
    return "handoff"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("intake", intake_node)
    g.add_node("agent", agent_node)
    g.add_node("guard_tools", guarded_tools_node)
    g.add_node("verify", verify_node)
    g.add_node("feedback", feedback_node)
    g.add_node("handoff", handoff_node)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake("agent"), ["agent", "handoff"])
    g.add_conditional_edges("agent", route_agent, ["guard_tools", "verify", "handoff"])
    g.add_conditional_edges("guard_tools", route_tools, ["agent", "handoff"])
    g.add_conditional_edges("verify", route_verify, [END, "feedback", "handoff"])
    g.add_edge("feedback", "agent")
    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer)
