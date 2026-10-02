"""MẪU 3 — LAI (Plan + ReAct + Re-plan).

Planner lập kế hoạch MỨC CAO (mục tiêu, không gắn tool). Mỗi bước được một vòng ReAct nhỏ thực thi
(tự chọn tool, tự xử lý lỗi). Sau mỗi bước, Replanner đánh giá: tiếp tục / sửa kế hoạch / kết thúc / bàn giao.
Nếu Verify trượt, harness chèn bước "khắc phục" và cho chạy tiếp (trong giới hạn max_replans).

    intake → planner → step_agent ⇄ guard_tools
                          └(xong bước)→ replanner ─continue/replan→ step_agent
                                                  └finish→ verify ─✓→ END
                                                                   └✗→ fix_step → step_agent / handoff
"""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from harness.completion import pick_booking
from harness.nodes import (HARNESS_RULES, budget_handoff, guarded_tools_node, handoff_node, intake_node,
                           over_budget, request_brief, route_after_intake, verify_node)
from shared.llm import get_llm
from shared.state import AgentState, ev
from shared.tools import ALL_TOOLS, TOOLS_BY_NAME

MODE = "Lai"
ALLOWED_TOOLS = [t.name for t in ALL_TOOLS]
DESCRIPTION = "Kế hoạch mức cao + mỗi bước chạy vòng ReAct + Replanner đánh giá & sửa kế hoạch sau mỗi bước."
DIAGRAM = {
    "nodes": [("intake", "Intake"), ("planner", "Planner LLM\n(plan mức cao)"), ("step_agent", "Step agent\n(ReAct trong bước)"),
              ("guard_tools", "Guard + Tools"), ("replanner", "Replanner LLM\n(đánh giá/sửa plan)"),
              ("verify", "Verify"), ("fix_step", "Fix step\n(chèn bước khắc phục)"), ("handoff", "Handoff"), ("END", "END")],
    "edges": [("intake", "planner", ""), ("planner", "step_agent", ""), ("step_agent", "guard_tools", "tool_calls"),
              ("guard_tools", "step_agent", "observation"), ("step_agent", "replanner", "xong bước"),
              ("replanner", "step_agent", "continue/replan"), ("replanner", "verify", "finish"),
              ("verify", "END", "✓"), ("verify", "fix_step", "✗"), ("fix_step", "step_agent", ""),
              ("replanner", "handoff", "handoff"), ("guard_tools", "handoff", ""), ("intake", "handoff", ""),
              ("verify", "handoff", "hết lượt")],
}


class HighLevelPlan(BaseModel):
    rationale: str
    steps: list[str] = Field(description="2–5 bước mức cao, mỗi bước là một mục tiêu (không cần tên tool)")


class Decision(BaseModel):
    action: Literal["continue", "replan", "finish", "handoff"] = Field(
        description="continue: sang bước kế; replan: thay các bước còn lại; finish: đã xuất vé xong; handoff: không thể làm tiếp")
    reason: str
    new_steps: list[str] = Field(default_factory=list, description="Chỉ khi replan: các bước MỚI thay cho phần còn lại")


def planner_node(state: AgentState) -> dict:
    sys = "Bạn là PLANNER. Lập kế hoạch mức cao 2–5 bước để đặt vé xong (đến khi xuất vé).\n" + request_brief(state)
    p = get_llm().with_structured_output(HighLevelPlan, method="function_calling").invoke(
        [SystemMessage(sys), HumanMessage("Lập kế hoạch.")])
    plan = [{"id": i + 1, "title": t, "status": "pending", "result": None} for i, t in enumerate(p.steps)]
    return {"plan": plan, "step_idx": 0, "step_iters": 0, "replans": 0, "llm_calls": state.get("llm_calls", 0) + 1,
            "events": [ev("planner", "plan", f"Kế hoạch mức cao {len(plan)} bước",
                          {"lý do": p.rationale, "các bước": [f"{s['id']}. {s['title']}" for s in plan]})]}


def step_agent_node(state: AgentState) -> dict:
    if over_budget(state):
        return budget_handoff(state)
    plan, i, iters = [dict(s) for s in state["plan"]], state["step_idx"], state.get("step_iters", 0)
    step = plan[i]
    new = []
    if iters == 0:
        step["status"] = "running"
        new.append(HumanMessage(f"BƯỚC {step['id']}/{len(plan)}: {step['title']}\n"
                                "Dùng tool cần thiết để hoàn thành RIÊNG bước này. Xong thì trả lời 'KẾT QUẢ BƯỚC: ...' (không gọi tool)."))
    if iters >= state["policy"]["limits"]["max_iters_per_step"]:
        new.append(HumanMessage("Đã hết số vòng cho bước này, hãy tóm tắt KẾT QUẢ BƯỚC ngay, không gọi tool."))
        llm = get_llm()
    else:
        llm = get_llm().bind_tools([TOOLS_BY_NAME[n] for n in state["allowed_tools"]])
    sys = SystemMessage("Bạn là agent thực thi (vòng ReAct trong từng bước kế hoạch). Viết ngắn 'Suy nghĩ:' trước khi gọi tool.\n"
                        + request_brief(state) + "\n" + HARNESS_RULES)
    ai = llm.invoke([sys, *(state.get("messages") or []), *new])
    events = [ev("step_agent", "thought", f"[Bước {step['id']}] Suy nghĩ", ai.content)] if ai.content else []
    events += [ev("step_agent", "action", f"[Bước {step['id']}] Hành động: {c['name']}", c["args"]) for c in ai.tool_calls]
    return {"plan": plan, "messages": [*new, ai], "step_iters": iters + 1,
            "llm_calls": state.get("llm_calls", 0) + 1, "events": events}


def replanner_node(state: AgentState) -> dict:
    plan, i = [dict(s) for s in state["plan"]], state["step_idx"]
    plan[i]["status"], plan[i]["result"] = "done", state["messages"][-1].content
    b = pick_booking(state)
    sys = ("Bạn là REPLANNER. Đánh giá tiến độ sau bước vừa xong và quyết định: continue / replan / finish / handoff. "
           "Chỉ chọn finish khi vé ĐÃ XUẤT (status TICKETED). Chọn handoff nếu không còn cách thoả ràng buộc.\n"
           + request_brief(state) + "\nKẾ HOẠCH:\n" + json.dumps(plan, ensure_ascii=False)
           + "\nBOOKING HIỆN TẠI:\n" + json.dumps(b, ensure_ascii=False))
    d = get_llm().with_structured_output(Decision, method="function_calling").invoke(
        [SystemMessage(sys), HumanMessage("Quyết định bước tiếp theo.")])
    out = {"llm_calls": state.get("llm_calls", 0) + 1, "step_iters": 0}
    action = d.action
    if action == "replan" and (not d.new_steps or state.get("replans", 0) >= state["policy"]["limits"]["max_replans"]):
        action = "continue"  # harness chặn re-plan khi hết hạn mức
    if action == "continue" and i + 1 >= len(plan):
        action = "finish"
    if action == "replan":
        plan = plan[: i + 1] + [{"id": i + 2 + k, "title": t, "status": "pending", "result": None} for k, t in enumerate(d.new_steps)]
        out["replans"] = state.get("replans", 0) + 1
    if action in ("continue", "replan"):
        out["step_idx"] = i + 1
    if action == "handoff":
        out.update(handoff_reason="AGENT_REQUESTED", handoff_detail=d.reason)
    out.update(plan=plan, events=[ev("replanner", "plan", f"Replanner: {action.upper()}",
                                     {"lý do": d.reason, "bước mới": d.new_steps or None})])
    return out


def fix_step_node(state: AgentState) -> dict:
    failed = state["completion"]["failed"]
    plan = [dict(s) for s in state["plan"]]
    title = "Khắc phục tiêu chí chưa đạt: " + "; ".join(f"{c['id']} {c['criterion']}" for c in failed)
    plan.append({"id": len(plan) + 1, "title": title, "status": "pending", "result": None})
    return {"plan": plan, "step_idx": len(plan) - 1, "step_iters": 0, "replans": state.get("replans", 0) + 1,
            "events": [ev("fix_step", "guard", "Verify trượt → chèn bước khắc phục", title)]}


def route_step(state: AgentState) -> str:
    if state.get("handoff_reason"):
        return "handoff"
    return "guard_tools" if state["messages"][-1].tool_calls else "replanner"


def route_replanner(state: AgentState) -> str:
    if state.get("handoff_reason"):
        return "handoff"
    return "step_agent" if state["step_idx"] < len(state["plan"]) and state["plan"][state["step_idx"]]["status"] == "pending" else "verify"


def route_verify(state: AgentState) -> str:
    if state["completion"]["done"]:
        return END
    return "fix_step" if state.get("replans", 0) < state["policy"]["limits"]["max_replans"] else "handoff"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for name, fn in [("intake", intake_node), ("planner", planner_node), ("step_agent", step_agent_node),
                     ("guard_tools", guarded_tools_node), ("replanner", replanner_node), ("verify", verify_node),
                     ("fix_step", fix_step_node), ("handoff", handoff_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake("planner"), ["planner", "handoff"])
    g.add_edge("planner", "step_agent")
    g.add_conditional_edges("step_agent", route_step, ["guard_tools", "replanner", "handoff"])
    g.add_conditional_edges("guard_tools", lambda s: "handoff" if s.get("handoff_reason") else "step_agent", ["step_agent", "handoff"])
    g.add_conditional_edges("replanner", route_replanner, ["step_agent", "verify", "handoff"])
    g.add_conditional_edges("verify", route_verify, [END, "fix_step", "handoff"])
    g.add_edge("fix_step", "step_agent")
    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer)
