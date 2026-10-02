"""MẪU 2 — Plan-then-Execute.

Planner lập TOÀN BỘ kế hoạch một lần (mỗi bước gắn đúng 1 tool). Harness kiểm kế hoạch bằng code.
Executor chạy tuần tự, mỗi bước LLM CHỈ được gọi đúng tool của bước đó. Không lập lại kế hoạch:
bước nào lỗi → bàn giao. Dễ đoán, dễ kiểm soát, nhưng kém linh hoạt.

    intake → planner → check_plan → executor → guard_tools → record ─(còn bước)→ executor
                                                                   └(hết bước)→ verify → END / handoff
"""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from harness.nodes import (budget_handoff, guarded_tools_node, handoff_node, intake_node, over_budget,
                           request_brief, route_after_intake, verify_node)
from shared.llm import get_llm
from shared.state import AgentState, ev
from shared.tools import TOOLS_BY_NAME

MODE = "Plan-then-Execute"
PLAN_TOOLS = ["search_flights", "get_flight_details", "hold_booking", "process_payment", "issue_ticket", "get_booking"]
ALLOWED_TOOLS = PLAN_TOOLS + ["request_handoff"]
DESCRIPTION = "Lập kế hoạch đầy đủ một lần → kiểm kế hoạch bằng code → thực thi tuần tự từng bước, không re-plan."
DIAGRAM = {
    "nodes": [("intake", "Intake"), ("planner", "Planner LLM\n(lập toàn bộ plan)"), ("check_plan", "Check plan\n(bằng code)"),
              ("executor", "Executor LLM\n(1 bước = 1 tool)"), ("guard_tools", "Guard + Tools"),
              ("record", "Record\n(đánh dấu bước)"), ("verify", "Verify"), ("handoff", "Handoff"), ("END", "END")],
    "edges": [("intake", "planner", ""), ("planner", "check_plan", ""), ("check_plan", "executor", "hợp lệ"),
              ("executor", "guard_tools", ""), ("guard_tools", "record", ""), ("record", "executor", "bước tiếp"),
              ("record", "verify", "hết bước"), ("verify", "END", "✓"), ("verify", "handoff", "✗"),
              ("check_plan", "handoff", "không hợp lệ"), ("record", "handoff", "bước lỗi"), ("intake", "handoff", "")],
}


class PlanStep(BaseModel):
    tool: Literal["search_flights", "get_flight_details", "hold_booking", "process_payment", "issue_ticket", "get_booking"]
    goal: str = Field(description="Mục tiêu cụ thể của bước, VD 'Giữ chỗ chuyến rẻ nhất thoả ràng buộc'")


class Plan(BaseModel):
    rationale: str = Field(description="Giải thích ngắn vì sao lập kế hoạch như vậy")
    steps: list[PlanStep]


def planner_node(state: AgentState) -> dict:
    sys = ("Bạn là PLANNER cho agent đặt vé máy bay. Lập kế hoạch ĐẦY ĐỦ từ đầu đến khi xuất vé, mỗi bước dùng đúng 1 tool. "
           f"Tool có thể dùng: {', '.join(PLAN_TOOLS)}. Kế hoạch sẽ được thực thi y nguyên, không sửa được giữa chừng.\n"
           + request_brief(state))
    plan = get_llm().with_structured_output(Plan, method="function_calling").invoke(
        [SystemMessage(sys), HumanMessage("Lập kế hoạch.")])
    steps = [{"id": i + 1, "title": s.goal, "tool": s.tool, "status": "pending", "result": None} for i, s in enumerate(plan.steps)]
    return {"plan": steps, "step_idx": 0, "llm_calls": state.get("llm_calls", 0) + 1,
            "events": [ev("planner", "plan", f"Lập kế hoạch {len(steps)} bước", {"lý do": plan.rationale,
                          "các bước": [f"{s['id']}. [{s['tool']}] {s['title']}" for s in steps]})]}


def check_plan_node(state: AgentState) -> dict:
    """Kiểm kế hoạch bằng code trước khi cho chạy."""
    tools = [s["tool"] for s in state["plan"]]
    errs = []
    if not 3 <= len(tools) <= 8:
        errs.append("Kế hoạch phải có 3–8 bước")
    for need in ("search_flights", "hold_booking", "process_payment", "issue_ticket"):
        if need not in tools:
            errs.append(f"Thiếu bước {need}")
    order = ["search_flights", "hold_booking", "process_payment", "issue_ticket"]
    idx = [tools.index(t) for t in order if t in tools]
    if idx != sorted(idx):
        errs.append("Sai thứ tự: phải search → hold → payment → ticket")
    # Sau khi xuất vé chỉ được phép có bước CHỈ ĐỌC (VD get_booking để xác nhận lại)
    levels = state["policy"]["permissions"]["tool_levels"]
    if "issue_ticket" in tools:
        last = len(tools) - 1 - tools[::-1].index("issue_ticket")
        if any(levels.get(t) != "read" for t in tools[last + 1:]):
            errs.append("Sau issue_ticket chỉ được có bước chỉ đọc")
    if errs:
        return {"handoff_reason": "PLAN_INVALID", "handoff_detail": "\n".join(f"- {e}" for e in errs),
                "events": [ev("check_plan", "error", "Kế hoạch KHÔNG hợp lệ", errs)]}
    return {"events": [ev("check_plan", "ok", "Kế hoạch hợp lệ (thứ tự & bước bắt buộc)", tools)]}


def executor_node(state: AgentState) -> dict:
    if over_budget(state):
        return budget_handoff(state)
    plan, i = [dict(s) for s in state["plan"]], state["step_idx"]
    step = plan[i]
    step["status"] = "running"
    done = [{"bước": s["id"], "tool": s["tool"], "kết quả": s["result"]} for s in plan[:i]]
    sys = (f"Bạn là EXECUTOR. Chỉ thực hiện BƯỚC {step['id']}: \"{step['title']}\" bằng tool `{step['tool']}`. "
           "Điền tham số từ yêu cầu và kết quả các bước trước. Khi chọn chuyến phải thoả MỌI ràng buộc "
           "(ưu tiên rẻ nhất nếu khách không nói gì khác). Nếu không có lựa chọn nào thoả, gọi request_handoff.\n"
           + request_brief(state) + "\nKẾT QUẢ CÁC BƯỚC TRƯỚC:\n" + json.dumps(done, ensure_ascii=False))
    llm = get_llm().bind_tools([TOOLS_BY_NAME[step["tool"]], TOOLS_BY_NAME["request_handoff"]], tool_choice="required")
    ai = llm.invoke([SystemMessage(sys), HumanMessage(f"Thực hiện bước {step['id']}.")])
    events = [ev("executor", "action", f"Bước {step['id']}/{len(plan)}: {c['name']}", c["args"]) for c in ai.tool_calls]
    return {"plan": plan, "messages": [ai], "llm_calls": state.get("llm_calls", 0) + 1, "events": events}


def record_node(state: AgentState) -> dict:
    plan, i = [dict(s) for s in state["plan"]], state["step_idx"]
    step = plan[i]
    res = [r for r in state.get("last_results") or [] if r["tool"] == step["tool"]]
    ok = bool(res) and all(r["ok"] for r in res)
    step["status"] = "done" if ok else "failed"
    step["result"] = res[-1]["data"] if res else None
    out = {"plan": plan, "step_idx": i + 1,
           "events": [ev("record", "ok" if ok else "error", f"Bước {step['id']} {'HOÀN THÀNH' if ok else 'THẤT BẠI'}", step["title"])]}
    if not ok and not state.get("handoff_reason"):
        err = (res[-1]["data"].get("error") if res else "Executor không gọi đúng tool của bước")
        out.update(handoff_reason="STEP_FAILED", handoff_detail=f"Bước {step['id']} ({step['tool']}) thất bại: {err}. "
                                                               "Mẫu Plan-then-Execute không lập lại kế hoạch.")
    return out


def route_record(state: AgentState) -> str:
    if state.get("handoff_reason"):
        return "handoff"
    return "executor" if state["step_idx"] < len(state["plan"]) else "verify"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for name, fn in [("intake", intake_node), ("planner", planner_node), ("check_plan", check_plan_node),
                     ("executor", executor_node), ("guard_tools", guarded_tools_node), ("record", record_node),
                     ("verify", verify_node), ("handoff", handoff_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake("planner"), ["planner", "handoff"])
    g.add_edge("planner", "check_plan")
    g.add_conditional_edges("check_plan", lambda s: "handoff" if s.get("handoff_reason") else "executor", ["executor", "handoff"])
    g.add_conditional_edges("executor", lambda s: "handoff" if s.get("handoff_reason") else "guard_tools", ["guard_tools", "handoff"])
    g.add_edge("guard_tools", "record")
    g.add_conditional_edges("record", route_record, ["executor", "verify", "handoff"])
    g.add_conditional_edges("verify", lambda s: END if s["completion"]["done"] else "handoff", [END, "handoff"])
    g.add_edge("handoff", END)
    return g.compile(checkpointer=checkpointer)
