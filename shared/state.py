"""State dùng chung cho cả 3 agent (ReAct, Plan-then-Execute, Lai)."""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_input: str
    mode: str
    policy: dict                 # policy.json (+ override từ UI) -> ràng buộc dạng dữ liệu
    allowed_tools: list[str]     # tool mà chế độ này được phép dùng
    request: dict                # TripRequest đã chuẩn hoá
    constraints: dict            # ràng buộc hiệu lực (gộp request + policy)
    plan: list[dict]             # [{id, title, tool?, status, result}]
    step_idx: int
    step_iters: int
    replans: int
    llm_calls: int
    tool_calls: int
    verify_retries: int
    booking_ids: list[str]
    booking_id: str | None
    last_results: list[dict]
    guard_log: Annotated[list, operator.add]
    events: Annotated[list, operator.add]
    completion: dict
    status: str                  # running | done | handoff
    handoff_reason: str | None
    handoff_detail: str
    handoff: dict
    final_answer: str


def ev(node: str, kind: str, title: str, detail: Any = None) -> dict:
    """Một sự kiện để hiển thị trên bảng xử lý của UI."""
    return {"time": datetime.now().strftime("%H:%M:%S"), "node": node, "kind": kind, "title": title, "detail": detail}
