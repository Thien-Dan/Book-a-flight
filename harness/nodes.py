"""Các node harness dùng chung cho cả 3 mẫu agent.

    intake  ->  [lõi agent: ReAct / Plan-then-Execute / Lai]  ->  verify  ->  END
                         │  mọi tool call đều đi qua guard_tools        │
                         └──────────────► handoff ◄──────────────────────┘
"""
from __future__ import annotations

import json
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from harness.completion import check_completion, pick_booking
from harness.constraints import (TripRequest, check_tool_constraints, effective_constraints,
                                 normalize_request, validate_request)
from harness.handoff import build_handoff, handoff_message
from harness.permissions import approval_card, check_permission
from shared import mock_db as db
from shared.llm import get_llm
from shared.state import ev
from shared.tools import TOOLS_BY_NAME

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


# ------------------------------- INTAKE -------------------------------

def intake_node(state: dict) -> dict:
    today = date.today()
    sys = (
        "Bạn trích yêu cầu đặt vé máy bay MỘT CHIỀU từ hội thoại thành dữ liệu có cấu trúc. "
        f"Hôm nay là {WEEKDAYS[today.weekday()]}, {today.isoformat()}. Quy đổi ngày tương đối sang YYYY-MM-DD. "
        f"Sân bay hỗ trợ: {json.dumps(db.AIRPORTS, ensure_ascii=False)}. "
        "Hãng: VN=Vietnam Airlines, VJ=Vietjet, QH=Bamboo, VU=Vietravel. "
        "Tiền: '3 triệu'=3000000, '2tr5'=2500000. Ngân sách 'mỗi người' thì nhân với số khách để ra tổng. "
        "KHÔNG bịa thông tin khách không nói (để null/rỗng). Nếu có nhiều lượt, lượt sau ghi đè/bổ sung lượt trước."
    )
    extractor = get_llm().with_structured_output(TripRequest, method="function_calling")
    req = normalize_request(extractor.invoke([SystemMessage(sys), HumanMessage(state["user_input"])]))
    policy = state["policy"]
    issues = validate_request(req, policy)
    cons = effective_constraints(req, policy)
    events = [ev("intake", "info", "Trích yêu cầu → dữ liệu (TripRequest)", req),
              ev("intake", "constraint", "Ràng buộc hiệu lực (request ⊕ policy.json)", cons)]
    out = {"request": req, "constraints": cons, "llm_calls": state.get("llm_calls", 0) + 1,
           "booking_ids": [], "status": "running"}
    if issues:
        events.append(ev("intake", "error", f"Yêu cầu chưa hợp lệ ({len(issues)} vấn đề)", issues))
        missing = all(i["severity"] == "missing" for i in issues)
        out.update(handoff_reason="MISSING_INFO" if missing else "CONSTRAINT_VIOLATION",
                   handoff_detail="\n".join(f"- {i['message']}" for i in issues))
    else:
        events.append(ev("intake", "ok", "Yêu cầu hợp lệ theo policy", None))
    out["events"] = events
    return out


def route_after_intake(next_node: str):
    def _route(state: dict) -> str:
        return "handoff" if state.get("handoff_reason") else next_node
    return _route


# ------------------------------- GUARD (tool boundary) -------------------------------

def guarded_tools_node(state: dict) -> dict:
    """Mọi tool call của LLM đều phải đi qua đây: hạn mức → kiểm quyền → ràng buộc → phê duyệt → thực thi."""
    ai: AIMessage = state["messages"][-1]
    policy = state["policy"]
    max_tools = policy["limits"]["max_tool_calls"]
    used = state.get("tool_calls", 0)
    decisions = []
    for i, call in enumerate(ai.tool_calls):
        name, args = call["name"], call.get("args") or {}
        d = {"call": call, "tool": name, "args": args, "checks": []}
        if used + i >= max_tools:
            d.update(decision="block", reason=f"Vượt hạn mức {max_tools} tool call/phiên", budget=True)
        else:
            perm = check_permission(name, args, state)
            d["permission"] = perm
            if perm["decision"] == "deny":
                d.update(decision="block", reason=perm["reason"])
            else:
                d["checks"] = check_tool_constraints(name, args, state)
                bad = [c for c in d["checks"] if not c["passed"]]
                if bad:
                    d.update(decision="block", reason="; ".join(f"[{c['id']}] {c['message']}" for c in bad))
                else:
                    d.update(decision="ask" if perm["decision"] == "ask" else "allow", reason=perm["reason"])
        decisions.append(d)

    # Human-in-the-loop: dừng graph, chờ UI phê duyệt (không có side effect nào trước dòng này)
    asks = [d for d in decisions if d["decision"] == "ask"]
    if asks:
        answer = interrupt({"type": "approval_request",
                            "requests": [approval_card(d["call"], d["permission"]) for d in asks]}) or {}
        for d in asks:
            ok = bool(answer.get(d["call"]["id"]))
            d["decision"] = "approved" if ok else "denied"
            d["reason"] = "Người dùng đã phê duyệt" if ok else "Người dùng TỪ CHỐI"

    msgs, events, log, results = [], [], [], []
    booking_ids = list(state.get("booking_ids") or [])
    out: dict = {}
    for d in decisions:
        call, name = d["call"], d["tool"]
        if d["decision"] in ("allow", "approved"):
            try:
                content = TOOLS_BY_NAME[name].invoke(d["args"])
            except Exception as e:  # noqa: BLE001
                content = json.dumps({"ok": False, "error": f"Tool lỗi: {e}"}, ensure_ascii=False)
        else:
            tag = "DENIED_BY_USER" if d["decision"] == "denied" else "BLOCKED_BY_HARNESS"
            content = json.dumps({"ok": False, "error": f"{tag}: {d['reason']}"}, ensure_ascii=False)
        data = json.loads(content)
        ok = bool(data.get("ok"))
        bk = (data.get("booking") or {}).get("booking_id") if ok else None
        if bk and bk not in booking_ids:
            booking_ids.append(bk)
        if name == "request_handoff" and ok:
            out.update(handoff_reason="AGENT_REQUESTED", handoff_detail=f"{d['args'].get('reason', '')}\n{d['args'].get('details', '')}".strip())
        if d["decision"] == "denied":
            out.update(handoff_reason="APPROVAL_DENIED", handoff_detail=f"Không được phép chạy `{name}` {d['args']}")
        if d.get("budget"):
            out.update(handoff_reason="BUDGET_EXCEEDED", handoff_detail=d["reason"])

        msgs.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
        results.append({"tool": name, "args": d["args"], "ok": ok, "decision": d["decision"], "data": data})
        log.append({"tool": name, "args": d["args"], "decision": d["decision"], "reason": d["reason"], "ok": ok,
                    "level": (d.get("permission") or {}).get("level"),
                    "checks": [c for c in d["checks"] if not c.get("skipped")]})
        kind = {"allow": "action", "approved": "approval", "denied": "error", "block": "guard"}[d["decision"]]
        events.append(ev("guard_tools", kind, f"{name}({_short(d['args'])}) → {d['decision'].upper()}",
                         {"lý do": d["reason"], "luật đã kiểm": [f"{c['id']}:{'✓' if c['passed'] else '✗'}" for c in d["checks"] if not c.get("skipped")]}))
        events.append(ev("guard_tools", "observation", f"Kết quả {name}: {'OK' if ok else 'LỖI'}", _brief(data)))

    out.update(messages=msgs, events=events, guard_log=log, last_results=results,
               booking_ids=booking_ids, tool_calls=used + len(ai.tool_calls))
    if booking_ids:
        b = pick_booking({"booking_ids": booking_ids})
        out["booking_id"] = b["booking_id"] if b else None
    return out


def _short(args: dict) -> str:
    s = ", ".join(f"{k}={v}" for k, v in args.items())
    return s if len(s) < 90 else s[:87] + "..."


def _brief(data: dict):
    if "flights" in data:
        return {"count": data["count"], "flights": [
            f"{f['flight_id']} {f['depart_time']} {f['airline']} {f['stops']} dừng {f['total_vnd']:,}đ" for f in data["flights"]]}
    if "booking" in data and data["booking"]:
        b = data["booking"]
        return {k: b[k] for k in ("booking_id", "flight_no", "status", "total_vnd")}
    return data


# ------------------------------- BUDGET -------------------------------

def over_budget(state: dict) -> bool:
    lim = state["policy"]["limits"]
    return state.get("llm_calls", 0) >= lim["max_llm_calls"] or state.get("tool_calls", 0) >= lim["max_tool_calls"]


def budget_handoff(state: dict) -> dict:
    lim = state["policy"]["limits"]
    return {"handoff_reason": "BUDGET_EXCEEDED",
            "handoff_detail": f"Đã dùng {state.get('llm_calls', 0)}/{lim['max_llm_calls']} lượt LLM, "
                              f"{state.get('tool_calls', 0)}/{lim['max_tool_calls']} tool call mà chưa hoàn thành."}


# ------------------------------- VERIFY & HANDOFF -------------------------------

def verify_node(state: dict) -> dict:
    comp = check_completion(state)
    events = [ev("verify", "verify", f"Kiểm tiêu chí hoàn thành: {sum(c['passed'] for c in comp['criteria'])}/{len(comp['criteria'])} đạt",
                 [f"{c['id']} {'✓' if c['passed'] else '✗'} {c['criterion']}" for c in comp["criteria"]])]
    out = {"completion": comp, "events": events}
    if comp["done"]:
        out.update(status="done", final_answer=success_message(db.BOOKINGS[comp["booking_id"]]))
    return out


def handoff_node(state: dict) -> dict:
    reason = state.get("handoff_reason") or "COMPLETION_FAILED"
    detail = state.get("handoff_detail") or "\n".join(f"- {c['id']} {c['criterion']}" for c in (state.get("completion") or {}).get("failed", []))
    packet = build_handoff(state, reason, detail)
    return {"status": "handoff", "handoff": packet, "final_answer": handoff_message(packet),
            "events": [ev("handoff", "handoff", f"Bàn giao → {packet['to']} ({reason})", packet)]}


def success_message(b: dict) -> str:
    tickets = "\n".join(f"| {t['passenger']} | `{t['ticket_no']}` |" for t in b["tickets"])
    return (
        f"✅ **Đặt vé thành công** — mã đặt chỗ `{b['booking_id']}`\n\n"
        f"- Chuyến **{b['flight_no']}** ({db.AIRLINES[b['airline']]['name']}): {b['origin']} → {b['destination']}\n"
        f"- Ngày {b['date']}, cất cánh {b['depart_time']}, {b['stops']} điểm dừng, hạng {b['cabin']}\n"
        f"- Tổng tiền **{b['total_vnd']:,}đ** — thanh toán {b['payment']['method']} ({b['payment']['payment_id']})\n\n"
        f"| Hành khách | Số vé |\n|---|---|\n{tickets}"
    )


def request_brief(state: dict) -> str:
    """Tóm tắt yêu cầu + ràng buộc để nhúng vào prompt các agent."""
    return ("YÊU CẦU (dữ liệu đã kiểm):\n" + json.dumps(state["request"], ensure_ascii=False) +
            "\nRÀNG BUỘC HIỆU LỰC:\n" + json.dumps(state["constraints"], ensure_ascii=False))


HARNESS_RULES = (
    "Luật harness: mọi tool call đều bị kiểm tra bởi lớp guard; nếu nhận BLOCKED_BY_HARNESS hãy đọc lý do và "
    "điều chỉnh (VD chọn chuyến khác), đừng lặp lại y nguyên. Thanh toán có thể cần người duyệt. "
    "Quy trình chuẩn: search_flights → (get_flight_details) → hold_booking → process_payment (đúng total_vnd) → issue_ticket. "
    "Nếu không có lựa chọn nào thoả ràng buộc thì gọi request_handoff và dừng. Trả lời bằng tiếng Việt."
)
