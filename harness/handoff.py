"""LỚP 4 — BÀN GIAO (HANDOFF).

Khi agent không thể/không được phép hoàn thành, harness đóng gói một "gói bàn giao" có cấu trúc
để người tiếp nhận (khách hoặc nhân viên CSKH) xử lý tiếp mà không phải đọc lại toàn bộ log.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from shared import mock_db as db

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

REASONS = {
    "MISSING_INFO":         {"to": "user",        "label": "Thiếu thông tin từ khách"},
    "CONSTRAINT_VIOLATION": {"to": "user",        "label": "Yêu cầu vi phạm chính sách"},
    "APPROVAL_DENIED":      {"to": "user",        "label": "Người dùng từ chối phê duyệt"},
    "AGENT_REQUESTED":      {"to": "user",        "label": "Agent chủ động bàn giao"},
    "BUDGET_EXCEEDED":      {"to": "human_agent", "label": "Vượt ngân sách bước/tool của agent"},
    "PLAN_INVALID":         {"to": "human_agent", "label": "Kế hoạch không qua kiểm tra"},
    "STEP_FAILED":          {"to": "human_agent", "label": "Một bước trong kế hoạch thất bại"},
    "COMPLETION_FAILED":    {"to": "human_agent", "label": "Không đạt tiêu chí hoàn thành"},
    "ERROR":                {"to": "human_agent", "label": "Lỗi hệ thống"},
}

NEXT_ACTIONS = {
    "MISSING_INFO": ["Khách bổ sung thông tin còn thiếu rồi gửi lại yêu cầu"],
    "CONSTRAINT_VIOLATION": ["Khách điều chỉnh yêu cầu cho phù hợp chính sách"],
    "APPROVAL_DENIED": ["Xác nhận lại với khách có muốn tiếp tục thanh toán không",
                        "Booking đang giữ sẽ tự hết hạn nếu không thanh toán"],
    "AGENT_REQUESTED": ["Khách nới ràng buộc (ngân sách / ngày / giờ / điểm dừng) hoặc chọn phương án khác"],
    "BUDGET_EXCEEDED": ["Nhân viên kiểm tra log và hoàn tất thủ công"],
    "PLAN_INVALID": ["Nhân viên kiểm tra kế hoạch / prompt planner"],
    "STEP_FAILED": ["Nhân viên kiểm tra bước lỗi và xử lý thủ công"],
    "COMPLETION_FAILED": ["Nhân viên đối chiếu các tiêu chí chưa đạt và xử lý booking"],
    "ERROR": ["Nhân viên kiểm tra lỗi hệ thống"],
}


def build_handoff(state: dict, reason: str, detail: str) -> dict:
    meta = REASONS.get(reason, REASONS["ERROR"])
    bookings = [db.BOOKINGS[i] for i in state.get("booking_ids") or [] if i in db.BOOKINGS]
    warnings = [f"Booking {b['booking_id']} đang GIỮ CHỖ, hết hạn {b['hold_expires_at']}"
                for b in bookings if b["status"] == "HELD"]
    warnings += [f"Booking {b['booking_id']} ĐÃ THANH TOÁN nhưng chưa xuất vé — cần xử lý gấp"
                 for b in bookings if b["status"] == "PAID"]
    packet = {
        "handoff_id": "HO" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": state.get("mode"),
        "to": meta["to"],
        "reason_code": reason,
        "reason": meta["label"],
        "detail": detail,
        "request": state.get("request"),
        "constraints": state.get("constraints"),
        "plan": state.get("plan"),
        "actions_taken": [{"tool": g["tool"], "decision": g["decision"], "ok": g.get("ok")} for g in state.get("guard_log") or []],
        "bookings": [{k: b[k] for k in ("booking_id", "flight_no", "status", "total_vnd", "passengers")} for b in bookings],
        "failed_criteria": (state.get("completion") or {}).get("failed", []),
        "warnings": warnings,
        "next_actions": NEXT_ACTIONS.get(reason, []),
        "usage": {"llm_calls": state.get("llm_calls", 0), "tool_calls": state.get("tool_calls", 0)},
    }
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / f"{packet['handoff_id']}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return packet


def handoff_message(p: dict) -> str:
    who = "bạn" if p["to"] == "user" else "nhân viên hỗ trợ"
    lines = [f"⚠️ **Bàn giao cho {who}** — {p['reason']}", "", p["detail"]]
    if p["warnings"]:
        lines += ["", *[f"- {w}" for w in p["warnings"]]]
    if p["next_actions"]:
        lines += ["", "**Bước tiếp theo:**", *[f"- {a}" for a in p["next_actions"]]]
    lines += ["", f"_Mã bàn giao: `{p['handoff_id']}` (đã lưu tại logs/)_"]
    return "\n".join(lines)
