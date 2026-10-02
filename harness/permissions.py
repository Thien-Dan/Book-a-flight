"""LỚP 2 — KIỂM QUYỀN.

Mỗi tool có một cấp quyền (read / write / payment / destructive) khai báo trong policy.json.
Cấp quyền quyết định có cần người phê duyệt (human-in-the-loop) hay không.
Ngoài ra mỗi chế độ agent chỉ được dùng một tập tool (allowlist).
"""
from __future__ import annotations

from shared import mock_db as db


def check_permission(tool: str, args: dict, state: dict) -> dict:
    perm = state["policy"]["permissions"]
    allowed_tools = state.get("allowed_tools") or []
    if tool not in allowed_tools:
        return {"decision": "deny", "level": None, "reason": f"Chế độ này không được cấp quyền dùng tool '{tool}'"}
    level = perm["tool_levels"].get(tool)
    if level is None:
        return {"decision": "deny", "level": None, "reason": f"Tool '{tool}' chưa được khai báo cấp quyền"}
    rule = perm["levels"][level]["approval"]
    if rule == "never":
        return {"decision": "allow", "level": level, "reason": "Cấp quyền không cần phê duyệt"}
    if rule == "always":
        return {"decision": "ask", "level": level, "reason": f"Thao tác '{perm['levels'][level]['label']}' luôn cần người duyệt"}
    # above_threshold (thanh toán)
    amount = int(args.get("amount_vnd") or 0)
    threshold = int(perm.get("auto_approve_payment_max_vnd") or 0)
    if amount <= threshold:
        return {"decision": "allow", "level": level, "reason": f"{amount:,}đ ≤ ngưỡng tự duyệt {threshold:,}đ"}
    return {"decision": "ask", "level": level, "reason": f"{amount:,}đ > ngưỡng tự duyệt {threshold:,}đ"}


def approval_card(call: dict, perm: dict) -> dict:
    """Thông tin hiển thị cho người duyệt."""
    card = {"id": call["id"], "tool": call["name"], "args": call["args"], "level": perm["level"], "reason": perm["reason"]}
    b = db.BOOKINGS.get(call["args"].get("booking_id", ""))
    if b:
        card["booking"] = {k: b[k] for k in ("booking_id", "flight_no", "origin", "destination", "date",
                                            "depart_time", "cabin", "passengers", "total_vnd", "status")}
    return card
