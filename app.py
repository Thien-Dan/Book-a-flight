"""UI chung cho 3 agent đặt vé máy bay. Chạy: streamlit run app.py"""
from __future__ import annotations

import importlib
import json
import os
import uuid
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import Lai.agent as lai_agent
import ReAct.agent as react_agent
from harness.constraints import load_policy
from shared import llm as _llm  # noqa: F401  (nạp .env)

# Tên thư mục có dấu '-' nên không dùng được câu lệnh import thường
pte_agent = importlib.import_module("Plan-then-Execute.agent")

st.set_page_config(page_title="Agent đặt vé máy bay", page_icon="✈️", layout="wide")

MODES = {"ReAct": react_agent, "Plan-then-Execute": pte_agent, "Lai (Hybrid)": lai_agent}
KIND_ICON = {"thought": "💭", "action": "🔧", "observation": "👁️", "guard": "🛡️", "approval": "✅", "error": "❌",
             "plan": "🗺️", "verify": "🧪", "handoff": "📦", "info": "ℹ️", "ok": "✔️", "constraint": "📐"}
STATUS_ICON = {"pending": "⏳", "running": "▶️", "done": "✅", "failed": "❌"}


@st.cache_resource
def get_graph(mode: str):
    return MODES[mode].build_graph(InMemorySaver())


ss = st.session_state
for k, v in {"chat": [], "turns": [], "thread_id": None, "run_mode": None, "pending": None,
             "path": [], "last_status": None}.items():
    ss.setdefault(k, v)


def new_conversation():
    ss.chat, ss.turns, ss.thread_id, ss.run_mode, ss.pending, ss.path, ss.last_status = [], [], None, None, None, [], None


# ------------------------------- SIDEBAR -------------------------------
with st.sidebar:
    st.header("✈️ Agent đặt vé")
    mode = st.radio("Chế độ agent", list(MODES), disabled=bool(ss.pending),
                    help="Chọn mẫu thiết kế. Không đổi được khi đang chờ phê duyệt.")
    st.caption(MODES[mode].DESCRIPTION)

    base = load_policy()
    st.subheader("🛡️ Harness (override policy.json)")
    threshold = st.number_input("Tự duyệt thanh toán ≤ (VND)", 0, 200_000_000,
                                base["permissions"]["auto_approve_payment_max_vnd"], step=1_000_000,
                                help="0 = mọi thanh toán đều cần bạn phê duyệt")
    max_llm = st.number_input("Số lượt LLM tối đa", 3, 60, base["limits"]["max_llm_calls"])
    max_tools = st.number_input("Số tool call tối đa", 3, 60, base["limits"]["max_tool_calls"])
    max_stops = st.number_input("Điểm dừng tối đa (policy)", 0, 2, base["limits"]["max_stops"])
    overrides = {"permissions.auto_approve_payment_max_vnd": int(threshold), "limits.max_llm_calls": int(max_llm),
                 "limits.max_tool_calls": int(max_tools), "limits.max_stops": int(max_stops)}
    with st.expander("Xem policy.json (ràng buộc dạng dữ liệu)"):
        st.json(load_policy(overrides), expanded=False)

    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key or key.startswith("sk-your"):
        st.error("Chưa có DEEPSEEK_API_KEY trong .env")
    else:
        st.success(f"DeepSeek: {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")
    st.button("🔄 Cuộc hội thoại mới", on_click=new_conversation, width="stretch")

    st.subheader("💡 Câu mẫu")
    d1 = (date.today() + timedelta(days=7)).strftime("%d/%m/%Y")
    d2 = (date.today() + timedelta(days=14)).strftime("%d/%m/%Y")
    samples = [
        f"Đặt vé Hà Nội đi Sài Gòn ngày {d1} cho 2 người: Nguyễn Văn An và Trần Thị Bình. Bay thẳng, ngân sách tổng 6 triệu, ưu tiên rẻ nhất.",
        f"Cho mình 1 vé Đà Nẵng → Phú Quốc ngày {d2}, tên Lê Minh Châu, cất cánh sau 9h sáng, tránh Vietjet.",
        f"Đặt vé business từ HAN đi ICN ngày {d2} cho Phạm Quốc Huy, ngân sách 3 triệu.",
        "Mình muốn bay ra Hà Nội tuần sau.",
    ]
    for i, s in enumerate(samples):
        if st.button(s, key=f"sample{i}", width="stretch", disabled=bool(ss.pending)):
            ss.queued = s


# ------------------------------- BẢNG XỬ LÝ (TOP) -------------------------------

def to_dot(diagram: dict, path: list[str], active: str | None) -> str:
    visited = set(path)
    walked = set(zip(path, path[1:]))
    out = ['digraph G {', 'rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.35; ranksep=0.45;',
           'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10 color="#94a3b8" fillcolor="#eef2f7" fontcolor="#1e293b"];',
           'edge [color="#94a3b8" fontsize=8 fontcolor="#64748b" arrowsize=0.7];']
    for nid, label in diagram["nodes"]:
        attrs = {"label": label.replace("\n", "\\n")}
        if nid == "END":
            attrs.update(shape="doublecircle", fillcolor="#1e293b", fontcolor="white", label="END", fontsize="8")
        if nid == "handoff":
            attrs.update(fillcolor="#fde68a" if nid in visited else "#fef3c7")
        if nid in visited:
            attrs.update(fillcolor="#bbf7d0", color="#16a34a")
        if nid == active:
            attrs.update(fillcolor="#f97316", color="#c2410c", fontcolor="white", penwidth="2")
        out.append(f'"{nid}" [' + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "];")
    for a, b, lbl in diagram["edges"]:
        hot = (a, b) in walked or (b == "END" and a == (path[-1] if path else None) and active is None)
        style = 'color="#16a34a" penwidth=2' if hot else ""
        out.append(f'"{a}" -> "{b}" [label="{lbl}" {style}];')
    out.append("}")
    return "\n".join(out)


def detail_text(d) -> str:
    if d is None:
        return ""
    s = d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
    return s if len(s) < 400 else s[:397] + "..."


def render_panel(ph, mode_key: str, vals: dict, running: bool):
    mod = MODES[mode_key]
    pol = vals.get("policy") or load_policy(overrides)
    path = ss.path
    active = path[-1] if (path and running) else None
    status = "⏸️ Chờ duyệt" if ss.pending else ("⏳ Đang chạy" if running else
              {"done": "✅ Xong", "handoff": "📦 Bàn giao", "running": "…"}.get(vals.get("status"), "Chưa chạy"))
    with ph.container(border=True):
        st.markdown(f"#### 🧭 Bảng xử lý — chế độ **{mode_key}**")
        m = st.columns(5)
        m[0].metric("Trạng thái", status)
        m[1].metric("Node hiện tại", path[-1] if path else "—")
        m[2].metric("Lượt LLM", f"{vals.get('llm_calls', 0)}/{pol['limits']['max_llm_calls']}")
        m[3].metric("Tool call", f"{vals.get('tool_calls', 0)}/{pol['limits']['max_tool_calls']}")
        plan = vals.get("plan") or []
        m[4].metric("Kế hoạch", f"{sum(s['status'] == 'done' for s in plan)}/{len(plan)} bước" if plan else "—")

        left, right = st.columns([5, 6])
        with left:
            st.graphviz_chart(to_dot(mod.DIAGRAM, path, active), width="stretch")
            if path:
                st.caption("Đường đi: " + " → ".join(path[-14:]))
        with right:
            t1, t2, t3, t4 = st.tabs(["Dòng xử lý", "Kế hoạch", "Harness", "Bàn giao / Kết quả"])
            events = vals.get("events") or []
            with t1:
                if events:
                    df = pd.DataFrame([{"Giờ": e["time"], "Node": e["node"], "": KIND_ICON.get(e["kind"], "•"),
                                        "Sự kiện": e["title"], "Chi tiết": detail_text(e["detail"])} for e in reversed(events)])
                    st.dataframe(df, hide_index=True, width="stretch", height=330)
                else:
                    st.info("Chưa có sự kiện. Gửi một yêu cầu đặt vé ở khung chat bên dưới.")
            with t2:
                if plan:
                    st.dataframe(pd.DataFrame([{"#": s["id"], "TT": STATUS_ICON.get(s["status"], s["status"]),
                                                "Tool": s.get("tool", "(ReAct tự chọn)"), "Bước": s["title"],
                                                "Kết quả": detail_text(s.get("result"))} for s in plan]),
                                 hide_index=True, width="stretch", height=330)
                elif mode_key == "ReAct":
                    st.info("ReAct không lập kế hoạch trước — mỗi bước được quyết định tại chỗ (xem Dòng xử lý).")
                else:
                    st.info("Chưa có kế hoạch.")
            with t3:
                a, b = st.columns(2)
                with a:
                    st.markdown("**📐 Ràng buộc hiệu lực**")
                    st.json(vals.get("constraints") or {}, expanded=True)
                with b:
                    st.markdown("**🧪 Tiêu chí hoàn thành (code)**")
                    comp = vals.get("completion")
                    if comp:
                        st.dataframe(pd.DataFrame([{"": "✅" if c["passed"] else "❌", "Tiêu chí": f"{c['id']} {c['criterion']}"}
                                                   for c in comp["criteria"]]), hide_index=True, width="stretch")
                    else:
                        st.caption("Chưa kiểm.")
                st.markdown("**🛡️ Nhật ký kiểm quyền & ràng buộc tại tool**")
                log = vals.get("guard_log") or []
                if log:
                    st.dataframe(pd.DataFrame([{"Tool": g["tool"], "Cấp quyền": g.get("level"), "Quyết định": g["decision"].upper(),
                                                "Lý do": g["reason"], "Luật": " ".join(f"{c['id']}{'✓' if c['passed'] else '✗'}" for c in g["checks"]),
                                                "OK": "✓" if g["ok"] else "✗"} for g in log]),
                                 hide_index=True, width="stretch")
                else:
                    st.caption("Chưa có tool call.")
            with t4:
                if vals.get("handoff"):
                    st.json(vals["handoff"], expanded=1)
                elif vals.get("status") == "done":
                    st.success("Đạt toàn bộ tiêu chí hoàn thành.")
                    st.markdown(vals.get("final_answer", ""))
                else:
                    st.caption("Chưa có.")


panel = st.empty()
view_mode = ss.run_mode or mode


def current_values() -> dict:
    if not ss.thread_id or not ss.run_mode:
        return {}
    return get_graph(ss.run_mode).get_state({"configurable": {"thread_id": ss.thread_id}}).values


render_panel(panel, view_mode, current_values(), running=False)


# ------------------------------- CHAT -------------------------------
for msg in ss.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if ss.pending:
    with st.chat_message("assistant", avatar="🛡️"):
        st.warning("**Harness yêu cầu phê duyệt** trước khi agent thực hiện thao tác sau:")
        decisions = {}
        for r in ss.pending["requests"]:
            with st.container(border=True):
                st.markdown(f"**`{r['tool']}`** — cấp quyền `{r['level']}` · {r['reason']}")
                if r.get("booking"):
                    b = r["booking"]
                    st.markdown(f"Booking `{b['booking_id']}` · {b['flight_no']} {b['origin']}→{b['destination']} "
                                f"{b['date']} {b['depart_time']} · {b['cabin']} · {', '.join(b['passengers'])} · "
                                f"**{b['total_vnd']:,}đ**")
                else:
                    st.json(r["args"])
                decisions[r["id"]] = st.checkbox("Cho phép", value=True, key=f"ap_{r['id']}")
        c1, c2 = st.columns(2)
        if c1.button("✅ Xác nhận quyết định", type="primary", width="stretch"):
            ss.resume = decisions
            ss.pending = None
            st.rerun()
        if c2.button("⛔ Từ chối tất cả", width="stretch"):
            ss.resume = {k: False for k in decisions}
            ss.pending = None
            st.rerun()


def run_graph(inp):
    graph = get_graph(ss.run_mode)
    cfg = {"configurable": {"thread_id": ss.thread_id}, "recursion_limit": 120}
    try:
        for chunk in graph.stream(inp, cfg, stream_mode="updates"):
            for node, _upd in chunk.items():
                if node == "__interrupt__":
                    ss.pending = _upd[0].value
                else:
                    ss.path.append(node)
            render_panel(panel, ss.run_mode, graph.get_state(cfg).values, running=not ss.pending)
    except Exception as e:  # noqa: BLE001
        ss.chat.append({"role": "assistant", "content": f"❌ Lỗi khi chạy agent: `{e}`"})
        ss.last_status = "error"
        return
    vals = graph.get_state(cfg).values
    if ss.pending:
        ss.chat.append({"role": "assistant", "content": "⏸️ Agent tạm dừng — cần bạn phê duyệt (xem khung bên dưới)."})
        return
    ss.last_status = vals.get("status")
    ss.chat.append({"role": "assistant", "content": f"**[{ss.run_mode}]**\n\n" + (vals.get("final_answer") or "(không có kết quả)")})
    if ss.last_status == "done":
        ss.turns = []  # đặt xong thì lượt sau là yêu cầu mới


if "resume" in ss:
    run_graph(Command(resume=ss.pop("resume")))
    st.rerun()

prompt = st.chat_input("Nhập yêu cầu đặt vé…", disabled=bool(ss.pending))
prompt = prompt or ss.pop("queued", None)
if prompt and not ss.pending:
    ss.chat.append({"role": "user", "content": prompt})
    ss.turns.append(prompt)
    ss.run_mode, ss.thread_id, ss.path = mode, str(uuid.uuid4()), []
    user_input = "\n".join(f"Lượt {i + 1}: {t}" for i, t in enumerate(ss.turns))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(f"Agent {mode} đang xử lý… (theo dõi ở Bảng xử lý phía trên)"):
            run_graph({"user_input": user_input, "mode": MODES[mode].MODE, "policy": load_policy(overrides),
                       "allowed_tools": MODES[mode].ALLOWED_TOOLS, "messages": [], "llm_calls": 0, "tool_calls": 0,
                       "verify_retries": 0, "replans": 0, "booking_ids": [], "guard_log": [], "events": []})
    st.rerun()
