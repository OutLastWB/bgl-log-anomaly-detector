"""
Streamlit app: authenticated log upload, sampling, BGL parsing, and Isolation Forest anomaly detection.
Refactored for production-style layout and a reusable process_log_file() API (FastAPI-ready).
"""

from __future__ import annotations

import uuid
import io
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Literal, TypedDict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from utils.log_processor import confusion_matrix_table

MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR = Path("uploads")
API_BASE = "https://bgl-log-anomaly-detector.onrender.com"
API_URL = f"{API_BASE}/analyze"
LOGS_URL = f"{API_BASE}/logs"

SubscriptionTier = Literal["free", "pro", "business", "admin"]


class PlanCard(TypedDict):
    title: str
    price_label: str
    log_limit: str
    features: list[str]


PLAN_CATALOG: dict[str, PlanCard] = {
    "free": {
        "title": "Free",
        "price_label": "$0",
        "log_limit": "Up to **10,000** lines per analysis (reservoir / first-N sampling).",
        "features": [
            "Isolation Forest anomaly detection on your sample",
            "Visual analytics (histograms, scatter, charts)",
            "Export results as CSV",
            "Analysis history (last 10 runs on server)",
        ],
    },
    "pro": {
        "title": "Pro",
        "price_label": "$5",
        "log_limit": "Up to **50,000** lines per analysis.",
        "features": [
            "Everything in Free",
            "Higher sampling cap for larger logs",
            "Priority processing",
            "Extended analysis history",
            "Email support",
        ],
    },
    "business": {
        "title": "Business",
        "price_label": "$20",
        "log_limit": "Up to **1,000,000** lines per analysis.",
        "features": [
            "Everything in Pro",
            "Team-friendly limits and reporting",
            "SSO and audit logs",
            "Dedicated support",
        ],
    },
}


def _is_admin_username(username: str) -> bool:
    return (username or "").strip() == "admin"


# Non-admin caps are clamped on the server to SERVER_MAX_ANALYZE_LINES.
SERVER_MAX_ANALYZE_LINES = 1_000_000
ADMIN_ANALYZE_LINES = 10_000_000

TIER_MAX_LINES: dict[SubscriptionTier, int] = {
    "free": 10_000,
    "pro": 50_000,
    "business": SERVER_MAX_ANALYZE_LINES,
    "admin": ADMIN_ANALYZE_LINES,
}


def _default_subscription(username: str) -> SubscriptionTier:
    if _is_admin_username(username):
        return "admin"
    return "free"


def _effective_tier(username: str) -> SubscriptionTier:
    """Admin always has admin tier; otherwise use session subscription."""
    if _is_admin_username(username):
        return "admin"
    t = st.session_state.get("subscription", "free")
    if t in ("free", "pro", "business", "admin"):
        return t  # type: ignore[return-value]
    return "free"


def _ensure_admin_subscription_state(username: str) -> None:
    """Lock session subscription to admin and clear checkout state; no-op for other users."""
    if not _is_admin_username(username):
        return
    st.session_state["subscription"] = "admin"
    st.session_state.pop("pending_subscription_tier", None)


def _tier_max_lines(tier: SubscriptionTier) -> int:
    return TIER_MAX_LINES.get(tier, TIER_MAX_LINES["free"])


def _tier_show_advanced_charts(tier: SubscriptionTier) -> bool:
    return tier in ("pro", "business", "admin")


def _tier_show_all_graphs(tier: SubscriptionTier) -> bool:
    return tier in ("business", "admin")


def _tier_show_advanced_metrics(tier: SubscriptionTier) -> bool:
    return tier in ("pro", "business", "admin")


def _render_admin_mode_badge() -> None:
    st.info("Admin Mode")


def _build_theme_css(theme: str) -> str:
    """Single global stylesheet from theme tokens (injected via st.markdown)."""
    t = theme if theme in ("dark", "light") else "dark"
    if t == "light":
        bg_color = "#ffffff"
        text_color = "#111111"
        card_color = "#f5f5f5"
        sidebar_bg = "#f5f5f5"
        title_color = "#111111"
        body_text = "#111111"
        text_muted = "#555555"
        border = "#e0e0e0"
        input_bg = "#ffffff"
        alert_text = "#1f2937"
        header_bg = "rgba(255, 255, 255, 0.95)"
        code_bg = "#eeeeee"
        code_text = "#111111"
        button_bg = "#e0e0e0"
        button_text = "#111111"
        button_bg_hover = "#d0d0d0"
        button_border = "#c4c4c8"
        link_color = "#1d4ed8"
        card_shadow = "0 4px 14px rgba(17, 24, 39, 0.07), 0 1px 3px rgba(17, 24, 39, 0.05)"
        card_shadow_hover = "0 10px 28px rgba(17, 24, 39, 0.11), 0 2px 6px rgba(17, 24, 39, 0.06)"
        card_border_hover = "#d0d0d0"
    else:
        bg_color = "#0e1117"
        text_color = "#ffffff"
        card_color = "#262a33"
        sidebar_bg = "#1c1f26"
        title_color = "#ffffff"
        body_text = "#ffffff"
        text_muted = "#a3a8b8"
        border = "#3a3d46"
        input_bg = "#1c1f26"
        alert_text = "#f3f4f6"
        header_bg = "rgba(14, 17, 23, 0.92)"
        code_bg = "#262b36"
        code_text = "#f3f4f6"
        button_bg = "#262730"
        button_text = "#ffffff"
        button_bg_hover = "#323742"
        button_border = "#3d424d"
        link_color = "#93c5fd"
        card_shadow = "0 4px 18px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.04)"
        card_shadow_hover = "0 10px 32px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(255, 255, 255, 0.07)"
        card_border_hover = "#4b5563"

    return f"""
<style>
    :root {{
        --bg-color: {bg_color};
        --text-color: {text_color};
        --card-color: {card_color};
        --sidebar-bg: {sidebar_bg};
        --title-color: {title_color};
        --body-text: {body_text};
        --text-muted: {text_muted};
        --border: {border};
        --input-bg: {input_bg};
        --alert-text: {alert_text};
        --header-bg: {header_bg};
        --code-bg: {code_bg};
        --code-text: {code_text};
        --button-bg: {button_bg};
        --button-text: {button_text};
        --button-bg-hover: {button_bg_hover};
        --button-border: {button_border};
        --link-color: {link_color};
        --primary-button-bg: #ff4b4b;
        --primary-button-bg-hover: #e63e3e;
        --card-shadow: {card_shadow};
        --card-shadow-hover: {card_shadow_hover};
        --card-border-hover: {card_border_hover};
    }}
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] .main .block-container {{
        background-color: var(--bg-color) !important;
        color: var(--body-text) !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border) !important;
        color: var(--body-text) !important;
    }}
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {{
        color: var(--title-color) !important;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: var(--title-color) !important;
    }}
    .stApp .stMarkdown p,
    .stApp .stMarkdown li,
    .stApp .stMarkdown td,
    .stApp .stMarkdown th {{
        color: var(--body-text) !important;
    }}
    .stApp .stMarkdown strong {{
        color: var(--title-color) !important;
    }}
    .stApp .stMarkdown span:not([style*="color"]) {{
        color: inherit !important;
    }}
    .stApp label,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: var(--body-text) !important;
    }}
    .stApp .stCaption,
    .stApp [data-testid="stCaption"],
    [data-testid="stSidebar"] [data-testid="stCaption"] {{
        color: var(--text-muted) !important;
    }}
    .stApp code {{
        color: var(--code-text) !important;
        background-color: var(--code-bg) !important;
        border: 1px solid var(--border) !important;
    }}
    [data-testid="stHeader"] {{
        background-color: var(--header-bg) !important;
    }}
    hr {{
        border-color: var(--border) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--card-color) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 1rem 1.15rem !important;
        color: var(--body-text) !important;
        box-shadow: var(--card-shadow) !important;
        transition: box-shadow 0.22s ease, transform 0.22s ease, border-color 0.22s ease !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: var(--card-shadow-hover) !important;
        border-color: var(--card-border-hover) !important;
        transform: translateY(-2px) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] h1,
    [data-testid="stVerticalBlockBorderWrapper"] h2,
    [data-testid="stVerticalBlockBorderWrapper"] h3,
    [data-testid="stVerticalBlockBorderWrapper"] h4,
    [data-testid="stVerticalBlockBorderWrapper"] h5,
    [data-testid="stVerticalBlockBorderWrapper"] h6 {{
        color: var(--title-color) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] li,
    [data-testid="stVerticalBlockBorderWrapper"] span {{
        color: var(--body-text) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] strong {{
        color: var(--title-color) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"],
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] p {{
        color: var(--body-text) !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] code {{
        color: var(--code-text) !important;
        background-color: var(--code-bg) !important;
    }}
    [data-testid="stAlert"] {{
        color: var(--alert-text) !important;
    }}
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] li,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] .stMarkdown {{
        color: var(--alert-text) !important;
    }}
    [data-testid="stAlert"] a {{
        color: var(--link-color) !important;
    }}
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInputField"] input {{
        color: var(--body-text) !important;
        -webkit-text-fill-color: var(--body-text) !important;
        background-color: var(--input-bg) !important;
        caret-color: var(--body-text) !important;
        border-color: var(--border) !important;
    }}
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {{
        color: var(--body-text) !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] span {{
        color: var(--body-text) !important;
        background-color: var(--input-bg) !important;
        border-color: var(--border) !important;
    }}
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] [data-testid="stWidgetLabel"] {{
        color: var(--body-text) !important;
    }}
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span {{
        color: var(--body-text) !important;
    }}
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] span {{
        color: var(--body-text) !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        color: var(--body-text) !important;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: var(--title-color) !important;
    }}
    .stApp button[data-testid^="baseButton-"]:not([data-testid="baseButton-primary"]):not([data-testid="baseButton-link"]),
    [data-testid="stSidebar"] button[data-testid^="baseButton-"]:not([data-testid="baseButton-primary"]):not([data-testid="baseButton-link"]) {{
        background-color: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--button-border) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease !important;
    }}
    .stApp button[data-testid^="baseButton-"]:not([data-testid="baseButton-primary"]):not([data-testid="baseButton-link"]):hover,
    [data-testid="stSidebar"] button[data-testid^="baseButton-"]:not([data-testid="baseButton-primary"]):not([data-testid="baseButton-link"]):hover {{
        background-color: var(--button-bg-hover) !important;
        color: var(--button-text) !important;
        border-color: var(--button-border) !important;
    }}
    .stApp button[data-testid="baseButton-link"],
    [data-testid="stSidebar"] button[data-testid="baseButton-link"] {{
        background-color: transparent !important;
        color: var(--link-color) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
    }}
    .stApp button[data-testid="baseButton-link"]:hover,
    [data-testid="stSidebar"] button[data-testid="baseButton-link"]:hover {{
        background-color: rgba(128, 128, 128, 0.12) !important;
        color: var(--link-color) !important;
    }}
    .stApp button[data-testid="baseButton-primary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-primary"] {{
        background-color: var(--primary-button-bg) !important;
        color: #ffffff !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: background-color 0.2s ease, filter 0.2s ease !important;
    }}
    .stApp button[data-testid="baseButton-primary"]:hover,
    [data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {{
        background-color: var(--primary-button-bg-hover) !important;
        color: #ffffff !important;
    }}
    [data-testid="stFileUploader"] button {{
        background-color: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--button-border) !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        transition: background-color 0.2s ease, color 0.2s ease !important;
    }}
    [data-testid="stFileUploader"] button:hover {{
        background-color: var(--button-bg-hover) !important;
        color: var(--button-text) !important;
    }}
    .stApp button[data-testid^="baseButton-"]:disabled,
    [data-testid="stSidebar"] button[data-testid^="baseButton-"]:disabled,
    [data-testid="stFileUploader"] button:disabled {{
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }}
    [data-testid="stMetricValue"] {{
        color: var(--title-color) !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--text-muted) !important;
    }}
    [data-testid="stDataFrame"] {{
        color: var(--body-text) !important;
    }}
</style>
"""


def _apply_theme_css() -> None:
    theme = st.session_state.get("theme", "dark")
    if theme not in ("dark", "light"):
        theme = "dark"
        st.session_state["theme"] = theme
    st.markdown(_build_theme_css(theme), unsafe_allow_html=True)


def _clear_session_fully() -> None:
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)


def _session_expired_logout() -> None:
    _clear_session_fully()
    st.session_state["_login_flash"] = "Session expired. Please log in again."
    st.rerun()


def _auth_headers() -> dict[str, str]:
    token = st.session_state.get("token")
    if not token:
        _clear_session_fully()
        st.session_state["_login_flash"] = "Not authenticated. Please sign in again."
        st.rerun()
    return {"Authorization": f"Bearer {token}"}


def _raise_if_unauthorized(response: requests.Response) -> None:
    if response.status_code == 401:
        _session_expired_logout()


# ---------------------------------------------------------------------------
# Upload size helper
# ---------------------------------------------------------------------------


def _uploaded_file_size_bytes(uploaded_file: BinaryIO) -> int:
    pos = uploaded_file.tell()
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(pos)
    return size


# ---------------------------------------------------------------------------
# Upload persistence
# ---------------------------------------------------------------------------


def save_upload_to_disk(uploaded_file) -> Path:
    """Write upload bytes to uploads/ with a unique name. Returns destination path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_base = Path(uploaded_file.name).name.replace("..", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:10]
    dest = UPLOAD_DIR / f"{ts}_{uid}_{safe_base}"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_login() -> None:
    st.markdown("## Authentication")
    st.caption("Sign in or create an account to access the Log Anomaly Detection Dashboard.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            try:
                resp = requests.post(
                    f"{API_BASE}/login",
                    json={"username": username, "password": password},
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                err = data.get("error")
                token = data.get("access_token")
                if err:
                    st.error(str(err))
                elif token:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["token"] = token
                    st.session_state["subscription"] = _default_subscription(username)
                    _ensure_admin_subscription_state(username)
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except requests.exceptions.RequestException as e:
                st.error(f"Authentication backend is not reachable: {e}")
            except ValueError:
                st.error("Invalid response from authentication backend.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input(
                "Confirm Password", type="password", key="signup_confirm_password"
            )
            register_submitted = st.form_submit_button("Register")
        if register_submitted:
            if len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                try:
                    register_resp = requests.post(
                        f"{API_BASE}/register",
                        json={"username": new_username, "password": new_password},
                        timeout=20,
                    )
                    register_data = register_resp.json()
                    message = str(register_data.get("message", ""))
                    error_message = register_data.get("error")

                    if message == "User created":
                        st.success("User created successfully")
                    elif error_message:
                        st.error(str(error_message))
                    else:
                        st.error("Registration failed. Please try again.")
                except requests.exceptions.RequestException:
                    st.error("Registration failed. Please try again.")
                except ValueError:
                    st.error("Registration failed. Please try again.")


def render_charts(df: pd.DataFrame, *, show_advanced: bool) -> None:
    """Plotly visualizations (histogram, pie, scatter)."""
    if df.empty:
        st.caption("No rows to visualize.")
        return

    if not show_advanced:
        st.warning("Upgrade to Pro to unlock this feature.")
        return

    st.markdown("### Visual analytics")

    len_col = "clean_msg_length" if "clean_msg_length" in df.columns else "msg_length"
    y_title = (
        "Clean message length (characters)"
        if len_col == "clean_msg_length"
        else "Raw line length (characters)"
    )

    fig_hist = px.histogram(
        df,
        x=len_col,
        nbins=min(50, max(10, len(df) // 5)),
        title="Histogram: message length distribution",
        labels={len_col: y_title},
    )
    fig_hist.update_layout(bargap=0.05, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    if "anomaly" in df.columns:
        counts = df["anomaly"].value_counts().reindex([0, 1], fill_value=0)
        labels_pie = ["Predicted normal (0)", "Predicted anomaly (1)"]
        _pie_line = "#d4d4d8" if st.session_state.get("theme") == "light" else "#ffffff"
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=labels_pie,
                    values=[counts[0], counts[1]],
                    hole=0.35,
                    marker=dict(line=dict(color=_pie_line, width=1)),
                )
            ]
        )
        fig_pie.update_layout(title_text="Model prediction distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    if len(df) >= 1 and len_col in df.columns:
        plot_df = df.reset_index(drop=True).reset_index(names="row_index")
        plot_df["prediction_label"] = plot_df["anomaly"].map(
            {0: "Normal", 1: "Anomaly"}
        )
        fig_scatter = px.scatter(
            plot_df,
            x="row_index",
            y=len_col,
            color="prediction_label",
            color_discrete_map={"Normal": "#2ecc71", "Anomaly": "#e74c3c"},
            title="Sample index vs message length (colored by model prediction)",
            labels={
                "row_index": "Row index in sample",
                len_col: y_title,
                "prediction_label": "Model prediction",
            },
        )
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.65))
        st.plotly_chart(fig_scatter, use_container_width=True)


def render_subscription_page(username: str) -> None:
    """Frontend-only plan picker; stores choice in st.session_state['subscription']."""
    _ensure_admin_subscription_state(username)
    st.markdown("## Subscription")
    checkout_flash = st.session_state.pop("_sub_checkout_flash", None)
    if checkout_flash:
        st.success(checkout_flash)
    st.caption(
        "Compare plans and choose one for this session. Billing and server enforcement are not connected yet."
    )

    is_admin = _is_admin_username(username)
    current = st.session_state.get("subscription", _default_subscription(username))
    if not isinstance(current, str):
        current = "free"

    if is_admin:
        _render_admin_mode_badge()

    st.markdown(f"**Current plan:** `{current}`")
    st.divider()

    tier_order = ("free", "pro", "business")
    cols = st.columns(3)
    for col, tier in zip(cols, tier_order):
        plan = PLAN_CATALOG[tier]
        with col:
            with st.container(border=True):
                st.markdown(f"### {plan['title']}")
                st.markdown(f"**{plan['price_label']}** / month")
                st.markdown("##### Log limits")
                st.markdown(plan["log_limit"])
                st.markdown("##### Features")
                for line in plan["features"]:
                    st.markdown(f"- {line}")
                if is_admin:
                    st.button(
                        "Choose Plan",
                        key=f"subscription_choose_{tier}",
                        disabled=True,
                        use_container_width=True,
                        help="Admin accounts use the admin tier.",
                    )
                else:
                    if st.button(
                        "Choose Plan",
                        key=f"subscription_choose_{tier}",
                        use_container_width=True,
                    ):
                        if tier == "free":
                            st.session_state["subscription"] = "free"
                            st.session_state.pop("pending_subscription_tier", None)
                            st.rerun()
                        else:
                            st.session_state["pending_subscription_tier"] = tier
                            st.rerun()

    pending = st.session_state.get("pending_subscription_tier")
    if pending in ("pro", "business") and not is_admin:
        st.divider()
        with st.container(border=True):
            _pad_l, _pay_main, _pad_r = st.columns([1, 1.35, 1])
            with _pay_main:
                st.markdown("## Payment")
                st.markdown("")
                with st.form("payment_form"):
                    st.text_input(
                        "Card Number",
                        placeholder="",
                        key="pay_card_number",
                    )
                    st.text_input(
                        "Expiry Date",
                        placeholder="MM / YY",
                        key="pay_expiry",
                    )
                    st.text_input(
                        "CVV",
                        type="password",
                        placeholder="",
                        key="pay_cvv",
                    )
                    st.markdown("")
                    payment_submitted = st.form_submit_button(
                        "Confirm Payment",
                        type="primary",
                        use_container_width=True,
                    )
            if payment_submitted:
                st.session_state["subscription"] = pending
                st.session_state.pop("pending_subscription_tier", None)
                st.session_state["_sub_checkout_flash"] = "Payment successful."
                st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Log Anomaly Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    _apply_theme_css()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        flash = st.session_state.pop("_login_flash", None)
        if flash:
            st.error(flash)
        st.title("Log Anomaly Detection Dashboard")
        render_login()
        return

    if not st.session_state.get("token"):
        _clear_session_fully()
        st.session_state["_login_flash"] = "Not authenticated. Please sign in again."
        st.rerun()

    username = st.session_state.get("username", "user")

    if "subscription" not in st.session_state:
        st.session_state["subscription"] = _default_subscription(username)
    _ensure_admin_subscription_state(username)

    tier = _effective_tier(username)
    max_lines_effective = _tier_max_lines(tier)
    show_advanced_charts = _tier_show_advanced_charts(tier)
    show_all_graphs = _tier_show_all_graphs(tier)
    show_advanced_metrics = _tier_show_advanced_metrics(tier)

    with st.sidebar:
        st.header("Account")
        st.caption(f"Signed in as **{username}**")
        st.caption(f"Plan: **{st.session_state.get('subscription', 'free')}**")
        if _is_admin_username(username):
            _render_admin_mode_badge()
        if st.button("Log out", use_container_width=True):
            _clear_session_fully()
            st.rerun()

        if st.button("Delete History", use_container_width=True):
            st.session_state["confirm_delete_history"] = True

        if st.session_state.get("confirm_delete_history"):
            st.warning("This permanently deletes your saved analyses on the server.")
            if st.button("Confirm delete", key="sidebar_delete_confirm", type="primary"):
                try:
                    del_resp = requests.delete(
                        LOGS_URL,
                        headers=_auth_headers(),
                        timeout=30,
                    )
                    _raise_if_unauthorized(del_resp)
                    del_resp.raise_for_status()
                    st.session_state["confirm_delete_history"] = False
                    st.success("History deleted")
                except requests.exceptions.RequestException:
                    st.session_state["confirm_delete_history"] = False
                    st.error("Could not delete history. Please try again.")
                except ValueError:
                    st.session_state["confirm_delete_history"] = False
                    st.error("Invalid response from server.")
            if st.button("Cancel", key="sidebar_delete_cancel"):
                st.session_state["confirm_delete_history"] = False
                st.rerun()

        st.divider()
        st.subheader("Appearance")
        st.caption("Theme: **Dark** / **Light**")
        st.radio(
            "Theme mode",
            options=["dark", "light"],
            format_func=lambda t: "Dark" if t == "dark" else "Light",
            horizontal=True,
            key="theme",
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Navigate")
        app_page = st.radio(
            "Page",
            options=["Dashboard", "Subscription"],
            label_visibility="collapsed",
            key="app_nav_page",
        )

        st.divider()
        st.header("Sampling")
        _ml = max_lines_effective
        sampling_mode = st.radio(
            "Sampling mode",
            options=["random", "first"],
            index=0,
            format_func=lambda m, cap=_ml: (
                f"Random sample ({cap:,} lines from full upload)"
                if m == "random"
                else f"First {cap:,} lines only"
            ),
        )
        sample_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            disabled=(sampling_mode == "first"),
            help="Change the seed to draw a different random sample. Ignored for 'first N lines'.",
        )
        st.caption(
            "Random mode uses **reservoir sampling**: one pass through the upload, "
            f"only **{max_lines_effective:,}** lines kept in memory (your plan limit)."
        )
        st.divider()
        st.header("Model settings")
        contamination = st.slider(
            "Contamination",
            min_value=0.01,
            max_value=0.2,
            value=0.1,
            step=0.01,
            help="Expected fraction of anomalies for Isolation Forest.",
        )
        n_estimators = st.slider(
            "n_estimators",
            min_value=50,
            max_value=300,
            value=100,
            step=10,
            help="Number of trees in Isolation Forest.",
        )

    st.success(f"Welcome, {username}")
    if _is_admin_username(username):
        _render_admin_mode_badge()

    if app_page == "Subscription":
        render_subscription_page(username)
        return

    st.title("Log Anomaly Detection Dashboard")
    _plan_blurb = {
        "free": f"**Free plan** — up to **{max_lines_effective:,}** lines per run; basic metrics. Upgrade for advanced analytics.",
        "pro": f"**Pro plan** — up to **{max_lines_effective:,}** lines; advanced charts and evaluation metrics enabled.",
        "business": f"**Business plan** — up to **{max_lines_effective:,}** lines; all charts and timelines enabled.",
        "admin": (
            f"**Admin** — up to **{max_lines_effective:,}** lines (elevated server cap); "
            "**Admin Mode**: all graphs and metrics, no plan restrictions."
        ),
    }[tier]
    st.info(_plan_blurb)

    # --- Upload / Input ---
    st.markdown("## Upload")
    st.warning(
        "⚠️ Maximum file size is 200MB (Streamlit Cloud limit). "
        "For larger logs, please upload a smaller sample."
    )
    st.info(
        f"This app samples up to **{max_lines_effective:,}** lines per analysis (your current plan). "
        "Larger uploads are read in one pass; only the sample is sent for modeling."
    )

    input_mode = st.radio(
        "Input source",
        options=["Upload file", "Paste logs"],
        horizontal=True,
    )

    source_sig: tuple
    if input_mode == "Upload file":
        uploaded = st.file_uploader(
            "Choose a log file",
            type=["log", "txt"],
            help="Plain-text logs (e.g. BGL).",
        )
        if uploaded is None:
            st.warning("Please upload a log file to continue.")
            st.stop()

        upload_size = _uploaded_file_size_bytes(uploaded)
        uploaded.seek(0)
        if upload_size > MAX_UPLOAD_BYTES:
            size_mb = upload_size / (1024 * 1024)
            st.error(
                f"This file is about **{size_mb:.1f} MB**, which exceeds the "
                f"**{MAX_UPLOAD_MB} MB** limit supported on Streamlit Cloud. "
                "Please export a smaller slice of your log and try again."
            )
            st.stop()

        source_sig = ("upload", uploaded.name, upload_size, max_lines_effective, tier)
        if st.session_state.get("input_sig") != source_sig:
            try:
                dest = save_upload_to_disk(uploaded)
            except OSError as e:
                st.error(f"Could not save upload: {e}")
                st.stop()
            st.session_state["input_sig"] = source_sig
            st.session_state["saved_path"] = str(dest)
            st.success("File uploaded and saved successfully")
        st.caption(f"Uploaded file size: {upload_size / (1024 * 1024):.2f} MB")
    else:
        pasted_logs = st.text_area(
            "Paste log lines",
            height=220,
            placeholder="Paste one log message per line...",
        )
        if not pasted_logs.strip():
            st.warning("Please paste log lines to continue.")
            st.stop()
        source_sig = ("paste", len(pasted_logs), pasted_logs[:1000], max_lines_effective, tier)
        st.caption(f"Pasted lines: {len(pasted_logs.splitlines()):,}")
        st.session_state["input_sig"] = source_sig

    def _run_processing_via_api() -> dict:
        headers = _auth_headers()
        form_data = {"max_lines": str(max_lines_effective)}
        if input_mode == "Upload file":
            with open(st.session_state["saved_path"], "rb") as f:
                files = {"file": ("uploaded.log", f, "text/plain")}
                response = requests.post(
                    API_URL,
                    files=files,
                    data=form_data,
                    headers=headers,
                    timeout=120,
                )
        else:
            paste_file = io.BytesIO(pasted_logs.encode("utf-8"))
            files = {"file": ("pasted_logs.log", paste_file, "text/plain")}
            response = requests.post(
                API_URL,
                files=files,
                data=form_data,
                headers=headers,
                timeout=120,
            )
        _raise_if_unauthorized(response)
        response.raise_for_status()
        return response.json()

    if st.session_state.get("result_sig") != source_sig:
        try:
            with st.spinner("Sending logs to API for analysis..."):
                api_data = _run_processing_via_api()
            st.session_state["api_result"] = api_data
            st.session_state["result_sig"] = source_sig
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach FastAPI backend at `{API_URL}`: {e}")
            st.session_state.pop("api_result", None)
            st.stop()
        except ValueError:
            st.error("Backend returned an invalid JSON response.")
            st.session_state.pop("api_result", None)
            st.stop()

    # --- Analysis ---
    st.markdown("---")
    st.markdown("## Analysis")
    st.caption(
        f"**{max_lines_effective:,}** lines (plan cap) — "
        + (
            f"uniform **random** sampling (seed **{sample_seed}**)."
            if sampling_mode == "random"
            else "**first** lines of the selected input."
        )
    )
    st.caption(
        "Adjust sampling/settings in the sidebar, then click below to re-run API analysis."
    )

    rerun = st.button("Re-run analysis with current settings", type="secondary")
    if rerun:
        try:
            with st.spinner("Sending logs to API for analysis..."):
                api_data = _run_processing_via_api()
            st.session_state["api_result"] = api_data
            st.session_state["result_sig"] = source_sig
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach FastAPI backend at `{API_URL}`: {e}")
            st.stop()
        except ValueError:
            st.error("Backend returned an invalid JSON response.")
            st.stop()

    if "api_result" not in st.session_state:
        st.warning("No analysis result yet. Upload/paste logs or fix the API error above.")
        st.stop()

    api_result = st.session_state["api_result"]
    total_rows = int(api_result.get("total_rows", 0))
    n_failed = int(api_result.get("failed_parsing", 0))
    anomalies = int(api_result.get("anomalies", 0))
    sample_rows = api_result.get("sample", [])
    df = pd.DataFrame(sample_rows) if isinstance(sample_rows, list) else pd.DataFrame()
    search_query = st.text_input(
        "Search logs by message content",
        placeholder="Type keyword(s) to filter by message / clean_message...",
    ).strip()
    if search_query:
        mask = (
            df["message"].astype(str).str.contains(search_query, case=False, na=False)
            | df["clean_message"]
            .astype(str)
            .str.contains(search_query, case=False, na=False)
        )
        df_view = df[mask].copy()
        st.caption(f"Filtered rows: {len(df_view):,} / {len(df):,}")
    else:
        df_view = df.copy()

    show_only_anomalies = st.checkbox("Show only anomalies", value=False)
    if show_only_anomalies:
        df_view = df_view[df_view["anomaly"] == 1].copy()
        st.caption(f"Anomaly-only rows: {len(df_view):,}")

    # --- Results ---
    st.markdown("---")
    st.markdown("## Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("total_rows", f"{total_rows:,}")
    m2.metric("failed_parsing", f"{n_failed:,}")
    m3.metric("anomalies", f"{anomalies:,}")
    st.subheader("sample")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if n_failed:
        st.warning(
            f"{n_failed:,} line(s) did not match the standard BGL layout; "
            "those rows use fallbacks (see **clean_message**)."
        )

    parsed_cols = [
        "label",
        "timestamp",
        "node",
        "type",
        "clean_message",
        "message",
    ]

    st.subheader("Parsed preview (first 100 rows)")
    st.dataframe(
        df_view[parsed_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Message length (line chart)")
    st.line_chart(df_view[["msg_length"]])

    render_charts(
        df_view,
        show_advanced=show_advanced_charts or _is_admin_username(username),
    )

    if len(df) < 2:
        st.info("Need at least 2 log lines for Isolation Forest; export still includes parsed rows.")
    else:
        y_true = df["true_anomaly"]
        y_pred = df["anomaly"]

        st.divider()
        st.subheader("Anomaly detection (Isolation Forest)")
        st.caption(
            "Features (StandardScaler): clean message length, hour of day, node frequency, "
            "type frequency, rare score (1 / message frequency); plus label-encoded node and type. "
            f"**contamination = {contamination:.2f}**, **n_estimators = {int(n_estimators)}**"
        )

        if show_advanced_metrics:
            st.subheader("Evaluation vs BGL ground truth")
            n_normal_gt = int((y_true == 0).sum())
            n_anom_gt = int((y_true == 1).sum())
            st.caption(
                "**true_anomaly** = **0** only when the first field is **`-`**; **1** otherwise. "
                f"In this sample: **{n_normal_gt:,}** normal, **{n_anom_gt:,}** true anomalies. "
                "Positive class = **1** (anomaly)."
            )

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Accuracy", f"{acc:.4f}")
            e2.metric("Precision", f"{prec:.4f}")
            e3.metric("Recall", f"{rec:.4f}")
            e4.metric("F1-score", f"{f1:.4f}")

            st.subheader("Confusion matrix")
            st.dataframe(
                confusion_matrix_table(y_true, y_pred),
                use_container_width=True,
            )

            total = len(df)
            n_anom = int(df["anomaly"].sum())
            pct = 100.0 * n_anom / total if total else 0.0

            st.subheader("Model output summary")
            m1, m2 = st.columns(2)
            m1.metric("Predicted anomalies", f"{n_anom:,}")
            m2.metric("Predicted anomaly rate", f"{pct:.2f}%")

            if pct > 10.0:
                st.error(
                    "Automatic insight: High anomaly rate detected (>10%). "
                    "This may indicate a major incident or unstable system behavior."
                )
            elif pct > 5.0:
                st.warning(
                    "Automatic insight: Elevated anomaly rate detected (>5%). "
                    "Review abnormal patterns and monitor closely."
                )
            else:
                st.success(
                    "Automatic insight: Anomaly rate is within expected range (<=5%)."
                )

            anomalies = df_view[df_view["anomaly"] == 1]
            st.subheader("Anomalous logs")
            if anomalies.empty:
                st.success("No rows flagged as anomalies with the current settings.")
            else:
                show_cols = [
                    "timestamp",
                    "node",
                    "type",
                    "clean_message",
                    "clean_msg_length",
                    "hour_of_day",
                    "anomaly_score",
                    "anomaly",
                    "severity",
                ]
                st.dataframe(
                    anomalies[show_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("Top anomalies")
            top_sort = st.radio(
                "Rank top anomalies by",
                options=["message length", "anomaly score"],
                horizontal=True,
            )
            top_anomalies = df_view[df_view["anomaly"] == 1].copy()
            if top_anomalies.empty:
                st.info("No anomalous rows available for top-10 ranking.")
            else:
                if top_sort == "message length":
                    top_anomalies = top_anomalies.sort_values(
                        "clean_msg_length", ascending=False
                    )
                else:
                    top_anomalies = top_anomalies.sort_values(
                        "anomaly_score", ascending=True
                    )
                st.dataframe(
                    top_anomalies[
                        [
                            "timestamp",
                            "node",
                            "type",
                            "clean_message",
                            "clean_msg_length",
                            "anomaly_score",
                            "severity",
                        ]
                    ].head(10),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning("Upgrade to Pro to unlock this feature.")

        if show_all_graphs:
            st.subheader("Anomalies per hour")
            if "hour_of_day" in df.columns:
                hourly = (
                    df[df["anomaly"] == 1]
                    .groupby("hour_of_day", dropna=False)
                    .size()
                    .reset_index(name="anomaly_count")
                    .sort_values("hour_of_day")
                )
                fig_hourly = px.bar(
                    hourly,
                    x="hour_of_day",
                    y="anomaly_count",
                    title="Detected anomalies by hour of day",
                    labels={
                        "hour_of_day": "Hour of day (-1 means unknown timestamp)",
                        "anomaly_count": "Anomaly count",
                    },
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.info("Hour-based analysis is unavailable for this dataset.")
        elif len(df) >= 2:
            st.warning("Upgrade to Business to unlock this feature.")

    st.subheader("Export Results")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download processed results (CSV)",
        data=csv_data,
        file_name="anomaly_results.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Analysis History")
    load_history = st.button("Load History")

    if load_history:
        try:
            history_resp = requests.get(
                LOGS_URL,
                headers=_auth_headers(),
                timeout=30,
            )
            _raise_if_unauthorized(history_resp)
            history_resp.raise_for_status()
            history_data = history_resp.json()
            logs = history_data.get("logs", [])
            history_df = pd.DataFrame(logs) if isinstance(logs, list) else pd.DataFrame()

            if history_df.empty:
                st.info("No logs found.")
            else:
                if "created_at" in history_df.columns:
                    history_df["created_at"] = pd.to_datetime(
                        history_df["created_at"], errors="coerce"
                    )
                st.dataframe(history_df, use_container_width=True, hide_index=True)

                if "created_at" in history_df.columns and "anomalies" in history_df.columns:
                    chart_df = (
                        history_df.dropna(subset=["created_at"])
                        .sort_values("created_at")
                        .copy()
                    )
                    if show_all_graphs:
                        if not chart_df.empty:
                            fig_history = px.line(
                                chart_df,
                                x="created_at",
                                y="anomalies",
                                markers=True,
                                title="Anomalies over time",
                            )
                            st.plotly_chart(fig_history, use_container_width=True)
                    elif not chart_df.empty:
                        st.warning("Upgrade to Business to unlock this feature.")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load history from backend: {e}")
        except ValueError:
            st.error("History endpoint returned invalid JSON.")
        except Exception as e:
            st.error(f"Unexpected error while loading history: {e}")


if __name__ == "__main__":
    main()
