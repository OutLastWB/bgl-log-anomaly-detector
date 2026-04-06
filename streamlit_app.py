"""
Streamlit app: authenticated log upload, sampling, BGL parsing, and Isolation Forest anomaly detection.
Refactored for production-style layout and a reusable process_log_file() API (FastAPI-ready).
"""

from __future__ import annotations

import random
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

# --- Auth (replace in production; use secrets management) ---
USERS: dict[str, str] = {
    "admin": "changeme",
    "demo": "demo123",
}

MAX_LINES = 10_000
MAX_UPLOAD_MB = 200
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR = Path("uploads")
_TS_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}(?:\.\d+)?)\b")


# ---------------------------------------------------------------------------
# File / sampling helpers (reservoir sampling unchanged)
# ---------------------------------------------------------------------------


def _uploaded_file_size_bytes(uploaded_file: BinaryIO) -> int:
    pos = uploaded_file.tell()
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(pos)
    return size


def _decode_uploaded_line(line_b: bytes) -> str:
    return line_b.decode("utf-8", errors="replace").rstrip("\n\r")


def read_first_n_uploaded(uploaded_file: BinaryIO, max_lines: int) -> list[str]:
    uploaded_file.seek(0)
    lines: list[str] = []
    for _ in range(max_lines):
        raw = uploaded_file.readline()
        if not raw:
            break
        lines.append(_decode_uploaded_line(raw))
    return lines


def reservoir_sample_uploaded(uploaded_file: BinaryIO, k: int, rng: random.Random) -> list[str]:
    """
    Uniform random sample of up to k lines from an upload in one pass (reservoir sampling).

    Only k lines are retained in memory at once; the upload is streamed via readline().
    """
    uploaded_file.seek(0)
    reservoir: list[str] = []
    i = 0
    while True:
        raw = uploaded_file.readline()
        if not raw:
            break
        line = _decode_uploaded_line(raw)
        if i < k:
            reservoir.append(line)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = line
        i += 1
    return reservoir


def load_log_sample_from_upload(
    uploaded_file: BinaryIO,
    mode: str,
    max_lines: int,
    sample_seed: int,
) -> pd.DataFrame:
    """Build ~max_lines rows from an uploaded file. mode: 'first' or 'random'."""
    if mode == "first":
        lines = read_first_n_uploaded(uploaded_file, max_lines)
    else:
        lines = reservoir_sample_uploaded(
            uploaded_file, max_lines, random.Random(sample_seed)
        )
    return pd.DataFrame({"message": lines})


# ---------------------------------------------------------------------------
# Parsing & model (Isolation Forest logic unchanged)
# ---------------------------------------------------------------------------


def _parse_timestamp(token: str) -> pd.Timestamp:
    ts = pd.to_datetime(token, format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(token, errors="coerce")
    return ts


def parse_bgl_line(line: str) -> dict:
    """
    Parse one BGL log line (space-separated).

    Typical layout:
    label seq_id YYYY.MM.DD node YYYY-MM-DD-hh.mm.ss.ffffff node RAS KERNEL INFO message...
    """
    raw = line.strip()
    empty = {
        "label": "",
        "timestamp": pd.NaT,
        "node": "",
        "type": "",
        "clean_message": "",
        "parse_ok": False,
    }
    if not raw:
        return empty

    parts = raw.split()
    n = len(parts)

    if n >= 9:
        ts = _parse_timestamp(parts[4])
        return {
            "label": parts[0],
            "timestamp": ts,
            "node": parts[3],
            "type": f"{parts[6]} {parts[7]} {parts[8]}",
            "clean_message": " ".join(parts[9:]) if n > 9 else "",
            "parse_ok": True,
        }

    label = parts[0] if n else ""
    ts = pd.NaT
    match = _TS_RE.search(raw)
    if match:
        ts = _parse_timestamp(match.group(1))

    return {
        "label": label,
        "timestamp": ts,
        "node": "",
        "type": "",
        "clean_message": raw,
        "parse_ok": False,
    }


def add_parsed_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    records = [parse_bgl_line(m) for m in df["message"]]
    extra = pd.DataFrame(records)
    parse_ok = extra["parse_ok"]
    out = pd.concat(
        [df.reset_index(drop=True), extra.drop(columns=["parse_ok"])],
        axis=1,
    )
    return out, parse_ok


def run_isolation_forest(
    df: pd.DataFrame,
    *,
    contamination: float = 0.05,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """
    Build numeric features, fit Isolation Forest, add column anomaly (1 = outlier).

    Features: clean message length, hour of day, per-sample node/type frequencies,
    label-encoded node and type.
    """
    out = df.copy()
    out["_node"] = out["node"].replace("", "__empty__").fillna("__empty__")
    out["_type"] = out["type"].replace("", "__empty__").fillna("__empty__")

    out["clean_msg_length"] = out["clean_message"].astype(str).str.len()
    out["hour_of_day"] = out["timestamp"].dt.hour
    out.loc[out["timestamp"].isna(), "hour_of_day"] = -1

    out["node_freq"] = out.groupby("_node")["_node"].transform("count")
    out["type_freq"] = out.groupby("_type")["_type"].transform("count")

    le_node = LabelEncoder()
    le_type = LabelEncoder()
    out["node_encoded"] = le_node.fit_transform(out["_node"])
    out["type_encoded"] = le_type.fit_transform(out["_type"])

    feature_cols = [
        "clean_msg_length",
        "hour_of_day",
        "node_freq",
        "type_freq",
        "node_encoded",
        "type_encoded",
    ]
    X = out[feature_cols].astype("float64")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
    )
    pred = model.fit_predict(X)
    # Lower score means "more abnormal" for Isolation Forest.
    out["anomaly_score"] = model.decision_function(X)
    out["anomaly"] = (pred == -1).astype(int)
    out["severity"] = out["anomaly"].map({0: "Normal", 1: "High Risk"})
    out = out.drop(columns=["_node", "_type"])
    return out


def _normalize_bgl_alert_token(token: str) -> str:
    if not token:
        return token
    t = token.strip().strip("\ufeff")
    for u, asc in (
        ("\u2212", "-"),
        ("\u2013", "-"),
        ("\u2014", "-"),
    ):
        t = t.replace(u, asc)
    return t


def add_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    first = (
        out["message"]
        .astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.split(n=1, expand=True)[0]
        .fillna("")
    )
    normalized = first.map(_normalize_bgl_alert_token)
    out["true_anomaly"] = (~normalized.eq("-")).astype(int)
    return out


def confusion_matrix_table(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual normal (0)", "Actual anomaly (1)"],
        columns=["Predicted normal (0)", "Predicted anomaly (1)"],
    )


# ---------------------------------------------------------------------------
# Reusable pipeline (FastAPI / batch jobs)
# ---------------------------------------------------------------------------


def process_log_file(
    file: Union[str, Path, BinaryIO],
    *,
    sampling_mode: str = "random",
    max_lines: int = MAX_LINES,
    sample_seed: int = 42,
    contamination: float = 0.05,
    n_estimators: int = 100,
) -> Tuple[pd.DataFrame, int]:
    """
    Run sampling → parse → features + Isolation Forest → ground-truth labels.

    Parameters
    ----------
    file
        Path to a log file, or a binary file-like object with ``readline`` / ``seek``
        (e.g. open(path, \"rb\") or Streamlit ``UploadedFile``).

    Returns
    -------
    (dataframe, n_parse_failed)
        ``n_parse_failed`` counts rows that did not match the standard BGL layout.
    """
    close_after = False
    fh: BinaryIO
    if isinstance(file, (str, Path)):
        fh = open(file, "rb")
        close_after = True
    else:
        fh = file
        fh.seek(0)

    try:
        df = load_log_sample_from_upload(fh, sampling_mode, max_lines, sample_seed)
        df, parse_ok = add_parsed_columns(df)
        n_failed = int((~parse_ok).sum())
        df["msg_length"] = df["message"].str.len()

        if len(df) < 2:
            df = add_ground_truth_labels(df)
            df["anomaly"] = 0
            df["anomaly_score"] = 0.0
            df["severity"] = "Normal"
        else:
            df = run_isolation_forest(
                df,
                contamination=contamination,
                n_estimators=n_estimators,
            )
            df = add_ground_truth_labels(df)
    finally:
        if close_after:
            fh.close()

    return df, n_failed


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
    st.markdown("## Login")
    st.caption("Sign in to access the Log Anomaly Detection Dashboard.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if USERS.get(username) == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")


def render_charts(df: pd.DataFrame) -> None:
    """Plotly visualizations (histogram, pie, scatter)."""
    if df.empty:
        st.caption("No rows to visualize.")
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
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=labels_pie,
                    values=[counts[0], counts[1]],
                    hole=0.35,
                    marker=dict(line=dict(color="#fff", width=1)),
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


def main() -> None:
    st.set_page_config(
        page_title="Log Anomaly Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.title("Log Anomaly Detection Dashboard")
        render_login()
        return

    with st.sidebar:
        st.markdown(f"**Signed in as** `{st.session_state.get('username', 'user')}`")
        if st.button("Log out"):
            for k in (
                "logged_in",
                "username",
                "upload_sig",
                "saved_path",
                "result_df",
                "n_parse_fail",
            ):
                st.session_state.pop(k, None)
            st.rerun()

        st.divider()
        st.header("Sampling")
        sampling_mode = st.radio(
            "Sampling mode",
            options=["random", "first"],
            index=0,
            format_func=lambda m: (
                f"Random sample ({MAX_LINES:,} lines from full upload)"
                if m == "random"
                else f"First {MAX_LINES:,} lines only"
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
            f"only **{MAX_LINES:,}** lines kept in memory."
        )
        st.divider()
        st.header("Model settings")
        contamination = st.slider(
            "Contamination",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
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

    st.title("Log Anomaly Detection Dashboard")
    st.info(
        "Free version: up to 10,000 lines. Upgrade for full dataset processing."
    )

    # --- Upload ---
    st.markdown("## Upload")
    st.warning(
        "⚠️ Maximum file size is 200MB (Streamlit Cloud limit). "
        "For larger logs, please upload a smaller sample."
    )
    st.info(
        "This app processes only up to 10,000 lines using efficient sampling, "
        "so you can upload a smaller portion of your log file."
    )

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

    sig = (uploaded.name, upload_size)
    if st.session_state.get("upload_sig") != sig:
        try:
            dest = save_upload_to_disk(uploaded)
        except OSError as e:
            st.error(f"Could not save upload: {e}")
            st.stop()
        st.session_state["upload_sig"] = sig
        st.session_state["saved_path"] = str(dest)
        saved_path = dest
        st.success("File uploaded and saved successfully")
        try:
            with st.spinner("Processing log file..."):
                df_new, n_fail_new = process_log_file(
                    saved_path,
                    sampling_mode=sampling_mode,
                    max_lines=MAX_LINES,
                    sample_seed=int(sample_seed),
                    contamination=float(contamination),
                    n_estimators=int(n_estimators),
                )
            st.session_state["result_df"] = df_new
            st.session_state["n_parse_fail"] = n_fail_new
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.session_state.pop("result_df", None)
            st.session_state.pop("n_parse_fail", None)
            st.stop()
    else:
        saved_path = Path(st.session_state["saved_path"])

    st.caption(f"Uploaded file size: {upload_size / (1024 * 1024):.2f} MB")

    # --- Analysis ---
    st.markdown("---")
    st.markdown("## Analysis")
    st.caption(
        f"**{MAX_LINES:,}** lines — "
        + (
            f"uniform **random** sampling (seed **{sample_seed}**)."
            if sampling_mode == "random"
            else "**first** lines of the saved file."
        )
    )
    st.caption(
        "Adjust sampling in the sidebar, then click below to re-run **process_log_file** on the saved upload."
    )

    rerun = st.button("Re-run analysis with current sampling settings", type="secondary")
    if rerun:
        try:
            with st.spinner("Re-processing log file..."):
                df_new, n_fail_new = process_log_file(
                    saved_path,
                    sampling_mode=sampling_mode,
                    max_lines=MAX_LINES,
                    sample_seed=int(sample_seed),
                    contamination=float(contamination),
                    n_estimators=int(n_estimators),
                )
            st.session_state["result_df"] = df_new
            st.session_state["n_parse_fail"] = n_fail_new
            st.rerun()
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()

    if "result_df" not in st.session_state:
        st.warning("No results yet. Upload a file or fix the error above.")
        st.stop()

    df = st.session_state["result_df"]
    n_failed = int(st.session_state.get("n_parse_fail", 0))
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

    # --- Results ---
    st.markdown("---")
    st.markdown("## Results")

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

    render_charts(df_view)

    if len(df) < 2:
        st.info("Need at least 2 log lines for Isolation Forest; export still includes parsed rows.")
    else:
        y_true = df["true_anomaly"]
        y_pred = df["anomaly"]

        st.divider()
        st.subheader("Anomaly detection (Isolation Forest)")
        st.caption(
            "Features: message length, hour, node/type counts in this sample, "
            f"label-encoded node and type. **contamination = {contamination:.2f}**, "
            f"**n_estimators = {int(n_estimators)}**"
        )

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

    st.subheader("Export Results")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download processed results (CSV)",
        data=csv_data,
        file_name="anomaly_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
