"""
Streamlit app: preview BGL logs from disk, parse lines, and run Isolation Forest anomaly detection.
"""

import random
import re
from typing import Tuple

import pandas as pd
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

LOG_PATH = r"C:\Users\bklen\OneDrive\Desktop\BGL.log\BGL.log"
MAX_LINES = 10_000

_TS_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}(?:\.\d+)?)\b")


def read_first_lines(path: str, max_lines: int) -> pd.DataFrame:
    """Stream the file line-by-line and stop after max_lines."""
    messages: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            messages.append(line.rstrip("\n\r"))
    return pd.DataFrame({"message": messages})


def reservoir_sample_lines(path: str, k: int, rng: random.Random) -> list[str]:
    """
    Uniform random sample of up to k lines from a large file in one pass.

    Uses reservoir sampling: O(k) memory, reads the whole file once (no full load).
    If the file has fewer than k lines, all lines are returned.
    """
    reservoir: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n\r")
            if i < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = line
    return reservoir


@st.cache_data(show_spinner="Loading log sample...")
def load_log_sample(path: str, mode: str, max_lines: int, sample_seed: int) -> pd.DataFrame:
    """
    Load ~max_lines log lines. mode: 'first' = head of file; 'random' = reservoir sample.
    Cached so the full file is not re-scanned on every Streamlit widget change.
    """
    if mode == "first":
        return read_first_lines(path, max_lines)
    rng = random.Random(sample_seed)
    lines = reservoir_sample_lines(path, max_lines, rng)
    return pd.DataFrame({"message": lines})


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


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
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
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )
    pred = model.fit_predict(X)
    out["anomaly"] = (pred == -1).astype(int)
    out = out.drop(columns=["_node", "_type"])
    return out


def _normalize_bgl_alert_token(token: str) -> str:
    """Map Unicode dashes / stray BOM to ASCII '-' so '-' normal lines match."""
    if not token:
        return token
    t = token.strip().strip("\ufeff")
    for u, asc in (
        ("\u2212", "-"),  # minus sign
        ("\u2013", "-"),  # en dash
        ("\u2014", "-"),  # em dash
    ):
        t = t.replace(u, asc)
    return t


def add_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    BGL ground truth from the first field of each raw log line (same as dataset spec).

    - Token '-' (after normalization) => normal => true_anomaly 0
    - Any other token (e.g. 'E', 'APPREAD') => anomaly => true_anomaly 1

    Uses the stored ``message`` line with strip + first token so leading spaces
    in the file cannot desync ground truth from the parsed ``label`` column.
    """
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
    """Human-readable confusion matrix (rows = actual, columns = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual normal (0)", "Actual anomaly (1)"],
        columns=["Predicted normal (0)", "Predicted anomaly (1)"],
    )


def main():
    st.set_page_config(page_title="BGL log preview", layout="wide")

    with st.sidebar:
        st.header("Sampling")
        sampling_mode = st.radio(
            "Sampling mode",
            options=["random", "first"],
            index=0,
            format_func=lambda m: (
                f"Random sample ({MAX_LINES:,} lines from full file)"
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
            "Random mode uses **reservoir sampling**: one pass through the file, "
            f"only **{MAX_LINES:,}** lines kept in memory. The first run may take a while on huge logs."
        )

    st.title("BGL log preview")
    if sampling_mode == "random":
        st.caption(
            f"**{MAX_LINES:,}** lines drawn by **uniform random sampling** over the full file "
            f"(seed **{sample_seed}**). Not the same as file order."
        )
    else:
        st.caption(
            f"**{MAX_LINES:,}** lines from the **start** of the file (head only)."
        )

    try:
        df = load_log_sample(LOG_PATH, sampling_mode, MAX_LINES, int(sample_seed))
    except FileNotFoundError:
        st.error(f"Log file not found. Check that this path exists:\n`{LOG_PATH}`")
        return
    except OSError as e:
        st.error(f"Could not read the log file: {e}")
        return

    with st.spinner("Parsing log lines..."):
        df, parse_ok = add_parsed_columns(df)

    failed = int((~parse_ok).sum())
    if failed:
        st.warning(
            f"{failed:,} line(s) did not match the standard BGL layout; "
            "those rows use fallbacks (see **clean_message**)."
        )

    df["msg_length"] = df["message"].str.len()

    parsed_cols = [
        "label",
        "timestamp",
        "node",
        "type",
        "clean_message",
        "message",
    ]

    st.subheader("Parsed log (first 100 rows)")
    st.dataframe(
        df[parsed_cols].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Message length (loaded lines)")
    st.line_chart(df[["msg_length"]])

    if len(df) < 2:
        st.info("Need at least 2 log lines to run anomaly detection.")
        return

    with st.spinner("Running Isolation Forest..."):
        df = run_isolation_forest(df)

    df = add_ground_truth_labels(df)
    y_true = df["true_anomaly"]
    y_pred = df["anomaly"]

    st.divider()
    st.subheader("Anomaly detection (Isolation Forest)")
    st.caption(
        "Features: message length, hour, node/type counts in this sample, "
        "label-encoded node and type. **contamination = 0.05**"
    )

    st.subheader("Evaluation vs BGL ground truth")
    n_normal_gt = int((y_true == 0).sum())
    n_anom_gt = int((y_true == 1).sum())
    st.caption(
        "**true_anomaly** = **0** only when the first field of the line is **`-`** "
        "(normal); **1** for any other first token (e.g. **E**, **APPREAD**). "
        "Derived from the raw **message** line (first token, Unicode dash normalized). "
        f"In this sample: **{n_normal_gt:,}** normal, **{n_anom_gt:,}** true anomalies. "
        "Compared to model **anomaly** (Isolation Forest). Positive class = **1** (anomaly)."
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

    anomalies = df[df["anomaly"] == 1]
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
            "anomaly",
        ]
        st.dataframe(
            anomalies[show_cols],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Anomalies in the sample (row index)")
    viz = df.copy()
    viz["row_index"] = range(len(viz))
    viz["status"] = viz["anomaly"].map({0: "normal", 1: "anomaly"})
    st.scatter_chart(
        viz,
        x="row_index",
        y="clean_msg_length",
        color="status",
    )


if __name__ == "__main__":
    main()
