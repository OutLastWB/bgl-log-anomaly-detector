"""
Pure Python log sampling, BGL parsing, and Isolation Forest anomaly detection.
No Streamlit or web framework imports.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import BinaryIO, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

MAX_LINES = 10_000

_TS_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}(?:\.\d+)?)\b")


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
    contamination: float = 0.1,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """
    Build numeric features, fit Isolation Forest, add column anomaly (1 = outlier).

    Features: StandardScaler on clean_msg_length, hour_of_day, node_freq, type_freq,
    rare_score; plus label-encoded node and type. Rare score is 1 / message_freq
    where message_freq counts identical clean_message strings in the sample.
    """
    out = df.copy()
    out["_node"] = out["node"].replace("", "__empty__").fillna("__empty__")
    out["_type"] = out["type"].replace("", "__empty__").fillna("__empty__")

    out["clean_msg_length"] = out["clean_message"].astype(str).str.len()
    out["hour_of_day"] = out["timestamp"].dt.hour
    out.loc[out["timestamp"].isna(), "hour_of_day"] = -1

    out["node_freq"] = out.groupby("_node", sort=False)["_node"].transform("count")
    out["type_freq"] = out.groupby("_type", sort=False)["_type"].transform("count")

    _msg = out["clean_message"].astype(str)
    _msg_counts = _msg.value_counts()
    out["message_freq"] = _msg.map(_msg_counts).astype(int)
    out["rare_score"] = 1.0 / out["message_freq"].astype(float)

    le_node = LabelEncoder()
    le_type = LabelEncoder()
    out["node_encoded"] = le_node.fit_transform(out["_node"])
    out["type_encoded"] = le_type.fit_transform(out["_type"])

    scale_cols = [
        "clean_msg_length",
        "hour_of_day",
        "node_freq",
        "type_freq",
        "rare_score",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(out[scale_cols].astype("float64"))
    X_cat = out[["node_encoded", "type_encoded"]].astype("float64").to_numpy()
    X = np.hstack([X_scaled, X_cat])

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


def process_log_file(
    file: Union[str, Path, BinaryIO],
    *,
    sampling_mode: str = "random",
    max_lines: int = MAX_LINES,
    sample_seed: int = 42,
    contamination: float = 0.1,
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
