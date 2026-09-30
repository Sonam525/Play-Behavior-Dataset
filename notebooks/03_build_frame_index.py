# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Build the frame index
# MAGIC
# MAGIC Builds one frame-index file per decoded session, as released in `metadata/frame_index/<session_id>.csv.gz`.
# MAGIC
# MAGIC Columns: `session_id, frame_name, chunk, ocr_timestamp, ocr_status, source`
# MAGIC * one row per decoded JPEG present in `video<k>/` (listed from the frames folder);
# MAGIC * `ocr_timestamp` = the burned-in camera clock read by OCR (notebook 02), re-written as ISO 8601 local camera time
# MAGIC   (`YYYY-MM-DDTHH:MM:SS`, no time zone). Only rows with OCR status `Success` carry a timestamp; the value is
# MAGIC   transcribed as read (misreads are **not** corrected, filled or interpolated);
# MAGIC * `ocr_status` = status class of the OCR row: `Success`, `ParseError`, `OCRerror`, `CannotRead`, or `missing`
# MAGIC   (no OCR row for this frame);
# MAGIC * `source` = where the OCR table came from (`ocr.table` in `config/sessions.yaml`): `original` (OCR run of the
# MAGIC   first processing of the session) or `re-run` (the same OCR code, notebook 02, run later because the first table
# MAGIC   was missing or overwritten).
# MAGIC
# MAGIC The recordings drop frames, therefore frame times come only from the OCR of the on-screen clock; no timing is
# MAGIC derived from frame indices here.
# MAGIC
# MAGIC Inputs: the frames of the session and `<output_root>/ocr/<session_id>/metadata.parquet` (notebook 02).
# MAGIC Output: `<output_root>/metadata/frame_index/<session_id>.csv.gz`.

# COMMAND ----------

# MAGIC %pip install pyyaml

# COMMAND ----------

import json, os, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import yaml
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def _param(name, default=""):
    """Databricks widget if available, otherwise the environment variable NAME (upper case)."""
    try:
        dbutils.widgets.text(name, default)  # noqa: F821  (defined on Databricks)
        return dbutils.widgets.get(name).strip()  # noqa: F821
    except NameError:
        return os.environ.get(name.upper(), default).strip()


CONFIG_PATH = _param("config_path", "../config/sessions.yaml")
with open(CONFIG_PATH, encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

DATA_ROOT = _param("data_root", CFG["paths"]["data_root"])
OUTPUT_ROOT = _param("output_root", CFG["paths"]["output_root"])
LAYOUT = _param("layout", CFG["paths"]["layout"])
FRAMES_FROM = _param("frames_from", "data")            # 'data' (data_root) or 'output' (frames decoded by 01)
sel = _param("sessions", "ALL")

DECODED = [sid for sid, s in CFG["sessions"].items() if s.get("decoded")]
SESSIONS = DECODED if sel.upper() == "ALL" else [s.strip() for s in sel.split(",") if s.strip()]
OUT = os.path.join(OUTPUT_ROOT, CFG["outputs"]["frame_index"])
os.makedirs(OUT, exist_ok=True)

FMT = {"mdy_space": "%d/%m/%Y %H:%M:%S",     # Tol (April 2024): the OCR step re-formats MM/DD/YYYY -> DD/MM/YYYY with a space
       "dmy_nospace": "%d/%m/%Y%H:%M:%S"}    # Tol2 / Tol3 / Eem: no space between date and time


def frames_root(sid):
    s = CFG["sessions"][sid]
    if FRAMES_FROM == "output":
        return os.path.join(OUTPUT_ROOT, CFG["outputs"]["frames"], sid)
    return CFG["layouts"][LAYOUT]["frames_dir"].format(
        data_root=DATA_ROOT, visit=s["visit"], session_id=sid, decoded_folder=s["decode"]["folder"])


def spark_uri(path):
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[len("/dbfs/"):]
    if path.startswith("/Volumes/") or "://" in path or path.startswith("dbfs:"):
        return path
    return "file:" + path


print(SESSIONS)

# COMMAND ----------

def list_frames(root):
    chunks = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    def _ls(name):
        return [(f, name) for f in os.listdir(os.path.join(root, name)) if f.endswith(".jpg")]
    rows = []
    with ThreadPoolExecutor(16) as ex:
        for r in ex.map(_ls, chunks):
            rows += r
    df = pd.DataFrame(rows, columns=["frame_name", "chunk"])
    df["_i"] = df.frame_name.str.replace(".jpg", "", regex=False).astype(int)
    return df.sort_values("_i").reset_index(drop=True), len(chunks)

def status_class(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "missing"
    if s == "Success":
        return "Success"
    for p, c in (("ParseError", "ParseError"), ("OCRerror", "OCRerror"), ("Cannot read", "CannotRead")):
        if s.startswith(p):
            return c
    return "other"

def to_iso(ts, fmt):
    try:
        return datetime.strptime(ts, fmt).isoformat()   # 'YYYY-MM-DDTHH:MM:SS', naive local camera time
    except Exception:
        return None

# COMMAND ----------

summary = []
for sid in SESSIONS:
    t0 = time.time()
    cfg = CFG["sessions"][sid]
    fmt = FMT[cfg["ocr"]["clock_style"]]
    frames, n_chunks = list_frames(frames_root(sid))
    info = {"session_id": sid, "n_chunks": n_chunks, "n_frames": len(frames),
            "frame_idx_min": int(frames._i.min()), "frame_idx_max": int(frames._i.max())}
    pq = os.path.join(OUTPUT_ROOT, CFG["outputs"]["ocr"], sid, "metadata.parquet")
    if os.path.exists(pq):
        ocr = spark.read.parquet(spark_uri(pq)).toPandas()
        info["source"] = cfg["ocr"]["table"]
        ok_rows = ocr[ocr.status == "Success"].sort_values("frame_name")
        if len(ok_rows):
            first_date = to_iso(ok_rows.actual_timestamp.iloc[0], fmt)
            if first_date and first_date[:10] != cfg["date"]:
                print(f"WARNING {sid}: first Success reading {first_date} is not on the session date {cfg['date']}")
    else:
        ocr = pd.DataFrame(columns=["frame_name", "actual_timestamp", "status"])
        info["source"] = ""
    info["n_ocr_rows"] = int(len(ocr))
    info["n_ocr_unique_frames"] = int(ocr.frame_name.nunique())
    ocr = ocr.drop_duplicates("frame_name")
    m = frames.merge(ocr, on="frame_name", how="left")
    info["n_ocr_rows_without_frame"] = int((~ocr.frame_name.isin(frames.frame_name)).sum())
    m["ocr_status"] = [status_class(s) for s in m.status]
    iso = [to_iso(t, fmt) if st == "Success" else None for t, st in zip(m.actual_timestamp, m.ocr_status)]
    m["ocr_timestamp"] = iso
    bad = (m.ocr_status == "Success") & m.ocr_timestamp.isna()
    info["n_success_unparseable"] = int(bad.sum())
    m.loc[bad, "ocr_status"] = "SuccessUnparseable"
    m["ocr_timestamp"] = m.ocr_timestamp.fillna("")
    m["session_id"] = sid
    m["source"] = info["source"]
    out = m[["session_id", "frame_name", "chunk", "ocr_timestamp", "ocr_status", "source"]]
    info["status_counts"] = {k: int(v) for k, v in out.ocr_status.value_counts().items()}
    info["n_success"] = int((out.ocr_status == "Success").sum())
    info["pct_success"] = round(100.0 * info["n_success"] / max(len(out), 1), 3)
    ok = out[out.ocr_status == "Success"].ocr_timestamp
    info["success_ts_min"] = ok.min() if len(ok) else None
    info["success_ts_max"] = ok.max() if len(ok) else None
    dest = os.path.join(OUT, f"{sid}.csv.gz")
    out.to_csv(dest, index=False, compression={"method": "gzip", "mtime": 0})
    info["bytes"] = int(os.path.getsize(dest))
    info["n_rows_written"] = int(len(out))
    assert len(out) == len(frames) and out.frame_name.is_unique
    info["minutes"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(info))
    summary.append(info)

with open(os.path.join(OUT, "_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
