# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Quality-control tables
# MAGIC
# MAGIC Builds the tables in the dataset's `metadata/qc/`. They **only count**: no OCR value or label is corrected, filtered
# MAGIC or interpolated. Thresholds and recording windows are in `config/sessions.yaml` (`qc`, `sessions.<id>.ocr.qc_window`).
# MAGIC
# MAGIC | Part (`parts`) | Output | Input |
# MAGIC |---|---|---|
# MAGIC | `frame_index` | `ocr_coverage.csv`, `ocr_plausibility.csv` | frame index (`metadata/frame_index/`, notebook 03) |
# MAGIC | `labels` | `labels_ocr_plausibility.csv` | frame index and released labels (`annotations/labels/`, notebooks 07-08) |
# MAGIC | `ocr_rerun_check` | `ocr_rerun_reproducibility_vs_original.csv`, `ocr_rerun_visual_validation_template.csv`, ROI contact sheets | frames, Tesseract (as notebook 02), the first OCR tables |
# MAGIC
# MAGIC **Plausibility of a `Success` reading** (frame index, frames ordered by index): implausible if it cannot be parsed as a
# MAGIC date-time, lies outside the session's recording window +- `window_slack_min` minutes, or is more than `jump_s` seconds
# MAGIC away from the rolling median of the `2 x neighbours + 1` `Success` readings centred on it (at least 5 readings).
# MAGIC
# MAGIC **OCR re-run check.** The OCR of notebook 02 (`process_frame_batch`, copied verbatim below) is run on
# MAGIC `rerun_sample` evenly spaced frames of each session. Where the first OCR table of the session exists
# MAGIC (`ocr.original_table`), the status strings and timestamps are compared with it
# MAGIC (`ocr_rerun_reproducibility_vs_original.csv`). For every session the notebook writes the sampled readings and contact
# MAGIC sheets of the ROI crops (x2, 20 per sheet); the clock on each crop was then read by eye and entered in
# MAGIC `visual_clock_reading` and `verdict` (`correct`, `misread`, `no_value_clock_legible`), giving
# MAGIC `ocr_rerun_visual_validation.csv` of the dataset.
# MAGIC
# MAGIC Output: `<output_root>/metadata/qc/`.

# COMMAND ----------

# MAGIC %pip install pyyaml

# COMMAND ----------

import csv, json, os

import numpy as np
import pandas as pd
import yaml


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
PARTS = [p.strip() for p in _param("parts", "frame_index,labels").split(",") if p.strip()]
FRAME_INDEX_DIR = _param("frame_index_dir", os.path.join(DATA_ROOT, "metadata", "frame_index"))   # or 03 output
LABELS_DIR = _param("labels_dir", os.path.join(DATA_ROOT, "annotations", "labels"))            # or 08 build output
sel = _param("sessions", "ALL")
RERUN_SESSIONS = _param("rerun_sessions", "Eem_2024-06-19,Tol2_2024-06-12,Tol3_2025-05-12,"
                                          "Eem_2024-06-18,Tol2_2024-06-11_PM,Tol2_2024-06-11_AM")

DECODED = [sid for sid, s in CFG["sessions"].items() if s.get("decoded")]
SESSIONS = DECODED if sel.upper() == "ALL" else [s.strip() for s in sel.split(",") if s.strip()]
LABELLED = [sid for sid in SESSIONS if CFG["sessions"][sid].get("labels")]
Q = CFG["qc"]
TOL_MIN, NEIGH, JUMP_S = int(Q["window_slack_min"]), int(Q["neighbours"]), int(Q["jump_s"])
OUT = os.path.join(OUTPUT_ROOT, "metadata", "qc")
os.makedirs(OUT, exist_ok=True)


def write_df(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    print("wrote", os.path.join(OUT, name))


def window(sid):
    return tuple(pd.Timestamp(x) for x in CFG["sessions"][sid]["ocr"]["qc_window"])


def read_frame_index(sid):
    df = pd.read_csv(os.path.join(FRAME_INDEX_DIR, f"{sid}.csv.gz"), dtype=str, keep_default_na=False)
    df["_i"] = df.frame_name.str.replace(".jpg", "", regex=False).astype(int)
    return df.sort_values("_i").reset_index(drop=True)


def implausible_mask(s, sid):
    """s: Success rows of the frame index in frame order, with column t (parsed time). Returns the two masks."""
    w0, w1 = window(sid)
    day = w0.normalize()
    out_mask = (s.t.isna() | (s.t < w0 - pd.Timedelta(minutes=TOL_MIN)) | (s.t > w1 + pd.Timedelta(minutes=TOL_MIN))).to_numpy()
    secs = (s.t - day).dt.total_seconds().to_numpy()
    med = pd.Series(secs).rolling(2 * NEIGH + 1, center=True, min_periods=5).median().to_numpy()
    with np.errstate(invalid="ignore"):
        jump_mask = np.isnan(secs) | (np.abs(secs - med) > JUMP_S)
    return out_mask, jump_mask

print(SESSIONS, PARTS)

# COMMAND ----------

# -------------------------------------------------------------------
# PART frame_index: ocr_coverage.csv, ocr_plausibility.csv
# -------------------------------------------------------------------
def qc(sid):
    df = read_frame_index(sid)
    assert df.frame_name.is_unique
    n = len(df)
    n_rows = int((df.ocr_status != "missing").sum())
    s = df[df.ocr_status == "Success"].copy()
    s["t"] = pd.to_datetime(s.ocr_timestamp, format="%Y-%m-%dT%H:%M:%S", errors="coerce")  # years > 2262 -> NaT
    w0, w1 = window(sid)
    day = w0.normalize()
    off_date = int((s.ocr_timestamp.str[:10] != str(day.date())).sum())
    out_mask, jump_mask = implausible_mask(s, sid)
    outside, outlier = int(out_mask.sum()), int(jump_mask.sum())
    implausible = int((out_mask | jump_mask).sum())
    # temporal coverage: minutes of the recording window with >= 1 Success frame whose value lies in that minute
    ok = s.t[(s.t >= w0) & (s.t <= w1)]
    mins_total = int((w1.floor("min") - w0.floor("min")).total_seconds() // 60) + 1
    mins_cov = int(ok.dt.floor("min").nunique())
    # longest run of consecutive frames (by index) without a Success read
    succ = (df.ocr_status == "Success").to_numpy()
    longest, cur = 0, 0
    for v in succ:
        cur = 0 if v else cur + 1
        longest = max(longest, cur)
    status_counts = df.ocr_status.value_counts().to_dict()
    cov = dict(session_id=sid, n_frames=n, n_ocr_rows=n_rows, n_success=len(s),
               pct_success=round(100 * len(s) / n, 2) if n else 0.0, source=df.source.iloc[0] if n else "")
    pla = dict(session_id=sid, n_success=len(s), success_first=s.ocr_timestamp.min(), success_last=s.ocr_timestamp.max(),
               n_success_wrong_date=off_date, n_success_outside_window_5min=outside,
               n_success_local_outlier_60s=outlier, n_success_implausible_any=implausible,
               pct_success_plausible=round(100 * (len(s) - implausible) / max(len(s), 1), 2),
               window_minutes=mins_total, minutes_with_success=mins_cov,
               pct_minutes_with_success=round(100 * mins_cov / mins_total, 1),
               longest_run_frames_without_success=longest, status_counts=json.dumps(status_counts, sort_keys=True))
    return cov, pla


if "frame_index" in PARTS:
    rows = [qc(s) for s in SESSIONS if os.path.exists(os.path.join(FRAME_INDEX_DIR, f"{s}.csv.gz"))]
    write_df(pd.DataFrame([r[0] for r in rows]), "ocr_coverage.csv")
    write_df(pd.DataFrame([r[1] for r in rows]), "ocr_plausibility.csv")

# COMMAND ----------

# -------------------------------------------------------------------
# PART labels: labels_ocr_plausibility.csv
# -------------------------------------------------------------------
def labels_qc(sid):
    s = read_frame_index(sid)
    s = s[s.ocr_status == "Success"].copy()
    s["t"] = pd.to_datetime(s.ocr_timestamp, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    out_mask, jump_mask = implausible_mask(s, sid)
    flag = dict(zip(s.frame_name, out_mask | jump_mask))
    ocr = dict(zip(s.frame_name, s.ocr_timestamp))
    lab = pd.read_csv(os.path.join(LABELS_DIR, f"{sid}.csv.gz"), dtype=str, keep_default_na=False)
    f = lab.frame_name.map(flag)
    same = (lab.ocr_timestamp == lab.frame_name.map(ocr))
    impl = f.fillna(False).astype(bool)
    return dict(session_id=sid, n_label_rows=len(lab), n_label_frames=lab.frame_name.nunique(),
                n_rows_ocr_timestamp_equals_frame_index=int(same.sum()),
                n_rows_frame_not_success_in_frame_index=int(f.isna().sum()),
                n_rows_implausible_ocr=int(impl.sum()),
                pct_rows_implausible_ocr=round(100 * impl.mean(), 2),
                n_rows_implausible_by_class=lab[impl].label.value_counts().to_dict(),
                ocr_timestamp_first=lab.ocr_timestamp.min(), ocr_timestamp_last=lab.ocr_timestamp.max())


if "labels" in PARTS:
    write_df(pd.DataFrame([labels_qc(s) for s in LABELLED if os.path.exists(os.path.join(LABELS_DIR, f"{s}.csv.gz"))]),
             "labels_ocr_plausibility.csv")

# COMMAND ----------

# -------------------------------------------------------------------
# PART ocr_rerun_check: the OCR of notebook 02 on a sample of frames
# (needs the frames and the OCR environment of notebook 02: Tesseract 4.1.1, opencv-python 4.11.0.86)
# -------------------------------------------------------------------
def process_frame_batch(frame_paths, roi, clock_style):
    """Identical to process_frame_batch of notebook 02."""
    import cv2, pytesseract, os
    from datetime import datetime

    # Region of interest where the timestamp appears (per session, config/sessions.yaml)
    x1, y1, x2, y2 = roi
    results = []

    for frame_path in frame_paths:
        frame_name = os.path.basename(frame_path)
        img = None
        try:
            file_path = frame_path.replace("dbfs:", "/dbfs")
            img = cv2.imread(file_path)
            if img is None:
                results.append((frame_name, "", f"Cannot read {file_path}"))
                continue

            # Crop ROI, convert to grayscale, threshold
            roi_img = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 128, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Run Tesseract OCR on the thresholded patch
            raw = pytesseract.image_to_string(thresh, config="--psm 6")

            if clock_style == "mdy_space":
                # Tol (April 2024): remove stray dots but keep the space between date and time;
                # the overlay reads "MM/DD/YYYY HH:MM:SS"
                txt = raw.strip().replace(".", "")
                try:
                    dt = datetime.strptime(txt, "%m/%d/%Y %H:%M:%S")
                    # Re-format into DD/MM/YYYY HH:MM:SS for the output
                    ts = dt.strftime("%d/%m/%Y %H:%M:%S")
                    results.append((frame_name, ts, "Success"))
                except Exception as pe:
                    results.append((frame_name, "", f"ParseError({txt}): {pe}"))
            else:
                # Tol2 / Tol3 / Eem: the overlay reads "DD/MM/YYYY HH:MM:SS"; dots and spaces are removed
                txt = raw.strip().replace(".", "").replace(" ", "")
                try:
                    dt = datetime.strptime(txt, "%d/%m/%Y%H:%M:%S")
                    ts = dt.strftime("%d/%m/%Y%H:%M:%S")
                    results.append((frame_name, ts, "Success"))
                except Exception as pe:
                    results.append((frame_name, "", f"ParseError({txt}): {pe}"))

        except Exception as e:
            results.append((frame_name, "", f"OCRerror: {e}"))
        finally:
            del img

    return results


def frames_root(sid):
    s = CFG["sessions"][sid]
    return CFG["layouts"][LAYOUT]["frames_dir"].format(
        data_root=DATA_ROOT, visit=s["visit"], session_id=sid, decoded_folder=s["decode"]["folder"])


def status_class(s):
    s = str(s)
    return "CannotRead" if s.startswith("Cannot read") else s.split("(")[0].split(":")[0]


def to_iso(ts, clock_style):
    from datetime import datetime
    fmt = "%d/%m/%Y %H:%M:%S" if clock_style == "mdy_space" else "%d/%m/%Y%H:%M:%S"
    try:
        return datetime.strptime(ts, fmt).isoformat()
    except Exception:
        return ""


def rerun_check(sid, n):
    import cv2
    from PIL import Image, ImageDraw
    cfg = CFG["sessions"][sid]
    roi, style = tuple(cfg["ocr"]["roi"]), cfg["ocr"]["clock_style"]
    root = frames_root(sid)
    subdirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
                     key=lambda d: int(d.split("video")[-1]))
    frames = []
    for d in subdirs:
        frames += sorted(os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d)) if f.endswith(".jpg"))
    idx = np.unique(np.linspace(0, len(frames) - 1, n).round().astype(int))
    sample = [frames[i] for i in idx]
    df = pd.DataFrame(process_frame_batch(sample, roi, style), columns=["frame_name", "actual_timestamp", "status"])
    df["chunk"] = [os.path.basename(os.path.dirname(p)) for p in sample]
    repro = None
    ref = cfg["ocr"].get("original_table")
    if ref:
        ref = ref if os.path.isabs(ref) else os.path.join(DATA_ROOT, ref)
    if ref and os.path.exists(ref):
        orig = pd.read_parquet(ref)
        m = df.merge(orig, on="frame_name", how="left", suffixes=("", "_orig"))
        repro = dict(session_id=sid, n_frames=len(m),
                     identical_status_string=int((m.status == m.status_orig).sum()),
                     identical_timestamp=int((m.actual_timestamp.fillna("") == m.actual_timestamp_orig.fillna("")).sum()),
                     n_success=int((m.status == "Success").sum()))
    vis = pd.DataFrame({"session_id": sid, "roi": ",".join(map(str, roi)), "frame_name": df.frame_name,
                        "chunk": df.chunk, "ocr_status": df.status.map(status_class),
                        "ocr_timestamp": [to_iso(t, style) if st == "Success" else "" for t, st in zip(df.actual_timestamp, df.status)],
                        "visual_clock_reading": "", "verdict": ""})
    # contact sheets: ROI crop (x2 upscale) + OCR result, 20 per sheet
    x1, y1, x2, y2 = roi
    tiles = []
    for p, (_, r) in zip(sample, df.iterrows()):
        img = cv2.imread(p)
        crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        im = Image.fromarray(crop)
        w = 640; h = max(1, int(im.height * w / im.width))
        im = im.resize((w, h))
        canvas = Image.new("RGB", (w, h + 16), (255, 255, 255))
        canvas.paste(im, (0, 0))
        st = str(r["status"]).replace("\n", "|")[:60]
        label = f"{r['frame_name']}  {r['actual_timestamp']}  {st}".encode("ascii", "replace").decode("ascii")
        ImageDraw.Draw(canvas).text((2, h + 2), label, fill=(200, 0, 0))
        tiles.append(canvas)
    sheet_dir = os.path.join(OUT, "ocr_rerun_contact_sheets")
    os.makedirs(sheet_dir, exist_ok=True)
    for k in range(0, len(tiles), 20):
        part = tiles[k:k + 20]
        H = sum(t.height + 4 for t in part)
        sheet = Image.new("RGB", (640, H), (0, 0, 255))
        y = 0
        for t in part:
            sheet.paste(t, (0, y)); y += t.height + 4
        sheet.save(os.path.join(sheet_dir, f"{sid}_sheet{k // 20 + 1}.png"))
    return repro, vis


if "ocr_rerun_check" in PARTS:
    repro_rows, vis_parts = [], []
    for sid in [s.strip() for s in RERUN_SESSIONS.split(",") if s.strip()]:
        repro, vis = rerun_check(sid, int(Q["rerun_sample"]))
        if repro:
            repro_rows.append(repro)
        else:
            vis_parts.append(vis)          # sessions without a first OCR table are checked by eye
    if repro_rows:
        write_df(pd.DataFrame(repro_rows), "ocr_rerun_reproducibility_vs_original.csv")
    if vis_parts:
        write_df(pd.concat(vis_parts, ignore_index=True), "ocr_rerun_visual_validation_template.csv")
