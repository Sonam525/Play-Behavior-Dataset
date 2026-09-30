# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Package the release and upload it to Hugging Face
# MAGIC
# MAGIC Reads a release manifest, builds the files of the dataset repository
# MAGIC (https://huggingface.co/datasets/Sonam5/Calf-Play-Behavior-Dataset) and uploads them in batches.
# MAGIC
# MAGIC **Manifest** (CSV) columns: `stage, category, session_id, source_blob, target_path, bytes, action, notes`.
# MAGIC * `source_blob`: source path, relative to `data_root` (or absolute). For tracking tars: `masks=<dir>/|initial_prompts=<dir>/`.
# MAGIC   For `event_log` and `labels` rows it may be empty: the source then comes from `config/sessions.yaml` (see below).
# MAGIC * `target_path`: path in the dataset repository; must start with one of `README.md`, `LICENSE`, `metadata/`,
# MAGIC   `annotations/`, `frames/`, `tracking/`, `videos/`.
# MAGIC * `bytes`: expected size (checked for copies and tars; empty = unknown).
# MAGIC * `notes`: `op=<name>;key=value;... | free text` (parameters of the action).
# MAGIC
# MAGIC **Actions**
# MAGIC * `copy`: copies a file unchanged (a source ending in `/` is copied as a directory). Videos, metadata tables.
# MAGIC * `tar_folder`: builds an uncompressed USTAR tar with deterministic headers (mode 0644, uid/gid 0, owner names empty,
# MAGIC   mtime = session date 00:00 UTC), members read in sorted order:
# MAGIC   * `op=frames_tar;members=;first=;last=`: one tar per 3,000-frame `video<k>` folder, members `<NNNNNNN>.jpg`,
# MAGIC     target `frames/<session>/<session>_video<NNN>.tar`;
# MAGIC   * `op=tracking_tar`: `<session>/masks/video<NNN>/annotations_<calf>.json` (from notebook 05 folders `video<k>`)
# MAGIC     plus `<session>/initial_prompts/*.txt`, target `tracking/<session>.tar`.
# MAGIC * `convert` with `op=event_log;session_id=<id>` -> `annotations/event_logs/<session_id>.csv`, from the authors'
# MAGIC   corrected Observer XT export of the session (`sessions.<id>.event_log` in `config/sessions.yaml`):
# MAGIC   1. observer initials at the start of observation names (`XX_Tol...`, `XX_Eem...`) are replaced by pseudonymous codes
# MAGIC      `O1..On` (assigned in chronological order of first appearance over all exports of the manifest; the mapping is never
# MAGIC      printed or stored). CSV exports are rewritten cell by cell and otherwise byte-faithful; the pilot's `.xlsx` export is
# MAGIC      read with pandas and written as CSV;
# MAGIC   2. `split` (Tol2 2024-06-11): keeps the observations that start before (`AM`) or at/after (`PM`) 13:20:00;
# MAGIC   3. `comments_from` (Eem 2024-06-19): restores the `Comment` column, lost when three partial exports were merged,
# MAGIC      row by row from those partial exports (after checking that they align with the merged export);
# MAGIC   4. common column order (`Result_Container` and `Comment` added empty where missing); the column `calf ID` is dropped
# MAGIC      (empty everywhere except in the pilot, where it repeats the focal calf); `rename_columns` for the pilot;
# MAGIC   5. `calf_id` from `metadata/calves.csv` (parameter `calves_csv`) and `observation_start` (ISO local time) from
# MAGIC      `Actual date and start time`, from the observation (= video file) name (`observation_start_from_name`) or from
# MAGIC      `observation_start_override`;
# MAGIC   6. check that no observer-like prefix other than `O<n>` remains.
# MAGIC * `convert` with `op=labels;session_id=<id>` -> `annotations/labels/<session_id>.csv.gz`, from the output of
# MAGIC   notebook 07: columns `session_id, frame_name, frame_tar, calf_id, subject, ocr_timestamp, primary_behaviour,
# MAGIC   secondary_behaviour, label` (`frame_tar` = the frame tar that holds the frame), gzip level 9 with mtime 0
# MAGIC   (the compressed bytes depend on the zlib version; the content does not).
# MAGIC * `files_manifest` (target `metadata/files_manifest.csv`): `path,bytes,sha256` of every uploaded file, written at
# MAGIC   the end of each upload run.
# MAGIC
# MAGIC **Modes** (`mode`):
# MAGIC * `dry_run` (default) never contacts the Hub: it validates the manifest, checks that every source exists (with its size),
# MAGIC   builds the annotation files in memory (and checks them) and builds one sample frame tar, verifying member names,
# MAGIC   sizes and the tar size computed from the member sizes;
# MAGIC * `build` writes every selected row to `<output_root>/release/build/<target_path>` (no Hub access) and prints
# MAGIC   `path, bytes, sha256` of each file;
# MAGIC * `upload`: batches of at most `max_files` (<= 100) files and `max_batch_gb` per commit with `HfApi.upload_folder`
# MAGIC   (Xet backend); the next batch is staged while the previous one uploads. Completed target paths are recorded in a JSON
# MAGIC   state file, so a re-run resumes. The Hugging Face token is read from the environment variable `HF_TOKEN` or from a
# MAGIC   Databricks secret (`secret_scope` / `secret_key` parameters) and is never printed.

# COMMAND ----------

# MAGIC %pip install -U 'huggingface_hub[hf_xet]' openpyxl pyyaml

# COMMAND ----------

# ---- release_lib: shared helpers for the release ----
# Contains no credentials and no observer identities: observer prefixes are discovered from the
# data at run time and replaced by pseudonymous codes O1..On (deterministic order, see below).
import csv, gzip, hashlib, io, os, re, tarfile, time, datetime as _dt
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Chronological session order (used to assign pseudonymous observer codes deterministically)
SESSION_ORDER = [
    "Tol_2024-04-18", "Tol_2024-04-24", "Tol2_2024-06-11_AM", "Tol2_2024-06-11_PM", "Tol2_2024-06-12",
    "Eem_2024-06-18", "Eem_2024-06-19", "Eem_2024-06-20", "Tol3_2025-05-12",
]
_FARM_LOOKAHEAD = r"(?=[_\-](?:Tol|Eem|eem|TOL|EEM))"
OBS_PREFIX_RE = re.compile(r"^\s*([A-Za-z]{1,3})" + _FARM_LOOKAHEAD)
RESIDUAL_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,3})" + _FARM_LOOKAHEAD)
OBS_COLUMNS = ("Observation", "Observations")

# ------------------------------------------------------------------ manifest notes
def parse_notes(notes):
    """notes = 'op=<name>;k=v;... | free text'  ->  dict (empty if no op section)."""
    out = {}
    if not isinstance(notes, str) or not notes.startswith("op="):
        return out
    spec = notes.split(" | ", 1)[0]
    for kv in spec.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out

# ------------------------------------------------------------------ csv helpers (byte-faithful)
def read_csv_bytes(b):
    bom = b.startswith(b"\xef\xbb\xbf")
    txt = b.decode("utf-8-sig")
    lt = "\r\n" if "\r\n" in txt else "\n"
    rows = list(csv.reader(io.StringIO(txt, newline="")))
    return rows, lt, bom

def write_csv_bytes(rows, lt="\n", bom=False):
    s = io.StringIO(newline="")
    csv.writer(s, lineterminator=lt).writerows(rows)
    b = s.getvalue().encode("utf-8")
    return (b"\xef\xbb\xbf" + b) if bom else b

# ------------------------------------------------------------------ observer pseudonymisation
def obs_prefix(name):
    m = OBS_PREFIX_RE.match(str(name))
    return m.group(1) if m else None

_PREFIX_SEP_RE = re.compile(r"^\s*[A-Za-z]{1,3}[_\-](?=(?:Tol|Eem|eem|TOL|EEM))")

def _obs_sort_key(name):
    return _PREFIX_SEP_RE.sub("", str(name)).strip()

def observations_from_csv_bytes(b):
    rows, _, _ = read_csv_bytes(b)
    if not rows:
        return []
    hdr = rows[0]
    idx = [i for i, h in enumerate(hdr) if h in OBS_COLUMNS]
    return [r[i] for r in rows[1:] for i in idx if i < len(r) and r[i]]

def observations_from_xlsx_bytes(b):
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(b), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        it = ws.iter_rows(values_only=True)
        hdr = next(it, None)
        if not hdr:
            continue
        idx = [i for i, h in enumerate(hdr) if h in OBS_COLUMNS]
        for r in it:
            for i in idx:
                if i < len(r) and r[i]:
                    out.append(str(r[i]))
    wb.close()
    return out

def build_observer_codes(session_obs):
    """session_obs: iterable of (session_id, [observation names]).  Codes O1..On are assigned by
    first appearance in SESSION_ORDER, then by observation name with the prefix stripped."""
    order = {s: i for i, s in enumerate(SESSION_ORDER)}
    codes = {}
    for sid, obs in sorted(session_obs, key=lambda t: order.get(t[0], 999)):
        for o in sorted(set(obs), key=_obs_sort_key):
            p = obs_prefix(o)
            if p and p not in codes and not re.fullmatch(r"O\d+", p):
                codes[p] = "O%d" % (len(codes) + 1)
    return codes

def _sub_re(codes):
    keys = sorted(codes, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, keys)) + r")" + _FARM_LOOKAHEAD)

def pseudo_text(s, codes, _cache={}):
    if not codes or not isinstance(s, str):
        return s
    key = tuple(sorted(codes.items()))
    rx = _cache.get(key)
    if rx is None:
        rx = _cache[key] = _sub_re(codes)
    return rx.sub(lambda m: codes[m.group(1)], s)

def residual_prefixes(text):
    """Return observer-like prefixes still present (O<n> codes never match: a digit precedes the separator)."""
    return sorted({m.group(1) for m in RESIDUAL_RE.finditer(text)})

# ------------------------------------------------------------------ splitting (AM/PM)
def _parse_obs_start(v):
    v = str(v).strip()
    for fmt in ("%d-%m-%Y %H:%M:%S,%f", "%d-%m-%Y %H:%M:%S"):
        try:
            return _dt.datetime.strptime(v, fmt)
        except ValueError:
            pass
    return None

def _split_keep(row, col_idx, part, split_at, kind):
    v = row[col_idx] if col_idx < len(row) else ""
    if kind == "obs_start":
        t = _parse_obs_start(v)
        if t is None:
            raise ValueError("cannot parse observation start %r" % v)
        am = t.strftime("%H:%M:%S") < split_at
    else:  # iso timestamp column
        am = v[: len(split_at)] < split_at
    return am if part == "AM" else (not am)

# ------------------------------------------------------------------ converters (observer exports)
def convert_pseudo_csv(src, codes, split=None, split_col=None, split_at=None, split_kind="obs_start"):
    rows, lt, bom = read_csv_bytes(src)
    hdr = rows[0]
    body = rows[1:]
    if split:
        ci = hdr.index(split_col)
        body = [r for r in body if _split_keep(r, ci, split, split_at, split_kind)]
    out = [[pseudo_text(c, codes) for c in r] for r in [hdr] + body]
    return write_csv_bytes(out, lt, bom), len(body)

def convert_xlsx_to_csv(src, codes, sheet_col=False):
    import pandas as pd
    sheets = pd.read_excel(io.BytesIO(src), sheet_name=None, dtype=object)
    parts = []
    for name, df in sheets.items():
        if sheet_col:
            df = df.copy(); df.insert(0, "source_sheet", name)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.apply(lambda col: col.map(lambda v: pseudo_text(v, codes) if isinstance(v, str) else v))
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8"), len(df)

# ------------------------------------------------------------------ tar building
RECORD = tarfile.RECORDSIZE  # 10240

def expected_ustar_size(member_sizes):
    """Exact size of an uncompressed USTAR archive written by tarfile (stream or file mode)."""
    tot = sum(512 + ((s + 511) // 512) * 512 for s in member_sizes) + 1024
    return ((tot + RECORD - 1) // RECORD) * RECORD

class _HashWriter:
    def __init__(self, f):
        self.f = f; self.h = hashlib.sha256(); self.n = 0
    def write(self, b):
        self.h.update(b); self.n += len(b); return self.f.write(b)
    def flush(self):
        self.f.flush()
    def close(self):
        pass

def _read_file(p):
    with open(p, "rb") as fh:
        return fh.read()

def _lookahead(ex, items, fn, depth):
    q = deque()
    it = iter(items)
    for x in it:
        q.append((x, ex.submit(fn, x)))
        if len(q) >= depth:
            break
    while q:
        x, f = q.popleft()
        nxt = next(it, None)
        if nxt is not None:
            q.append((nxt, ex.submit(fn, nxt)))
        yield x, f.result()

def build_tar(members, out_path, mtime=0, threads=32, depth=256):
    """members: list of (arcname, source_path). Writes an uncompressed USTAR tar with
    deterministic headers (mode 0644, uid/gid 0, fixed mtime). Returns stats incl. sha256."""
    t0 = time.time(); read_bytes = 0
    with open(out_path, "wb") as raw:
        hw = _HashWriter(raw)
        with tarfile.open(fileobj=hw, mode="w|", format=tarfile.USTAR_FORMAT) as tf:
            with ThreadPoolExecutor(threads) as ex:
                for (arc, _src), data in _lookahead(ex, members, lambda m: _read_file(m[1]), depth):
                    ti = tarfile.TarInfo(arc)
                    ti.size = len(data); ti.mtime = int(mtime); ti.mode = 0o644
                    ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
                    tf.addfile(ti, io.BytesIO(data))
                    read_bytes += len(data)
    return {"members": len(members), "read_bytes": read_bytes, "tar_bytes": hw.n,
            "sha256": hw.h.hexdigest(), "seconds": round(time.time() - t0, 2)}

def copy_with_hash(src, dst, bufsize=16 * 1024 * 1024):
    t0 = time.time(); h = hashlib.sha256(); n = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            b = fi.read(bufsize)
            if not b:
                break
            h.update(b); fo.write(b); n += len(b)
    return {"bytes": n, "sha256": h.hexdigest(), "seconds": round(time.time() - t0, 2)}

def sha256_file(p, bufsize=16 * 1024 * 1024):
    h = hashlib.sha256(); n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b); n += len(b)
    return n, h.hexdigest()
# ---- end release_lib ----

# COMMAND ----------

# ---- annotation files: event logs and labels ----
import datetime as dt
import pandas as pd

OBSERVER_COLS = ["Date_Time_Absolute_dmy_hmsf", "Date_dmy", "Time_Absolute_hms", "Time_Absolute_f",
                 "Time_Relative_hmsf", "Time_Relative_hms", "Time_Relative_f", "Time_Relative_sf", "Duration_sf",
                 "Result_Container", "Observation", "Event_Log", "Subject", "Behavior", "Event_Type", "Comment",
                 "Actual date and start time"]
EVENT_COLS = OBSERVER_COLS + ["calf_id", "observation_start"]
LABEL_COLS = ["session_id", "frame_name", "frame_tar", "calf_id", "subject", "ocr_timestamp",
              "primary_behaviour", "secondary_behaviour", "label"]
LABEL_CLASSES = {"Active Playing", "Non Active Playing", "Not Playing"}


def table_bytes(header, rows, gz=False):
    s = io.StringIO(newline="")
    w = csv.writer(s, lineterminator="\n")
    w.writerow(header); w.writerows(rows)
    data = s.getvalue().encode("utf-8")
    return gzip.compress(data, compresslevel=9, mtime=0) if gz else data


def calf_map(calves_csv_path):
    """Every subject spelling -> calf_id (metadata/calves.csv, column subject_strings separated by ' | ')."""
    c = pd.read_csv(calves_csv_path, dtype=str, keep_default_na=False)
    m = {}
    for _, r in c.iterrows():
        for s in r["subject_strings"].split(" | "):
            assert s not in m, s
            m[s] = r["calf_id"]
    return m


def parse_typed_start(v):
    v = v.strip()
    for fmt in ("%d-%m-%Y %H:%M:%S,%f", "%d-%m-%Y %H:%M:%S"):
        try:
            return dt.datetime.strptime(v, fmt)
        except ValueError:
            pass
    return None


def start_from_name(obs):
    """Observation names of Eem 2024-06-20 are video file names: ..._YYYYMMDD_HHMMSS_HHMMSS."""
    m = re.search(r"_(\d{8})_(\d{6})_\d{6}$", obs.strip())
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def iso_ms(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + "%03d" % (t.microsecond // 1000)


def pseudonymised_export(src_path, codes, ev):
    """Step 1-2: the export as CSV bytes, observer initials replaced, optionally split AM/PM."""
    b = open(src_path, "rb").read()
    if src_path.lower().endswith(".xlsx"):
        out, _n = convert_xlsx_to_csv(b, codes, sheet_col=False)
        return out
    sp = ev.get("split") or {}
    out, _n = convert_pseudo_csv(b, codes, sp.get("part"), sp.get("column"), sp.get("at"), "obs_start")
    return out


def restored_comments(df, part_paths, codes):
    """Step 3: Comment column of a merged export, restored row by row from its partial exports."""
    dfs = []
    for q in part_paths:
        d = pd.read_csv(q, dtype=str, keep_default_na=False)
        if "Comment" not in d.columns:
            d["Comment"] = ""
        dfs.append(d)
    bk = pd.concat(dfs, ignore_index=True)
    assert len(bk) == len(df), "partial exports and merged export differ in row count"

    def num(s):
        return s.map(lambda x: round(float(x), 3) if str(x).strip() != "" else None)

    strip = lambda s: s.map(lambda x: pseudo_text(str(x), codes).strip())
    chk = {
        "Time_Relative_sf": int((num(bk["Time_Relative_sf"]).values != num(df["Time_Relative_sf"]).values).sum()),
        "Duration_sf": int((num(bk["Duration_sf"]).fillna(-1).values != num(df["Duration_sf"]).fillna(-1).values).sum()),
        "Behavior": int((bk["Behavior"].values != df["Behavior"].values).sum()),
        "Event_Type": int((bk["Event_Type"].values != df["Event_Type"].values).sum()),
        # the last partial export added ' (Inactive)' to every calf name
        "Subject": int((bk["Subject"].str.replace(" (Inactive)", "", regex=False).values != df["Subject"].values).sum()),
        "Observation": int((strip(bk["Observation"]).values != strip(df["Observation"]).values).sum()),
    }
    assert all(v == 0 for v in chk.values()), f"partial exports do not align with the merged export: {chk}"
    return bk["Comment"].map(lambda v: pseudo_text(v, codes) if isinstance(v, str) else v).values


def build_event_log(sid, src_path, codes, cmap):
    """Steps 1-6 for one session. Returns (csv bytes, report)."""
    cfg = CFG["sessions"][sid]
    ev = cfg["event_log"]
    rows, _lt, _bom = read_csv_bytes(pseudonymised_export(src_path, codes, ev))
    df = pd.DataFrame(rows[1:], columns=rows[0])
    rep = {"rows": len(df)}
    df = df.rename(columns=ev.get("rename_columns") or {})
    if "calf ID" in df.columns:
        if not ev.get("drop_filled_calf_id_column"):
            assert (df["calf ID"] == "").all(), f"{sid}: non-empty 'calf ID'"
        df = df.drop(columns=["calf ID"])
    if ev.get("comments_from"):
        assert "Comment" not in df.columns
        df["Comment"] = restored_comments(df, [to_local(p) for p in ev["comments_from"]], codes)
        rep["comments_restored"] = int((df["Comment"].str.strip() != "").sum())
    for c in OBSERVER_COLS:
        if c not in df.columns:
            df[c] = ""
    assert set(df.columns) == set(OBSERVER_COLS), (sid, sorted(set(df.columns) ^ set(OBSERVER_COLS)))
    df = df[OBSERVER_COLS].copy()

    # calf_id
    unknown = sorted(set(df["Subject"]) - set(cmap))
    assert not unknown, (sid, unknown)
    df["calf_id"] = df["Subject"].map(cmap)

    # observation_start (local camera-clock time, no zone)
    override = ev.get("observation_start_override") or {}
    starts, sources = {}, {"typed": 0, "from_name": 0, "override": 0}
    for obs, g in df.groupby("Observation", sort=False):
        typed = sorted(set(g["Actual date and start time"]))
        assert len(typed) == 1, (sid, obs)
        t = parse_typed_start(typed[0])
        name_t = start_from_name(obs)
        if obs.strip() in override:
            starts[obs] = override[obs.strip()]; sources["override"] += 1
        elif ev.get("observation_start_from_name"):
            assert name_t is not None, obs
            if t is None or t != name_t:
                starts[obs] = iso_ms(name_t); sources["from_name"] += 1
            else:
                starts[obs] = iso_ms(t); sources["typed"] += 1
        else:
            assert t is not None, (sid, obs)
            assert t.strftime("%Y-%m-%d") == cfg["date"], (sid, obs, t)
            starts[obs] = iso_ms(t); sources["typed"] += 1
    df["observation_start"] = df["Observation"].map(starts)
    rep["observation_start_sources"] = sources
    rep["n_observations"] = len(starts)

    # no observer initials left (only O<n> codes)
    assert not residual_prefixes("\n".join(df["Observation"]) + "\n" + "\n".join(df["Comment"])), \
        f"{sid}: observer-like prefixes remain"
    rep["calves"] = sorted(set(df["calf_id"]))
    return table_bytes(EVENT_COLS, df[EVENT_COLS].values.tolist()), rep


def build_labels(sid, labels_csv_path, cmap):
    """Notebook 07 output -> released label table. Returns (csv.gz bytes, report)."""
    df = pd.read_csv(labels_csv_path, dtype=str, keep_default_na=False)
    df["session_id"] = sid
    ch = df["frame_name"].str[:7].astype(int).sub(1).floordiv(3000).add(1)
    df["frame_tar"] = [f"frames/{sid}/{sid}_video{c:03d}.tar" for c in ch]
    unknown = sorted(set(df["ID"]) - set(cmap))
    assert not unknown, unknown
    o = pd.DataFrame({
        "session_id": df["session_id"], "frame_name": df["frame_name"], "frame_tar": df["frame_tar"],
        "calf_id": df["ID"].map(cmap), "subject": df["ID"], "ocr_timestamp": df["sec_ts"],
        "primary_behaviour": df["primary_raw"], "secondary_behaviour": df["secondary_raw"], "label": df["final_label"]})
    assert set(o["label"]) <= LABEL_CLASSES
    assert not o.duplicated(["frame_name", "calf_id"]).any()
    return (table_bytes(LABEL_COLS, o[LABEL_COLS].values.tolist(), gz=True),
            {"rows": len(o), "calves": sorted(set(o["calf_id"])), "labels": o["label"].value_counts().to_dict()})

# COMMAND ----------

import json, shutil, traceback
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
RELEASE_DIR = os.path.join(OUTPUT_ROOT, CFG["outputs"]["release"])

W = {k: _param(k, d) for k, d in [
    ("manifest", os.path.join(RELEASE_DIR, "manifests", "release_manifest.csv")),
    ("mode", "dry_run"),                  # 'dry_run', 'build' or 'upload'
    ("calves_csv", os.path.join(DATA_ROOT, "metadata", "calves.csv")),   # subject spelling -> calf_id
    ("repo_id", ""),                      # e.g. <namespace>/<name>
    ("repo_type", "dataset"),
    ("stage", "all"),                     # 'all' or a stage value of the manifest
    ("categories", "all"),                # comma list of manifest categories, or 'all'
    ("private", "true"),                  # visibility when the repository is created
    ("secret_scope", ""),                 # Databricks secret holding the token (optional; else env HF_TOKEN)
    ("secret_key", ""),
    ("state_dir", os.path.join(RELEASE_DIR, "state")),
    ("build_dir", os.path.join(RELEASE_DIR, "build")),
    ("tmp_dir", os.path.join(RELEASE_DIR, "tmp")),   # local disk with room for two batches
    ("max_files", "100"),
    ("max_batch_gb", "60"),
    ("read_threads", "32"),
    ("tar_workers", "3"),
]}
MODE = W["mode"].lower()
assert MODE in ("dry_run", "build", "upload"), MODE
PRIVATE = W["private"].lower() in ("true", "1", "yes")
MAX_FILES = max(1, min(100, int(W["max_files"])))
MAX_BATCH_BYTES = int(float(W["max_batch_gb"]) * 1e9)
THREADS = int(W["read_threads"])
TAR_WORKERS = max(1, int(W["tar_workers"]))
TMP = W["tmp_dir"]
TOP_DIRS = ("README.md", "LICENSE", "metadata/", "annotations/", "frames/", "tracking/", "videos/")
os.makedirs(TMP, exist_ok=True)

def to_local(p):
    if p.startswith("dbfs:/"):
        return "/dbfs/" + p[len("dbfs:/"):]
    if os.path.isabs(p):
        return p
    return os.path.join(DATA_ROOT, p)     # path relative to data_root

def source_path(r):
    """Local source of a manifest row; event_log / labels rows may leave source_blob empty (-> config)."""
    if r["source_blob"]:
        return to_local(r["source_blob"])
    p = parse_notes(r["notes"]); sid = p.get("session_id", r["session_id"])
    if p.get("op") == "event_log":
        s = CFG["sessions"][sid]
        return CFG["layouts"]["original"]["event_log"].format(data_root=DATA_ROOT, visit=s["visit"],
                                                              source=s["event_log"]["source"])
    if p.get("op") == "labels":
        return os.path.join(OUTPUT_ROOT, CFG["outputs"]["labels"], f"{sid}.csv")
    raise ValueError(f"no source for {r['target_path']}")

def load_manifest(path):
    m = pd.read_csv(to_local(path), dtype=str, keep_default_na=False)
    need = ["stage", "category", "session_id", "source_blob", "target_path", "bytes", "action", "notes"]
    assert list(m.columns[:8]) == need, f"unexpected columns in {path}: {list(m.columns)}"
    return m

def select_rows(m):
    if W["stage"] != "all":
        m = m[m.stage == W["stage"]]
    if W["categories"].lower() != "all":
        cats = {c.strip() for c in W["categories"].split(",") if c.strip()}
        m = m[m.category.isin(cats)]
    return m.reset_index(drop=True)

def session_epoch(sid):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", sid or "")
    return int(dt.datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()) if m else 0

def frame_members(r):
    """(arcname, local path) list for a frames tar_folder row, sorted by frame name."""
    d = to_local(r["source_blob"]).rstrip("/")
    names = sorted(n for n in os.listdir(d) if re.fullmatch(r"\d{7}\.jpg", n))
    return [(n, f"{d}/{n}") for n in names]

def tracking_members(r):
    sid = r["session_id"]; out = []
    for part in r["source_blob"].split("|"):
        label, pref = part.split("=", 1)
        base = to_local(pref).rstrip("/")
        for root, _dirs, files in os.walk(base):
            for fn in files:
                p = os.path.join(root, fn)
                if os.path.getsize(p) == 0:
                    continue
                rel = os.path.relpath(p, base).replace(os.sep, "/")
                if label == "masks":
                    ch, rest = rel.split("/", 1)
                    rel = "video%03d/%s" % (int(ch[5:]), rest)
                out.append((f"{sid}/{label}/{rel}", p))
    return sorted(out)

def members_for(r):
    return tracking_members(r) if parse_notes(r["notes"]).get("op") == "tracking_tar" else frame_members(r)

# ------------------------------------------------------------------ observer codes (from every export in the manifest)
def build_codes(rows):
    sess_obs = []
    for _, r in rows.iterrows():
        p = parse_notes(r["notes"])
        if p.get("op") == "event_log":
            src = source_path(r)
            b = open(src, "rb").read()
            obs = observations_from_xlsx_bytes(b) if src.lower().endswith(".xlsx") else observations_from_csv_bytes(b)
            sess_obs.append((p.get("session_id", r["session_id"]), obs))
    return build_observer_codes(sess_obs)

_CMAP = {}
def cmap():
    if not _CMAP:
        _CMAP.update(calf_map(to_local(W["calves_csv"])))
    return _CMAP

def run_convert(r, codes):
    """Returns (bytes, info) for a convert row."""
    p = parse_notes(r["notes"]); op = p.get("op"); sid = p.get("session_id", r["session_id"])
    if op == "event_log":
        out, rep = build_event_log(sid, source_path(r), codes, cmap())
    elif op == "labels":
        out, rep = build_labels(sid, source_path(r), cmap())
    else:
        raise ValueError(f"unknown convert op {op!r}")
    return out, {"op": op, "out_bytes": len(out), **rep}

print(json.dumps({k: v for k, v in W.items() if k not in ("secret_key",)}, indent=1))

# COMMAND ----------

# ================================================================== DRY RUN (never contacts the Hub)
def dry_run():
    t_start = time.time()
    S = {"mode": "dry_run", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "hub_contacted": False}
    man = load_manifest(W["manifest"])

    # ---------- 1. manifest
    bad_action = man[~man.action.isin(["copy", "tar_folder", "convert"])]
    bad_top = man[~man.target_path.apply(lambda t: t.startswith(TOP_DIRS))]
    dup = man[man.target_path.duplicated(keep=False) & (man.target_path != "metadata/files_manifest.csv")]
    S["manifest_checks"] = {"rows": int(len(man)), "bad_action": int(len(bad_action)), "bad_top_level": int(len(bad_top)),
                            "duplicate_targets": int(len(dup))}

    # ---------- 2. source existence (with sizes)
    def check(r):
        src = r["source_blob"]
        if src.startswith("GENERATED"):
            return ("generated", r["target_path"], None)
        try:
            lp = to_local(src.split("|")[0].split("=", 1)[-1]) if r["action"] == "tar_folder" else source_path(r)
            if r["action"] == "tar_folder" or src.endswith("/"):
                return ("ok" if os.path.isdir(lp.rstrip("/")) else "missing", r["target_path"], None)
            sz = os.path.getsize(lp)
            if r["action"] == "copy" and r["bytes"] and int(r["bytes"]) != sz:
                return ("size_mismatch", r["target_path"], [int(r["bytes"]), sz])
            return ("ok", r["target_path"], None)
        except (FileNotFoundError, ValueError):
            return ("missing", r["target_path"], None)
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(check, [r for _, r in man.iterrows()]))
    cnt = {}
    for s, _, _ in res:
        cnt[s] = cnt.get(s, 0) + 1
    S["source_check"] = {"counts": cnt, "problems": [(s, t, x) for s, t, x in res if s in ("missing", "size_mismatch")][:50]}

    # ---------- 3. annotation files in memory (nothing kept)
    ann = man[(man.action == "convert") & (man.category == "annotations")]
    conv = []
    codes = build_codes(ann)
    S["observer_codes_assigned"] = len(codes)          # the mapping itself is never printed or stored
    for _, r in ann.iterrows():
        try:
            out, info = run_convert(r, codes)
            info["target"] = r["target_path"]; info["status"] = "ok"
            conv.append(info)
        except Exception as e:
            conv.append({"target": r["target_path"], "status": f"ERROR {type(e).__name__}: {e}"})
    S["annotation_files"] = conv

    # ---------- 4. one sample frame tar (first frames row), verified
    fr = man[(man.category == "frames") & (man.action == "tar_folder")].sort_values("target_path")
    if len(fr):
        r = fr.iloc[0]; notes = parse_notes(r["notes"])
        ddir = os.path.join(TMP, "dryrun"); shutil.rmtree(ddir, ignore_errors=True); os.makedirs(ddir)
        out_tar = os.path.join(ddir, os.path.basename(r["target_path"]))
        mem = members_for(r)
        st = build_tar(mem, out_tar, mtime=session_epoch(r["session_id"]), threads=THREADS)
        with tarfile.open(out_tar, "r:") as tf:
            infos = tf.getmembers()
        names = [i.name for i in infos]
        verify = {"members": len(infos), "all_regular_files": all(i.isfile() for i in infos),
                  "tar_bytes": os.path.getsize(out_tar),
                  "tar_bytes_expected_from_members": expected_ustar_size([i.size for i in infos]), "sha256": st["sha256"]}
        if "first" in notes and "last" in notes:
            verify["names_ok"] = names == ["%07d.jpg" % i for i in range(int(notes["first"]), int(notes["last"]) + 1)]
        if r["bytes"]:
            verify["tar_bytes_expected"] = int(r["bytes"])
        verify["PASS"] = (verify["all_regular_files"] and verify["tar_bytes"] == verify["tar_bytes_expected_from_members"]
                          and verify.get("names_ok", True)
                          and verify.get("tar_bytes_expected", verify["tar_bytes"]) == verify["tar_bytes"])
        S["sample_tar"] = {"target_path": r["target_path"], "verify": verify}
        shutil.rmtree(ddir, ignore_errors=True)
    S["elapsed_s"] = round(time.time() - t_start, 1)
    return S

# COMMAND ----------

# ================================================================== BUILD / UPLOAD
def materialize(r, bdir, codes):
    """Writes the row's target file(s) under bdir. Returns list of (target_path, bytes, sha256)."""
    act = r["action"]; tgt = r["target_path"]; out = []
    if act == "copy" and r["source_blob"].endswith("/"):
        base = to_local(r["source_blob"]).rstrip("/")
        for root, _d, files in os.walk(base):
            for fn in files:
                src = os.path.join(root, fn); rel = os.path.relpath(src, base).replace(os.sep, "/")
                dst = os.path.join(bdir, tgt.rstrip("/"), rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
                c = copy_with_hash(src, dst); out.append((f"{tgt.rstrip('/')}/{rel}", c["bytes"], c["sha256"]))
        return out
    dst = os.path.join(bdir, tgt); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if act == "copy":
        c = copy_with_hash(to_local(r["source_blob"]), dst)
        if r["bytes"] and int(r["bytes"]) != c["bytes"]:
            raise RuntimeError(f"size mismatch for {tgt}: manifest {r['bytes']} vs read {c['bytes']}")
        return [(tgt, c["bytes"], c["sha256"])]
    if act == "tar_folder":
        mem = members_for(r)
        n_exp = int(parse_notes(r["notes"]).get("members", len(mem)))
        if len(mem) != n_exp:
            raise RuntimeError(f"{tgt}: {len(mem)} members found, manifest says {n_exp}")
        st = build_tar(mem, dst, mtime=session_epoch(r["session_id"]), threads=THREADS)
        if r["bytes"] and int(r["bytes"]) != st["tar_bytes"]:
            raise RuntimeError(f"{tgt}: tar bytes {st['tar_bytes']} != manifest {r['bytes']}")
        return [(tgt, st["tar_bytes"], st["sha256"])]
    if act == "convert":
        data, _info = run_convert(r, codes)
        with open(dst, "wb") as f:
            f.write(data)
        n, h = sha256_file(dst)
        return [(tgt, n, h)]
    raise ValueError(act)

def build_all():
    """mode = build: every selected row -> build_dir/<target_path>; no Hub access."""
    man = select_rows(load_manifest(W["manifest"]))
    codes = build_codes(load_manifest(W["manifest"]))     # codes always from every export of the manifest
    files = []
    for _, r in man.iterrows():
        if r["target_path"] == "metadata/files_manifest.csv":
            continue
        files += materialize(r.to_dict(), W["build_dir"], codes)
    for tgt, n, h in files:
        print(f"{tgt},{n},{h}")
    return {"mode": "build", "build_dir": W["build_dir"], "files": len(files), "bytes": sum(n for _, n, _ in files)}

def state_path(stage_label):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", W["repo_id"])
    return os.path.join(W["state_dir"], f"{safe}_stage{stage_label}.json")

def load_state(p):
    if os.path.exists(p):
        return json.load(open(p))
    return {"repo_id": W["repo_id"], "done": {}, "batches": []}

def save_state(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=0)
    os.replace(tmp, p)

def files_manifest_bytes():
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", W["repo_id"])
    d = W["state_dir"]; rows = {}
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.startswith(safe + "_stage") and fn.endswith(".json"):
            for path, v in json.load(open(os.path.join(d, fn)))["done"].items():
                if "sha256" in v:
                    rows[path] = (v["bytes"], v["sha256"])
    lines = ["path,bytes,sha256"] + [f"{p},{b},{h}" if "," not in p else f'"{p}",{b},{h}' for p, (b, h) in sorted(rows.items())]
    return ("\n".join(lines) + "\n").encode("utf-8"), len(rows)

def get_token():
    if W["secret_scope"] and W["secret_key"]:
        return dbutils.secrets.get(W["secret_scope"], W["secret_key"])  # noqa: F821  (Databricks only; never printed)
    tok = os.environ.get("HF_TOKEN", "")
    assert tok, "set HF_TOKEN or the secret_scope/secret_key parameters"
    return tok

def real_upload():
    assert W["repo_id"] and "/" in W["repo_id"], "set repo_id (namespace/name)"
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ["HF_XET_CACHE"] = os.path.join(TMP, "hf_xet_cache")
    from huggingface_hub import HfApi
    api = HfApi(token=get_token())
    api.create_repo(W["repo_id"], repo_type=W["repo_type"], private=PRIVATE, exist_ok=True)
    man = select_rows(load_manifest(W["manifest"]))
    stage_label = W["stage"] if W["stage"] != "all" else "-".join(sorted(set(man.stage)))
    sp = state_path(stage_label); state = load_state(sp)
    codes = build_codes(load_manifest(W["manifest"]))
    todo = [r.to_dict() for _, r in man.iterrows()
            if r["target_path"] != "metadata/files_manifest.csv" and r["target_path"] not in state["done"]]
    print(f"{len(todo)} rows to upload, {len(state['done'])} already done")
    batches, cur, cur_b = [], [], 0
    for r in todo:
        b = int(r["bytes"]) if r["bytes"] else 0
        if cur and (len(cur) >= MAX_FILES or cur_b + b > MAX_BATCH_BYTES):
            batches.append(cur); cur, cur_b = [], 0
        cur.append(r); cur_b += b
    if cur:
        batches.append(cur)
    def prepare(k, batch):
        bdir = os.path.join(TMP, f"batch_{k:05d}"); shutil.rmtree(bdir, ignore_errors=True); os.makedirs(bdir)
        t0 = time.time(); results = []
        is_copy = [r["action"] == "copy" and not r["source_blob"].endswith("/") for r in batch]
        is_tar = [r["action"] == "tar_folder" for r in batch]
        copies = [r for r, c in zip(batch, is_copy) if c]
        tars = [r for r, t in zip(batch, is_tar) if t]
        others = [r for r, c, t in zip(batch, is_copy, is_tar) if not c and not t]
        with ThreadPoolExecutor(4) as ex:
            results += list(ex.map(lambda r: (r, materialize(r, bdir, codes)), copies))
        with ThreadPoolExecutor(TAR_WORKERS) as ex:
            results += list(ex.map(lambda r: (r, materialize(r, bdir, codes)), tars))
        for r in others:
            results.append((r, materialize(r, bdir, codes)))
        return bdir, results, time.time() - t0

    t_all = time.time(); up_bytes = 0
    prep_pool = ThreadPoolExecutor(1)            # stages batch k+1 while batch k uploads (at most 2 batches on disk)
    fut = prep_pool.submit(prepare, 1, batches[0]) if batches else None
    for k, batch in enumerate(batches, 1):
        bdir, results, t_prep = fut.result()
        fut = prep_pool.submit(prepare, k + 1, batches[k]) if k < len(batches) else None
        nb = sum(x[1] for _, lst in results for x in lst)
        nfiles = sum(len(lst) for _, lst in results)
        assert nfiles <= 100 or len(batch) == 1, "more than 100 files in one commit"
        for attempt in range(1, 6):
            try:
                t1 = time.time()
                info = api.upload_folder(folder_path=bdir, path_in_repo="", repo_id=W["repo_id"], repo_type=W["repo_type"],
                                         commit_message=f"stage {stage_label}: batch {k}/{len(batches)} ({nfiles} files)")
                t_up = time.time() - t1
                break
            except Exception as e:
                print(f"batch {k} attempt {attempt} failed: {type(e).__name__}: {str(e)[:300]}")
                if attempt == 5:
                    raise
                time.sleep(min(600, 30 * 2 ** attempt))
        commit = getattr(info, "oid", None) or str(info)[:80]
        for r, lst in results:
            for tgt, n, h in lst:
                state["done"][tgt] = {"bytes": n, "sha256": h, "batch": k, "commit": commit}
            state["done"].setdefault(r["target_path"], {"row_done": True, "batch": k, "commit": commit})
        state["batches"].append({"batch": k, "files": nfiles, "bytes": nb, "prep_s": round(t_prep, 1), "upload_s": round(t_up, 1),
                                 "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        save_state(sp, state)
        shutil.rmtree(bdir, ignore_errors=True)
        up_bytes += nb
        print(f"batch {k}/{len(batches)}: {nfiles} files, {nb/1e9:.2f} GB, prep {t_prep:.0f}s, upload {t_up:.0f}s")
    prep_pool.shutdown(wait=True)
    # files_manifest (all stages uploaded so far for this repo)
    data, n = files_manifest_bytes()
    fdir = os.path.join(TMP, "files_manifest"); shutil.rmtree(fdir, ignore_errors=True); os.makedirs(os.path.join(fdir, "metadata"))
    open(os.path.join(fdir, "metadata", "files_manifest.csv"), "wb").write(data)
    api.upload_folder(folder_path=fdir, path_in_repo="", repo_id=W["repo_id"], repo_type=W["repo_type"],
                      commit_message=f"files_manifest.csv ({n} files)")
    shutil.rmtree(fdir, ignore_errors=True)
    return {"mode": "upload", "repo_id": W["repo_id"], "batches": len(batches), "GB_uploaded_this_run": round(up_bytes / 1e9, 2),
            "hours": round((time.time() - t_all) / 3600, 2), "state_file": sp, "files_manifest_rows": n}

# COMMAND ----------

try:
    summary = {"dry_run": dry_run, "build": build_all, "upload": real_upload}[MODE]()
except Exception as e:
    summary = {"mode": MODE, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
print(json.dumps(summary, indent=1, default=str)[:20000])
