# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Generate the per-calf frame labels
# MAGIC
# MAGIC Produces the per-calf, per-frame play labels released in `annotations/labels/<session_id>.csv.gz`
# MAGIC (sessions `Tol_2024-04-24`, `Tol2_2024-06-11_AM`, `Tol2_2024-06-11_PM`, `Eem_2024-06-19`).
# MAGIC
# MAGIC **Time alignment (burned-in camera clock).** The recordings drop frames, so frame times come only from the OCR
# MAGIC of the on-screen clock (notebooks 02-03). Every decoded frame carries the whole camera-clock second read from it
# MAGIC (`status == "Success"` rows only). A 0.1-s tick of the per-calf state table (notebook 06) at time `gt_ts` is
# MAGIC attached to every frame whose OCR second satisfies `ocr_ts <= gt_ts < ocr_ts + 1 s`; all frames that share an OCR
# MAGIC second therefore share that second's label. Frames whose clock could not be read get no label.
# MAGIC
# MAGIC **Label rule (per calf and second).**
# MAGIC 1. For each (second, calf) the tick-level `Primary Raw` behaviours of that calf are counted over the joined rows and
# MAGIC    ranked by count (ties broken by behaviour name). Rank 1 is the second's `primary_raw`, rank 2 its `secondary_raw`.
# MAGIC 2. Both are mapped to 3 classes (`behavior_categories` in `config/sessions.yaml`); `final_label` = `Active Playing`
# MAGIC    if either is Active, else `Non Active Playing` if either is Non Active, else `Not Playing`.
# MAGIC 3. One row per (frame, calf); its label is the calf's label for the frame's OCR second.
# MAGIC 4. A row is dropped when the calf's `primary_raw` for that second is `out of view` (any spelling or case).
# MAGIC
# MAGIC The tick-level `Secondary Raw` column of the state table is not voted. Behaviours that occur there only
# MAGIC (buck, kick, jump, turn) therefore never decide a label.
# MAGIC
# MAGIC **Which calf-frames are labelled.** Rows are kept only for frames that exist and for (frame, calf) pairs that have a
# MAGIC per-calf file under `calf_files_root` (`<ID>/<NNNNNNN>_<ID><calf_files_ext>`). For the released labels these were the
# MAGIC per-calf files (one per calf and frame, made from the calf's tracking box) of the play-behaviour study pipeline
# MAGIC (https://github.com/Sonam525/Individual-Behavior-Analysis-with-CV); they are not part of the dataset. In
# MAGIC `Eem_2024-06-19` they exist for frames up to 0096000 (06:03-11:23) of calf '9, black' only. Set
# MAGIC `calf_files_root = tracking` to label instead every frame in which the calf has a non-empty tracking box
# MAGIC (notebook 05 output or the released `tracking/` files); this option was not used for the released labels.
# MAGIC
# MAGIC **OCR input.** `ocr_source = ocr_table` (default for `layout = original`) reads
# MAGIC `<output_root>/ocr/<session_id>/metadata.csv` (notebook 02) and parses it as in the release run (`dd/MM/yyyy HH:mm:ss`,
# MAGIC after inserting the missing space of the day-first overlays); `ocr_source = frame_index` (default for
# MAGIC `layout = release`) reads the frame index `metadata/frame_index/<session_id>.csv.gz` instead (same seconds).
# MAGIC For `Eem_2024-06-19`, whose first OCR table was overwritten, the release run used the frame-second pairs kept from
# MAGIC that first OCR run; they equal the re-run OCR (frame index) for all 14,194 labelled frames.
# MAGIC
# MAGIC **Inputs.** The per-calf state table of notebook 06 (`<output_root>/per_calf_states/`; in the `release` layout 06
# MAGIC builds it from the published `annotations/event_logs/`), the OCR seconds, the frames and the calf-frame list.
# MAGIC
# MAGIC Output: `<output_root>/labels/<session_id>.csv` with columns
# MAGIC `session_id, frame_name, ID, sec_ts, primary_raw, secondary_raw, final_label, frame_path`
# MAGIC (`sec_ts` = OCR camera-clock second, local time, ISO 8601 without time zone). Notebook 08 writes the released
# MAGIC table from it: `ID` -> `subject` (plus `calf_id`), `sec_ts` -> `ocr_timestamp`, `primary_raw` / `secondary_raw` ->
# MAGIC `primary_behaviour` / `secondary_behaviour`, `final_label` -> `label`, `frame_path` -> `frame_tar`.

# COMMAND ----------

# MAGIC %pip install pyyaml

# COMMAND ----------

# -------------------------------------------------------------------
# 0) IMPORTS, SPARK SETTINGS, CONFIGURATION
# -------------------------------------------------------------------
import glob, json, os, shutil, datetime

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, expr, concat, lit,
    regexp_replace, regexp_extract, floor, udf, date_format, when
)
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("FrameLabels").getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "12")
spark.conf.set("spark.sql.ansi.enabled", "false")         # unparseable OCR strings -> NULL -> not joined
spark.conf.set("spark.sql.session.timeZone", "UTC")       # state-table '...Z' strings and OCR wall clock on the same basis


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
OCR_SOURCE = _param("ocr_source", "frame_index" if LAYOUT == "release" else "ocr_table")   # or 'ocr_table' (02 output)
CALF_FILES_ROOT = _param("calf_files_root", "")                # folder with <ID>/<frame>_<ID><ext>, or 'tracking'
CALF_FILES_EXT = _param("calf_files_ext", ".pt")
sel = _param("sessions", "ALL")

LABELLED = [sid for sid, s in CFG["sessions"].items() if s.get("labels")]
SESSIONS = LABELLED if sel.upper() == "ALL" else [s.strip() for s in sel.split(",") if s.strip()]
OUT = os.path.join(OUTPUT_ROOT, CFG["outputs"]["labels"])
RUN_UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spark_uri(path):
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[len("/dbfs/"):]
    if path.startswith("/Volumes/") or "://" in path or path.startswith("dbfs:"):
        return path
    return "file:" + path


def fmt(template, sid):
    s = CFG["sessions"][sid]
    return template.format(data_root=DATA_ROOT, output_root=OUTPUT_ROOT, visit=s["visit"], session_id=sid,
                           decoded_folder=s["decode"]["folder"])


def export_of(sid):
    return next(e for e in CFG["annotation_exports"] if sid in e["release_sessions"])


def state_table_paths(sid):
    """Per-calf state table of the whole Observer XT export that covers the session (notebook 06 output; the
    state tables are intermediate and not part of the dataset, so run 06 first in either layout)."""
    exp = export_of(sid)
    return [os.path.join(OUTPUT_ROOT, CFG["outputs"]["per_calf_states"], f"{exp['table']}.csv")]


print(SESSIONS, "| OCR:", OCR_SOURCE, "| calf files:", CALF_FILES_ROOT or "(not set)")

# COMMAND ----------

# -------------------------------------------------------------------
# 1) BEHAVIOUR DICTIONARY + FINAL LABEL RULE
# -------------------------------------------------------------------
behavior_categories = CFG["behavior_categories"]

@udf(StringType())
def categorize(b):
    return behavior_categories.get(b, "Not Playing")

def is_out_of_view(c):
    """'out of view' in any spelling/case: letters only, lower case, starts with 'outofview'."""
    return F.lower(F.regexp_replace(F.coalesce(c, F.lit("")), "[^A-Za-z]", "")).startswith("outofview")

# COMMAND ----------

# -------------------------------------------------------------------
# 2) ONE SESSION
# -------------------------------------------------------------------
def list_files(root, ext):
    """<root>/<subdir>/*<ext> (frames: video<k>/*.jpg; calf files: <ID>/*<ext>)."""
    root = root.rstrip("/")
    out = []
    for d in sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))):
        for f in os.listdir(os.path.join(root, d)):
            if f.lower().endswith(ext):
                out.append(f"{root}/{d}/{f}")
    return out


def calf_frames_from_tracking(sid):
    """(frame_name, ID) pairs with a non-empty tracking box. Not used for the released labels.
    Reads the tracks of notebook 05 (output_root) or, if there are none, the released tracks (data_root/tracking/,
    extracted). The calf names of the tracks must match the `Subject` names of the annotations
    (spellings differ for some calves, see metadata/calves.csv)."""
    bases = [os.path.join(OUTPUT_ROOT, CFG["outputs"]["tracking"], sid, "masks"),
             os.path.join(DATA_ROOT, "tracking", sid, "masks")]
    files = []
    for base in bases:
        files = glob.glob(os.path.join(base, "video*", "annotations_*.json"))
        if files:
            break
    pairs = set()
    for p in files:
        with open(p) as fh:
            ann = json.load(fh)
        for frame, a in ann.items():
            if a.get("bounding_box", [0, 0, 0, 0])[2:] != [0, 0]:
                pairs.add((frame, a["object_name"]))
    return pairs


def run_session(sid):
    cfg = CFG["sessions"][sid]
    rep = dict(session_id=sid)
    print(f"\n==================== {sid} ====================")
    decoded_root = fmt(CFG["layouts"][LAYOUT]["frames_dir"], sid).rstrip("/")

    # --- 2a) valid frame paths and labelled (frame, calf) pairs
    valid_frames = list_files(decoded_root, ".jpg")
    frame_set = set(valid_frames)
    bc_frames = spark.sparkContext.broadcast(frame_set)
    rep["n_frames_listed"] = len(valid_frames)
    if CALF_FILES_ROOT == "tracking":
        pairs = calf_frames_from_tracking(sid)
    else:
        assert CALF_FILES_ROOT, "set calf_files_root (folder with <ID>/<frame>_<ID><ext>) or 'tracking'"
        root = fmt(CALF_FILES_ROOT, sid)
        pairs = set()
        for p in list_files(root, CALF_FILES_EXT):
            calf = os.path.basename(os.path.dirname(p))
            stem = os.path.basename(p)[: -len(CALF_FILES_EXT)]
            if stem.endswith("_" + calf):
                pairs.add((stem[: -len(calf) - 1] + ".jpg", calf))
    rep["n_calf_frame_pairs_listed"] = len(pairs)
    bc_pairs = spark.sparkContext.broadcast(pairs)

    @udf(StringType())
    def validate_frame(path_str):
        return path_str if path_str in bc_frames.value else None

    @udf("boolean")
    def has_calf_file(frame_name, calf):
        return (frame_name, calf) in bc_pairs.value

    # --- 2b) per-calf state table (0.1-s ticks)
    gt = None
    for p in state_table_paths(sid):
        d = spark.read.option("header", True).csv(spark_uri(p))
        gt = d if gt is None else gt.unionByName(d)
    gt = gt.withColumn("gt_ts", to_timestamp(col("Timestamp"), "yyyy-MM-dd'T'HH:mm:ss.SSSX"))
    rep["n_gt_ticks"] = gt.count()
    rep["n_gt_ticks_unparsed"] = gt.filter(col("gt_ts").isNull()).count()

    # --- 2c) OCR seconds per frame
    if OCR_SOURCE == "frame_index":
        fi = spark.read.option("header", True).csv(spark_uri(fmt(CFG["layouts"][LAYOUT]["frame_index"], sid)))
        ocr = (fi.filter(col("ocr_status") == "Success")
                 .withColumn("ocr_ts", to_timestamp(col("ocr_timestamp"), "yyyy-MM-dd'T'HH:mm:ss")))
    else:
        ocr_dir = os.path.join(OUTPUT_ROOT, CFG["outputs"]["ocr"], sid, "metadata.csv")
        ocr = (spark.read.option("header", True).csv(spark_uri(ocr_dir) + "/*.csv")
                  .filter(col("status") == "Success"))
        rep["ocr_strings_missing_space"] = ocr.filter(
            col("actual_timestamp").rlike(r"^\d{2}/\d{2}/\d{4}\d{2}:\d{2}:\d{2}$")).count()
        ocr = (ocr.withColumn("fixed_ts", regexp_replace(col("actual_timestamp"),
                                                         r"(^\d{2}/\d{2}/\d{4})(\d{2}:\d{2}:\d{2}$)", "$1 $2"))
                  .withColumn("ocr_ts", to_timestamp(col("fixed_ts"), "dd/MM/yyyy HH:mm:ss")))
    ocr = ocr.filter(col("ocr_ts").isNotNull()).select("frame_name", "ocr_ts").distinct()
    rep["ocr_frames_with_gt1_second"] = ocr.groupBy("frame_name").count().filter("count > 1").count()

    # whole-second OCR times (makes the equi-key below an exact accelerator of the 1-s range join)
    rep["ocr_non_whole_second"] = ocr.filter(col("ocr_ts") != F.date_trunc("second", col("ocr_ts"))).count()
    assert rep["ocr_non_whole_second"] == 0

    # --- 2d) JOIN: tick inside the frame's OCR second (equi-key on the truncated second only for speed)
    joined = gt.withColumn("gt_sec", F.date_trunc("second", col("gt_ts"))).join(
        ocr,
        (col("gt_sec") == col("ocr_ts")) &
        (col("gt_ts") >= col("ocr_ts")) &
        (col("gt_ts") < expr("ocr_ts + interval 1 second")),
        how="inner"
    ).withColumn("sec_ts", col("ocr_ts"))

    # --- 2e) frame paths (3,000 frames per video<k> folder)
    joined = (
        joined
        .withColumn("frame_num", regexp_replace(col("frame_name"), "\\D+", "").cast("int"))
        .withColumn("video_index", floor((col("frame_num") - 1) / 3000) + 1)
        .withColumn("Frame Directory", concat(lit(f"{decoded_root}/video"),
                                              col("video_index").cast("string"), lit("/"), col("frame_name")))
    )

    # --- 2f) keep existing frames and labelled (frame, calf) pairs
    joined = (
        joined
        .withColumn("Frame Directory", validate_frame(col("Frame Directory")))
        .filter(col("Frame Directory").isNotNull() & has_calf_file(col("frame_name"), col("ID")))
        .withColumnRenamed("Primary Raw", "orig_primary_raw")
        .withColumnRenamed("Secondary Raw", "orig_secondary_raw")
    )
    joined.cache()
    rep["n_joined_tick_frame_rows"] = joined.count()

    # --- 2g) per-calf vote per OCR second
    sec_counts = joined.groupBy("sec_ts", "ID", "orig_primary_raw").count()
    rank_window = (Window.partitionBy("sec_ts", "ID")
                   .orderBy(F.desc("count"), F.asc_nulls_last("orig_primary_raw")))   # deterministic tie-break
    ranked = sec_counts.withColumn("rn", F.row_number().over(rank_window))
    ranked.cache()
    primary_per_sec = ranked.filter("rn = 1").select("sec_ts", "ID", col("orig_primary_raw").alias("Primary Raw"))
    secondary_per_sec = ranked.filter("rn = 2").select("sec_ts", "ID", col("orig_primary_raw").alias("Secondary Raw"))
    r2 = ranked.filter("rn = 2").select("sec_ts", "ID", col("count").alias("c2"))
    r3 = ranked.filter("rn = 3").select("sec_ts", "ID", col("count").alias("c3"))
    rep["calf_seconds"] = ranked.select("sec_ts", "ID").distinct().count()
    rep["calf_seconds_tie_rank2_rank3"] = r2.join(r3, ["sec_ts", "ID"]).filter("c2 = c3").count()

    # --- 2h) one row per (frame_name, ID)
    rows = (joined.select("frame_name", "frame_num", "ID", "sec_ts", "Frame Directory")
            .distinct()
            .join(primary_per_sec, ["sec_ts", "ID"], "left")
            .join(secondary_per_sec, ["sec_ts", "ID"], "left"))
    rows = (rows
            .withColumn("Primary Label", categorize(col("Primary Raw")))
            .withColumn("Secondary Label", categorize(col("Secondary Raw")))
            .withColumn("Final Label",
                        when((col("Primary Label") == "Active Playing") | (col("Secondary Label") == "Active Playing"),
                             "Active Playing")
                        .when((col("Primary Label") == "Non Active Playing") | (col("Secondary Label") == "Non Active Playing"),
                              "Non Active Playing")
                        .otherwise("Not Playing")))

    # --- 2i) exclude out-of-view (per-calf Primary raw of that second)
    rows_all = rows.withColumn("oov", is_out_of_view(col("Primary Raw")))
    rows_all.cache()
    rep["n_rows_before_out_of_view"] = rows_all.count()
    rep["n_removed_out_of_view"] = rows_all.filter("oov").count()
    rows = rows_all.filter(~col("oov"))

    out_df = (rows
              .orderBy(col("sec_ts"), col("ID"), col("frame_num"))
              .select(lit(sid).alias("session_id"),
                      "frame_name", "ID",
                      date_format(col("sec_ts"), "yyyy-MM-dd'T'HH:mm:ss").alias("sec_ts"),
                      col("Primary Raw").alias("primary_raw"),
                      col("Secondary Raw").alias("secondary_raw"),
                      col("Final Label").alias("final_label"),
                      col("Frame Directory").alias("frame_path")))
    pdf = out_df.toPandas()

    # ---------------- checks ----------------
    n = len(pdf)
    n_unique = int(pdf[["frame_name", "ID"]].drop_duplicates().shape[0])
    rep["n_rows"] = n
    assert n == n_unique, f"{sid}: duplicated (frame, ID) rows"
    rep["class_counts"] = pdf["final_label"].value_counts().to_dict()
    rep["n_calves"] = int(pdf["ID"].nunique())
    rep["check_all_frame_paths_in_listing"] = bool(pdf["frame_path"].isin(frame_set).all())

    # ---------------- write ----------------
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f"{sid}.csv")
    pdf.to_csv(dest, index=False)
    rep["output"] = dest
    rep["run_utc"] = RUN_UTC
    with open(os.path.join(OUT, f"checks_{sid}.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    for d in (joined, ranked, rows_all):
        d.unpersist()
    print(json.dumps({k: rep[k] for k in ["n_rows", "n_removed_out_of_view", "class_counts",
                                          "calf_seconds_tie_rank2_rank3", "output"]}, indent=1, default=str))
    return rep

# COMMAND ----------

# -------------------------------------------------------------------
# 3) RUN
# -------------------------------------------------------------------
reports = {sid: run_session(sid) for sid in SESSIONS}
