# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - OCR of the burned-in camera clock
# MAGIC
# MAGIC Reads the date and time that the camera burns into every frame, with Tesseract OCR, for every decoded frame of
# MAGIC one session. **This is the time base of the dataset**: the recorder drops frames, so a frame's index or its
# MAGIC position in the video does not give its time.
# MAGIC
# MAGIC Method (unchanged from the pipeline that produced the release; per-session parameters in `config/sessions.yaml`):
# MAGIC 1. crop a fixed region of interest (ROI) around the clock (`ocr.roi = x1, y1, x2, y2`);
# MAGIC 2. convert to grey scale and binarise with an Otsu threshold;
# MAGIC 3. run Tesseract with `--psm 6`;
# MAGIC 4. strip `.` characters and, for the day-first overlays, all spaces;
# MAGIC 5. parse the text as a date-time. `clock_style = dmy_nospace` (Tol2, Tol3, Eem): parse and store
# MAGIC    `%d/%m/%Y%H:%M:%S`. `clock_style = mdy_space` (Tol, April 2024): parse `%m/%d/%Y %H:%M:%S` and store
# MAGIC    `%d/%m/%Y %H:%M:%S`.
# MAGIC
# MAGIC Status per frame: `Success`, `ParseError(<text>): <error>`, `OCRerror: <error>` or `Cannot read <path>`.
# MAGIC Values are **not** corrected, filled or interpolated; screen them against neighbouring frames before use
# MAGIC (the dataset's `metadata/qc/` tables count implausible readings).
# MAGIC
# MAGIC The work is distributed with Spark (`mapPartitions`), 10 frame folders per batch; batch results are written as
# MAGIC parquet and combined at the end.
# MAGIC
# MAGIC **Environment.** Tesseract 4.1.1 and `opencv-python` 4.11.0.86 on every node, installed by the cluster init
# MAGIC script `config/install_ocr.sh` (on Databricks Runtime 15.4 LTS ML / Ubuntu 22.04, `apt-get install tesseract-ocr`
# MAGIC gives 4.1.1). Set `OMP_THREAD_LIMIT=1` in the cluster environment: otherwise Tesseract's OpenMP threads
# MAGIC oversubscribe the parallel Spark tasks. These are the versions printed by the runs that produced the release
# MAGIC (`check_env` below).
# MAGIC
# MAGIC Outputs: `<output_root>/ocr/<session_id>/metadata.csv` (Spark CSV folder) and `metadata.parquet`,
# MAGIC columns `frame_name, actual_timestamp, status`. Notebook 03 turns them into the frame index.

# COMMAND ----------

# MAGIC %pip install pyyaml

# COMMAND ----------

# -------------------------------------------------------------------
# 0) CONFIGURATION
# -------------------------------------------------------------------
import os
import shutil

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
SESSION_ID = _param("session_id", "Eem_2024-06-19")
# Frames to read: 'data' = frames under data_root (released or original layout); 'output' = frames decoded by 01.
FRAMES_FROM = _param("frames_from", "data")

S = CFG["sessions"][SESSION_ID]
assert S.get("decoded"), f"{SESSION_ID} has no frames"
ROI = tuple(S["ocr"]["roi"])
CLOCK_STYLE = S["ocr"]["clock_style"]
assert CLOCK_STYLE in ("dmy_nospace", "mdy_space"), CLOCK_STYLE

if FRAMES_FROM == "output":
    root = os.path.join(OUTPUT_ROOT, CFG["outputs"]["frames"], SESSION_ID)
else:
    root = CFG["layouts"][LAYOUT]["frames_dir"].format(
        data_root=DATA_ROOT, visit=S["visit"], session_id=SESSION_ID, decoded_folder=S["decode"]["folder"])
metadata_root = os.path.join(OUTPUT_ROOT, CFG["outputs"]["ocr"], SESSION_ID)
batch_dir = os.path.join(metadata_root, "metadata_batches")


def spark_uri(path):
    """Local/FUSE path -> URI for Spark writers (DBFS FUSE paths, Unity Catalog volumes or local files)."""
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[len("/dbfs/"):]
    if path.startswith("/Volumes/") or "://" in path or path.startswith("dbfs:"):
        return path
    return "file:" + path


print(SESSION_ID, "ROI", ROI, CLOCK_STYLE, "\n frames:", root, "\n output:", metadata_root)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment check (every node)

# COMMAND ----------

import subprocess

def check_env(_):
    import cv2, pytesseract, subprocess as sp
    # gather versions
    py_opencv   = cv2.__version__
    py_pytess   = pytesseract.get_tesseract_version()  # e.g. 4.1.1
    tess_bin    = sp.check_output(["tesseract","--version"]).decode().split("\n")[0]
    return f"opencv={py_opencv}  pytesseract={py_pytess}  tesseract_bin={tess_bin}"

# parallelize onto, say, 4 tasks (or more)
results = (spark
    .sparkContext
    .parallelize(range(4), 4)
    .map(check_env)
    .collect()
)

print("\n".join(results))

# COMMAND ----------

# -------------------------------------------------------------------
# 3) DEFINE OCR PROCESSING FUNCTION
# -------------------------------------------------------------------
def process_frame_batch(frame_paths, roi=ROI, clock_style=CLOCK_STYLE):
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

# COMMAND ----------

# -------------------------------------------------------------------
# 4) SPARK CONFIG & PARALLEL PROCESSING
# -------------------------------------------------------------------
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType

spark.conf.set("spark.sql.adaptive.enabled", True)
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", True)
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", True)

# discover subdirectories (video1, video2, ...); lexicographic order as in the original run
subdirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
subdirs.sort()
print(f"Found {len(subdirs)} subdirectories under {root}")

batch_size = 10
num_batches = (len(subdirs) + batch_size - 1) // batch_size

schema = StructType([
    StructField("frame_name", StringType(), True),
    StructField("actual_timestamp", StringType(), True),
    StructField("status", StringType(), True)
])

for i in range(num_batches):
    lo = i * batch_size
    hi = min(lo + batch_size, len(subdirs))
    this_batch = subdirs[lo:hi]
    print(f"--- Batch {i+1}/{num_batches}: folders {lo+1}-{hi}")

    # collect frame paths
    frame_paths = []
    for d in this_batch:
        try:
            frame_paths += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")]
        except Exception as e:
            print(f"  ls error on {d}: {e}")

    if not frame_paths:
        continue

    # how many Spark partitions? ~100 frames each, but at least 32 total
    npart = max(32, len(frame_paths) // 100)
    rdd = spark.sparkContext.parallelize(frame_paths, npart)

    def proc_part(iterable):
        batch = []
        for path in iterable:
            batch.append(path)
            if len(batch) >= 10:
                yield from process_frame_batch(batch)
                batch.clear()
        if batch:
            yield from process_frame_batch(batch)

    res_rdd = rdd.mapPartitions(proc_part)
    df = spark.createDataFrame(res_rdd, schema)

    out_path = f"{batch_dir}/batch_{i:03d}"
    df.write.mode("overwrite").parquet(spark_uri(out_path))

    # avoid an extra Spark job: log input count instead of df.count()
    n_input = len(frame_paths)
    print(f"  -> Batch {i:03d} written to {out_path} ({n_input} frames)")

    df.unpersist()
    spark.catalog.clearCache()

# -------------------------------------------------------------------
# 5) COMBINE & FINAL WRITE
# -------------------------------------------------------------------
print("Combining all batches...")
combined = spark.read.parquet(spark_uri(f"{batch_dir}") + "/batch_*")
combined = combined.orderBy("frame_name")

combined.coalesce(4) \
        .write.mode("overwrite") \
        .option("header", "true") \
        .csv(spark_uri(f"{metadata_root}/metadata.csv"))

combined.write.mode("overwrite") \
        .parquet(spark_uri(f"{metadata_root}/metadata.parquet"))

print("Final outputs:")
print(f" - CSV:     {metadata_root}/metadata.csv")
print(f" - Parquet: {metadata_root}/metadata.parquet")

# -------------------------------------------------------------------
# 6) CLEANUP INTERMEDIATE FILES
# -------------------------------------------------------------------
print("Deleting intermediate batch files...")
shutil.rmtree(batch_dir, ignore_errors=True)
print(f"Deleted: {batch_dir}")

# COMMAND ----------

# Summary
t = spark.read.parquet(spark_uri(f"{metadata_root}/metadata.parquet"))
n_rows = t.count()
n_success = t.filter(col("status") == "Success").count()
print(f"{SESSION_ID}: {n_rows} frames, {n_success} Success ({100.0 * n_success / max(n_rows, 1):.2f} %)")
