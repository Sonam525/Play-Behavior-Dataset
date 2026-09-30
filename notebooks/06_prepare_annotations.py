# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Prepare the annotations: Observer XT export -> per-calf state table
# MAGIC
# MAGIC The observers coded every calf in Noldus The Observer XT (continuous sampling, one observation per video file). Their
# MAGIC exports were checked and corrected by the authors and saved as one file per annotated day, with the column
# MAGIC `Actual date and start time` (the camera-clock time at which the observation's video starts) typed in by the
# MAGIC observers. These corrected exports are the input of the pipeline; the dataset publishes them, pseudonymised, as
# MAGIC `annotations/event_logs/<session_id>.csv` (notebook 08). This notebook turns them into per-calf state tables at
# MAGIC 0.1-s ticks, the intermediate table from which notebook 07 labels frames (not published).
# MAGIC
# MAGIC **Step A (optional) - xlsx to csv.** Observer XT exports saved as `.xlsx` were converted with pandas
# MAGIC (`read_excel` + `to_csv`). This is why `Time_Relative_hmsf` appears in two formats: `MM:SS.f` in the Tol and Tol2
# MAGIC exports and `0 days HH:MM:SS.ffffff` (pandas) in the Eem, Tol3 and pilot-day exports.
# MAGIC
# MAGIC **Step B - per-calf state table** (one per export listed under `annotation_exports` in `config/sessions.yaml`):
# MAGIC 1. event start = `Actual date and start time` + `Time_Relative_hmsf`; event end = start + `Duration_sf`;
# MAGIC 2. a 0.1-s timeline from the first start to the last end of the export;
# MAGIC 3. for each calf (`Subject`) and tick, the state events covering the tick (`start <= t < end - 0.05 s`), ordered by
# MAGIC    start: the first gives `Primary Raw`, the second (if any, e.g. a play state during `rest behavior`) `Secondary Raw`;
# MAGIC 4. `Primary/Secondary Label` = 3-class mapping of the raw behaviour (`behavior_categories` in the config; unlisted
# MAGIC    behaviours are `Not Playing`); `Final Label` = `Active Playing` if either label is Active, else `Non Active Playing`
# MAGIC    if either is Non Active, else `Not Playing`;
# MAGIC 5. Not Playing ticks inside gaps of more than 1 s between consecutive ticks with a play Primary Label (any calf)
# MAGIC    are removed, and ticks without a `Primary Raw` are dropped. The tables are therefore **not complete timelines**.
# MAGIC
# MAGIC Output columns: `Timestamp, Primary Raw, Primary Label, Secondary Raw, Secondary Label, ID, Final Label`.
# MAGIC `Timestamp` is local camera-clock time. Spark writes it with a `Z` suffix because the session time zone is UTC;
# MAGIC the suffix does **not** mean UTC.
# MAGIC
# MAGIC Outputs: `<output_root>/observer_csv/<name>.csv` (step A) and `<output_root>/per_calf_states/<table>.csv` (step B;
# MAGIC one table per export, e.g. `Tol2_2024-06-11` for both halves of that day).

# COMMAND ----------

# MAGIC %pip install openpyxl pyyaml

# COMMAND ----------

# -------------------------------------------------------------------
# 0) CONFIGURATION
# -------------------------------------------------------------------
import glob
import os
import shutil
import time

import yaml
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("GroundTruthCreation").getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")   # as in the runs that produced the release
spark.conf.set("spark.sql.ansi.enabled", "false")     # Spark 3 semantics of those runs (invalid casts -> NULL)
os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()


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
TABLES = _param("tables", "ALL")            # comma list of annotation_exports[].table, or ALL
XLSX_FILES = _param("xlsx_files", "")       # step A: comma list of .xlsx paths to convert (optional)

behavior_categories = CFG["behavior_categories"]
OUT_CSV = os.path.join(OUTPUT_ROOT, CFG["outputs"]["observer_csv"])
OUT_STATES = os.path.join(OUTPUT_ROOT, CFG["outputs"]["per_calf_states"])


def spark_uri(path):
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[len("/dbfs/"):]
    if path.startswith("/Volumes/") or "://" in path or path.startswith("dbfs:"):
        return path
    return "file:" + path


def export_input_files(exp):
    """Observer XT export(s) of one annotation export in the chosen layout: the authors' corrected export as used by
    the pipeline (layout 'original'), or the published event logs of the sessions it covers (layout 'release'; the
    Tol2 2024-06-11 export is published as two session files, which are read together)."""
    lay = CFG["layouts"][LAYOUT]
    if LAYOUT == "original":
        return [lay["event_log"].format(data_root=DATA_ROOT, visit=exp["visit"], source=exp["original_file"])]
    return [lay["event_log"].format(data_root=DATA_ROOT, session_id=sid) for sid in exp["release_sessions"]]

# COMMAND ----------

# -------------------------------------------------------------------
# STEP A) XLSX -> CSV (pandas / openpyxl)
# -------------------------------------------------------------------
import pandas as pd


def convert_xlsx_to_csv(input_path, output_dir=OUT_CSV):
    try:
        # Extract filename and directory
        file_name = os.path.basename(input_path).replace(".xlsx", ".csv")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)

        # Read the Excel file using Pandas
        df = pd.read_excel(input_path, engine='openpyxl')

        # Write DataFrame as CSV
        df.to_csv(output_path, index=False)

        print(f"Converted {input_path} -> {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


for file in [f.strip() for f in XLSX_FILES.split(",") if f.strip()]:
    convert_xlsx_to_csv(file)

# COMMAND ----------

# -------------------------------------------------------------------
# STEP B) PER-CALF STATE TABLE (0.1-s ticks)
# -------------------------------------------------------------------
from pyspark.sql.functions import (
    udf, col, when, lit, explode, sequence, expr, regexp_replace,
    min as spark_min, max as spark_max, count as spark_count,
    coalesce, row_number
)
from pyspark.sql.types import StringType, TimestampType, DoubleType
from pyspark.sql.window import Window
from datetime import datetime
import pyspark.sql.functions as F

# UDFs for parsing
@udf(TimestampType())
def parse_actual(x):
    if not x: return None
    for fmt in ('%d-%m-%Y %H:%M:%S,%f','%d-%m-%Y %H:%M:%S'):
        try: return datetime.strptime(x.strip(), fmt)
        except: pass
    return None

# Time_Relative_hmsf parser used for the Tol / Tol2 exports ('MM:SS.f')
@udf(DoubleType())
def parse_rel_tenths(x):
    if not x: return 0.0
    s = str(x).strip().split()[-1]
    if '.' in s:
        parts = s.split(':')
        if len(parts)==3:
            h,m,sec = parts; s0,f = sec.split('.') if '.' in sec else (sec,'0')
            return round(int(h)*3600+int(m)*60+int(s0)+int(f)/10.0,1)
        if len(parts)==2:
            m,sec = parts; s0,f = sec.split('.') if '.' in sec else (sec,'0')
            return round(int(m)*60+int(s0)+int(f)/10.0,1)
        s0,f = s.split('.') if '.' in s else (s,'0')
        return round(int(s0)+int(f)/10.0,1)
    try:
        dt = datetime.strptime(s, '%H:%M:%S')
    except:
        try: dt = datetime.strptime(s, '%M:%S')
        except: return 0.0
    return dt.hour*3600+dt.minute*60+dt.second

# Time_Relative_hmsf parser used for the Eem exports (adds the pandas timedelta format '0 days HH:MM:SS.ffffff')
@udf(DoubleType())
def parse_rel_timedelta(x):
    if not x:
        return 0.0
    s = str(x).strip()
    # Handle pandas timedelta format like '0 days 00:00:00'
    if 'day' in s:
        parts = s.split()
        try:
            days = int(parts[0])
        except:
            days = 0
        # e.g. ['0', 'days', '00:00:00'] or ['1', 'day', '01:02:03']
        time_part = parts[-1]
        # parse time_part
        if '.' in time_part:
            try:
                dt = datetime.strptime(time_part, '%H:%M:%S.%f')
            except:
                dt = datetime.strptime(time_part, '%H:%M:%S')
        else:
            dt = datetime.strptime(time_part, '%H:%M:%S')
        total_seconds = days*86400 + dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6
        return round(total_seconds, 1)
    # original parsing logic
    # extract last token (supports formats like 'HH:MM:SS.f', 'MM:SS.f', 'SS.f')
    token = s.split()[-1]
    # fractional seconds
    if '.' in token:
        parts = token.split(':')
        if len(parts) == 3:
            h, m, secf = parts
            if '.' in secf:
                sec, frac = secf.split('.')
            else:
                sec, frac = secf, '0'
            seconds = int(h)*3600 + int(m)*60 + int(sec) + int(frac[:1]) / 10.0
            return round(seconds, 1)
        if len(parts) == 2:
            m, secf = parts
            if '.' in secf:
                sec, frac = secf.split('.')
            else:
                sec, frac = secf, '0'
            seconds = int(m)*60 + int(sec) + int(frac[:1]) / 10.0
            return round(seconds, 1)
        # only seconds.f
        sec, frac = token.split('.') if '.' in token else (token, '0')
        return round(int(sec) + int(frac[:1]) / 10.0, 1)
    # no fractional part, parse as HH:MM:SS or MM:SS
    try:
        dt = datetime.strptime(token, '%H:%M:%S')
    except:
        try:
            dt = datetime.strptime(token, '%M:%S')
        except:
            return 0.0
    return dt.hour*3600 + dt.minute*60 + dt.second

PARSERS = {"tenths": parse_rel_tenths, "timedelta": parse_rel_timedelta}

# Categorize UDF
@udf(StringType())
def categorize(b):
    return behavior_categories.get(b.strip(), 'Not Playing') if b else 'Not Playing'


def build_state_table(input_files, parse_rel):
    df = None
    for p in input_files:
        d = spark.read.csv(spark_uri(p), header=True)
        df = d if df is None else df.unionByName(d, allowMissingColumns=True)
    print(f"Loaded {df.count()} rows from {len(input_files)} file(s)")

    # Drop all-null columns
    dropped = [c for c in df.columns if df.filter(col(c).isNotNull()).count()==0]
    for c in dropped:
        df = df.drop(c)
        print(f"Dropped all-NA column: {c}")

    # Behavior -> Category mapping summary
    raws = [r['Behavior'] for r in df.select('Behavior').distinct().collect()]
    print("Behavior -> Category mapping:")
    for b in raws:
        cat = behavior_categories.get(b.strip(), 'Not Playing') if b else 'Not Playing'
        print(f"  '{b}' -> '{cat}'")

    # Parse and compute times
    df = df.withColumn('ActualDateTime', parse_actual(col('Actual date and start time'))) \
           .withColumn('TimeRel', parse_rel(col('Time_Relative_hmsf'))) \
           .withColumn('PresentTime', (col('ActualDateTime').cast('double')+col('TimeRel')).cast('timestamp')) \
           .withColumn('Duration', col('Duration_sf').cast(DoubleType())) \
           .withColumn('EndTime', (col('PresentTime').cast('double')+col('Duration')).cast('timestamp'))

    # Build timeline at 0.1s
    t = df.agg(spark_min('PresentTime').alias('start'), spark_max('EndTime').alias('end')).collect()[0]
    start,end = t['start'], t['end']
    exp_cnt = int((end-start).total_seconds()*10)+1
    print(f"Timeline from {start} to {end}, expecting {exp_cnt} rows")

    tl = spark.createDataFrame([(start,end)], ['s','e']) \
        .select(explode(sequence(col('s'),col('e'),expr('INTERVAL 0.1 SECOND'))).alias('Timestamp'))
    gen_cnt = tl.count()
    print(f"Generated {gen_cnt} timestamps")

    # Initialize defaults
    tl2 = tl.withColumn('Primary Raw', lit(None).cast(StringType())) \
            .withColumn('Secondary Raw', lit(None).cast(StringType()))

    # Prepare behavior intervals
    dfB = df.select('Subject','PresentTime','EndTime','Behavior') \
            .withColumn('EndAdj',(col('EndTime').cast('double')-0.05).cast('timestamp'))

    # Window for row_number
    delta = Window.partitionBy('Timestamp','Subject').orderBy('PresentTime')

    # Assemble final
    elem = None
    for row in dfB.select('Subject').distinct().collect():
        subj = row['Subject']
        subB = dfB.filter(col('Subject')==subj)
        joined = subB.join(
            tl2,
            (tl2['Timestamp']>=subB['PresentTime']) & (tl2['Timestamp']<subB['EndAdj']),
            how='right'
        ).withColumn('rn', row_number().over(delta))

        p = joined.filter(col('rn')==1) \
                  .select('Timestamp','Behavior') \
                  .withColumnRenamed('Behavior','Primary Raw') \
                  .withColumn('Primary Label', categorize(col('Primary Raw')))
        s = joined.filter(col('rn')==2) \
                  .select('Timestamp','Behavior') \
                  .withColumnRenamed('Behavior','Secondary Raw') \
                  .withColumn('Secondary Label', categorize(col('Secondary Raw')))

        merged = p.join(s, 'Timestamp', 'left') \
            .withColumn('ID', lit(subj)) \
            .withColumn('Final Label',
                when((col('Primary Label')=='Active Playing') | (col('Secondary Label')=='Active Playing'), 'Active Playing')
                .when((col('Primary Label')=='Non Active Playing') | (col('Secondary Label')=='Non Active Playing'), 'Non Active Playing')
                .otherwise('Not Playing')
            )

        elem = merged if elem is None else elem.union(merged)

    # Final ordering
    elem = elem.orderBy('Timestamp','ID')

    # Gap detection & drop
    times = elem.filter(col('Primary Label')!='Not Playing').select('Timestamp').orderBy('Timestamp').collect()
    gaps=[]
    for i in range(1,len(times)):
        d=(times[i]['Timestamp']-times[i-1]['Timestamp']).total_seconds()
        if d>1.0: gaps.append((times[i-1]['Timestamp'],times[i]['Timestamp']))
    print(f"Removing {len(gaps)} auto-filled gaps")
    for st,et in gaps:
        elem = elem.filter(~((col('Timestamp')>st)&(col('Timestamp')<et)&(col('Primary Label')=='Not Playing')))

    # Delete rows without a primary label
    before_drop = elem.count()
    elem = elem.filter(col('Primary Raw').isNotNull())
    after_drop = elem.count()
    print(f"Dropped {before_drop - after_drop} rows without Primary Raw label")

    # Summaries
    sec_cnt = elem.filter(col('Secondary Raw').isNotNull()).count()
    print(f"Secondary rows: {sec_cnt}/{elem.count()}")
    elem.groupBy('Final Label').count().show()
    return elem


def write_single_csv(sdf, dest):
    """Spark -> one CSV file (coalesce(1), then move the part file)."""
    tmp = dest + ".tmp"
    sdf.coalesce(1).write.csv(spark_uri(tmp), header=True, mode='overwrite')
    part = [f for f in os.listdir(tmp) if f.startswith('part-') and f.endswith('.csv')]
    assert len(part) == 1, part
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(os.path.join(tmp, part[0]), dest)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Saved to {dest}")

# COMMAND ----------

for exp in CFG["annotation_exports"]:
    if TABLES.upper() != "ALL" and exp["table"] not in [t.strip() for t in TABLES.split(",")]:
        continue
    files = export_input_files(exp)
    print(f"\n==== {exp['table']}: {files}")
    table = build_state_table(files, PARSERS[exp["time_relative_parser"]])
    write_single_csv(table, os.path.join(OUT_STATES, f"{exp['table']}.csv"))
