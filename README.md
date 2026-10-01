# Play-Behavior-Dataset: processing code

Code that produced the derived files of the dataset

**Video recordings and play behaviour annotations of group-housed dairy calves on two Dutch farms**
Dataset: https://huggingface.co/datasets/Sonam5/Calf-Play-Behavior-Dataset (DOI: https://doi.org/10.57967/hf/10695)

The dataset contains video of group-housed dairy calves from two farms in the Netherlands (farm codes **Tol**, a
research and teaching dairy farm, recorded in April 2024 (`Tol`), June 2024 (`Tol2`) and May 2025 (`Tol3`); and **Eem**,
a commercial dairy farm, recorded in June 2024), behaviour event logs coded continuously for every calf in Noldus
The Observer XT, and files derived from them: decoded frames, a per-frame camera-clock time read by OCR, per-calf
tracking boxes and contours, and per-calf frame labels. This repository documents how those derived files were made.
The notebooks are the code that was run, with storage paths replaced by configuration and without credentials;
algorithms and parameters are unchanged.

The frame-decoding, detection and tracking steps come from the video-processing pipeline of the related play-behaviour
study, https://github.com/Sonam525/Individual-Behavior-Analysis-with-CV, which also holds the later steps of that
study. This repository contains only the steps that produced files in the dataset.

## Pipeline

```
 videos/ (raw video)                                 Observer XT exports, corrected by the authors
     |                                                     |
     | 01 decode: every 5th frame -> JPEG,                 | 08 event logs: observer initials -> O1..O5,
     |    3,000 frames per video<k> folder                 |    common columns, calf_id, observation_start
     v                                                     v
 frames/ ---------------------------+             annotations/event_logs/
     |                              |                      |
     | 02 OCR of the burned-in      | 04 YOLO12x boxes on  | 06 per-calf state table at 0.1-s ticks
     |    camera clock              |    the first frame,  |    (intermediate, not published)
     v                              |    named by hand     |
 OCR table                          v                      |
     |                   tracking/<session>/initial_prompts|
     | 03 frame index               |                      |
     v                              | 05 SAMURAI (SAM 2.1),|
 metadata/frame_index/              |    3,000-frame chunks|
     |                              v                      |
     |                   tracking/<session>/masks          |
     |                                                     |
     +-------------------------+---------------------------+
                               | 07 labels: tick -> frame through the frame's OCR second,
                               |    vote per calf and second (08 writes the release format)
                               v
                     annotations/labels/

 08 also builds the tar shards of frames/ and tracking/ and uploads the release to Hugging Face.
 09 counts OCR coverage, implausible OCR readings and labels on such readings: metadata/qc/.
```

| Notebook | What it does | Dataset files it produced |
|---|---|---|
| `notebooks/01_decode_frames.py` | Decodes the session's videos with OpenCV, keeps every 5th frame, writes `<NNNNNNN>.jpg` (7-digit index running across the session's videos) into `video1`, `video2`, ... folders of 3,000 frames | `frames/` (08 packs each `video<k>` folder into one tar) |
| `notebooks/02_ocr_camera_clock.py` | Reads the camera clock burned into every frame (fixed ROI per camera, Otsu threshold, Tesseract `--psm 6`), distributed with Spark | input of `metadata/frame_index/` |
| `notebooks/03_build_frame_index.py` | One row per decoded frame: chunk, OCR timestamp (ISO 8601, local camera time; only for `Success` readings, never corrected or interpolated), OCR status, OCR source | `metadata/frame_index/` |
| `notebooks/04_initial_prompts_yolo.py` | YOLO12x detections on the first frame of the first tracked chunk; boxes kept, corrected and named by hand; writes `<calf name>: x, y, w, h` | `tracking/<session>.tar` → `initial_prompts/` |
| `notebooks/05_track_calves_samurai.py` | SAMURAI (SAM 2.1 hiera-large) per 3,000-frame chunk, each chunk prompted with the last box of the previous chunk; per frame and calf a box `[x, y, w, h]` and external contours | `tracking/<session>.tar` → `masks/` |
| `notebooks/06_prepare_annotations.py` | Per-calf state table at 0.1-s ticks (primary and secondary state, three-class label) from an Observer XT export | none: intermediate table, input of 07 |
| `notebooks/07_generate_labels.py` | Attaches the ticks to frames through the frame's OCR second, votes each calf's behaviour per second, one row per frame and calf, drops `out of view` | `annotations/labels/` (written by 08) |
| `notebooks/08_package_release.py` | Builds deterministic tar shards, the event logs (observer initials replaced by `O1`-`O5`, common columns, `calf_id`, `observation_start`) and the release format of the labels, uploads in resumable batches, writes `metadata/files_manifest.csv` | `frames/`, `tracking/`, `annotations/`, `metadata/files_manifest.csv` |
| `notebooks/09_qc_tables.py` | OCR coverage, plausibility of the OCR readings, labels on implausible readings, OCR re-run check (sample, reproducibility, contact sheets for reading the clock by eye) | `metadata/qc/` |
| `config/sessions.yaml` | Per-session inputs and parameters (video lists, frame folders, OCR ROI and clock format, QC windows, prompt files, tracked chunks and calves, Observer XT exports) | |
| `config/install_ocr.sh` | Cluster init script: Tesseract 4.1.1 and OpenCV for notebooks 02 and 09 | |

The other metadata tables (`metadata/sessions.csv`, `videos.csv`, `calves.csv`, `ethogram.csv`),
`metadata/known_issues.md` and the README files document the release and are part of the dataset itself.

## Annotations

The observers' Observer XT exports were checked and corrected by the authors (placeholder rows, a calf-name variant,
mistyped observation dates, merging of partial exports; no behaviour code, event time or duration was changed; every
change is listed in the dataset's `annotations/README.md`). These corrected exports are the input of the whole
pipeline: notebook 06 builds the per-calf state tables from them, and notebook 08 publishes them, with the observers'
initials replaced by pseudonymous codes, as `annotations/event_logs/`. The event logs therefore hold exactly the
annotations from which the labels were made. The per-calf state tables at 0.1-s ticks are an intermediate step and
are not published; notebook 06 rebuilds them from the event logs (`layout: release`).

## Time base: the burned-in camera clock

The recorders drop frames, so the time of a frame cannot be derived from its index, from its position in the video
file or from the nominal frame rate. Every frame therefore gets the wall-clock time printed on it by the camera, read
by OCR (notebook 02). Annotations are linked to frames only through this clock (notebook 07): a 0.1-s annotation tick
at time `t` is attached to every frame whose OCR second `s` satisfies `s <= t < s + 1 s`. Frames whose clock could
not be read carry no timestamp and no label. OCR readings are released as read; `metadata/qc/` counts implausible
readings, which users should screen against neighbouring frames (for example with a rolling median, as in notebook 09).

## Sessions

| Session | Frames | OCR ROI (x1, y1, x2, y2), clock format | Tracked calves (chunks) | Labels |
|---|---|---|---|---|
| `Tol_2024-04-18` | annotation only (pilot, one focal calf) | | | |
| `Tol_2024-04-24` | 89,571 (2304×1296, 25 fps source) | 1010, 0, 1273, 40; `MM/DD/YYYY HH:MM:SS` | 3 (1-30) | yes |
| `Tol2_2024-06-11_AM` | 111,860 (frames from 0015001) | 463, 33, 767, 65; `DD/MM/YYYY HH:MM:SS` | 2 (6-43) | yes |
| `Tol2_2024-06-11_PM` | 177,111 | 465, 30, 770, 65 | 2 (3-60) | yes |
| `Tol2_2024-06-12` | 80,884 | 463, 33, 767, 65 | prompts only | |
| `Tol3_2025-05-12` | 143,880 (1920×1080, 12 fps source) | 725, 20, 1295, 90 | prompts only | |
| `Eem_2024-06-18` | 180,485 | 465, 30, 770, 65 | 5 (19-61) | |
| `Eem_2024-06-19` | 322,319 | 465, 30, 770, 65 | 1 (1-108) | yes (06:03-11:23) |
| `Eem_2024-06-20` | annotation only | | | |

`config/sessions.yaml` holds the full parameters, including the ordered video list of every decoded session. The OCR
tables of `Tol2_2024-06-12`, `Tol3_2025-05-12` and `Eem_2024-06-19` were produced (again) with the code of notebook 02
after the first processing, because they were missing or had been overwritten; the Tol3 ROI is the only parameter
adapted, for its 1920×1080 overlay. For `Eem_2024-06-19` the labels were computed with the frame-second pairs kept
from the first OCR run of that session; these agree with the re-run OCR in `metadata/frame_index/` for all 14,194
labelled frames. Known gaps and quirks of each session are listed in the dataset's `metadata/known_issues.md`.

## Running the notebooks

### Databricks
The notebooks are Databricks notebooks in SOURCE format: `.py` files with a `# Databricks notebook source` header,
cells separated by `# COMMAND ----------` and `%pip`, `%sh` and `%md` cells written as `# MAGIC` lines. Clone the
repository into a Databricks Git folder (or import the files) and open them as notebooks. Parameters are notebook
widgets (`session_id`, `sessions`, `data_root`, `output_root`, `layout`, ...), with defaults from
`config/sessions.yaml`. Notebooks 02, 03, 06 and 07 use Spark; notebook 05 needs a GPU. `data_root` and
`output_root` must be paths that every node can read and write, for example a Unity Catalog volume (`/Volumes/...`)
or a DBFS FUSE path (`/dbfs/...`). The notebooks contain no credentials and mount no storage.

For notebook 02 (and the OCR re-run check of 09), attach `config/install_ocr.sh` as a cluster init script and set
the cluster environment variable `OMP_THREAD_LIMIT=1`. The re-run OCR tables were produced on Databricks Runtime
15.4 LTS ML, where this script installs Tesseract 4.1.1.

### Without Databricks
The files are also plain Python. Run the `%pip`/`%sh` cells by hand (they are comments in the `.py` files), then run
a notebook from the `notebooks/` folder, passing parameters as upper-case environment variables:

```bash
pip install -r requirements.txt
cd notebooks
SESSION_ID=Tol2_2024-06-12 DATA_ROOT=/data/Calf-Play-Behavior-Dataset OUTPUT_ROOT=/data/out python 02_ocr_camera_clock.py
```

Databricks-only calls and their plain equivalents:

| Databricks | Elsewhere |
|---|---|
| `dbutils.widgets.text/get` | environment variable with the upper-case name (handled by `_param()` in every notebook) |
| `dbfs:/` and `/dbfs/` paths | plain file paths; the notebooks use `os`, `shutil` and `file:` URIs for Spark |
| `dbutils.secrets.get(scope, key)` (08) | environment variable `HF_TOKEN` |
| `dbutils.library.restartPython()` (05) | restart the interpreter after installing SAMURAI |
| `display(image)` (04) | `matplotlib` (also used) |
| cluster init script (02, 09) | `apt-get install tesseract-ocr` (4.1.1 on Ubuntu 22.04) and `pip install opencv-python==4.11.0.86 pytesseract` |
| Databricks Spark | local `pyspark` (`SparkSession.builder.getOrCreate()`) |

### Data layout
`paths.layout: release` (default) reads the dataset as downloaded from Hugging Face, after extracting the tar shards:

```python
import pathlib, re, tarfile

root = pathlib.Path("/data/Calf-Play-Behavior-Dataset")
for tar in sorted(root.glob("frames/*/*_video*.tar")):          # -> frames/<session>/video<k>/<NNNNNNN>.jpg
    k = int(re.search(r"_video(\d+)\.tar$", tar.name).group(1))
    with tarfile.open(tar) as tf:
        tf.extractall(tar.parent / f"video{k}", filter="data")
for tar in sorted(root.glob("tracking/*.tar")):                  # -> tracking/<session>/{initial_prompts,masks}/
    with tarfile.open(tar) as tf:
        tf.extractall(root / "tracking", filter="data")
```

`paths.layout: original` describes the folder layout of the storage on which the pipeline ran
(`<visit>/Videos/`, `<visit>/Decoded Frames/<folder>/video<k>/`, ...). Every notebook writes below `output_root`.

Re-running a notebook on the same inputs gives the same table content. The bytes of gzip-compressed outputs
(`*.csv.gz`) can differ with the zlib version, so compare decompressed content; the sha256 values in the dataset's
`metadata/files_manifest.csv` are those of the published (compressed) files.

With the public dataset alone, notebooks 01-07 and 09 run in the `release` layout (07 with
`calf_files_root=tracking`, see below). The event-log step of notebook 08 needs the observers' exports before
pseudonymisation, which are not public; it is included to document how the published files were made. Its tar,
label and upload steps work on any data.

## Running OCR or tracking on more frames or calves

The dataset is released as it was produced; no further frames were OCR-read and no further calves were tracked.
Users who need more can run the same code:

* **More frames.** Only 7 annotated sessions were decoded, but all videos are released. Add a session to
  `config/sessions.yaml` with its ordered video list, the visit's OCR ROI and clock format (one camera per visit, so
  the ROI of another session of the same visit applies; two ROIs were used for Tol2, compare both on a sample) and run
  notebooks 01, 02 and 03. Check a sample of OCR readings by eye before using them (notebook 09, part
  `ocr_rerun_check`, writes contact sheets): daylight glare on the Tol2 clock causes many unreadable frames.
* **More calves or chunks.** Tracks exist only for the calves and chunks listed above (`Tol2_2024-06-12` and
  `Tol3_2025-05-12` have initial prompts but no tracks). Make a prompt file for the first frame of the first chunk
  to track with notebook 04 and run notebook 05 with `chunks=<first>-<last>` and `prompt_path=<file>`. Tracks are
  chained chunk by chunk without an identity check, so inspect them.
* **Labels for other calves, periods or sessions.** Add the session to `annotation_exports` in `config/sessions.yaml`
  (with `time_relative_parser: timedelta` for exports whose `Time_Relative_hmsf` reads `0 days ...`), run notebook 06
  (`layout: release`, reads `annotations/event_logs/`) and then notebook 07 with `sessions=<session_id>` and
  `calf_files_root=tracking` to label every frame in which a calf has a non-empty tracking box. The released labels used a narrower set of calf-frames: those
  with a per-calf file of the play-behaviour study pipeline (see notebook 07). Calf names in the tracks must match the
  `Subject` names of the event logs (`metadata/calves.csv` lists every spelling).

## Software versions

| Step | Environment |
|---|---|
| 01 decoding | `opencv-python` installed unpinned at run time (2024-2025; version not recorded); 4.11.0.86 recommended |
| 02 OCR | Tesseract 4.1.1 with `opencv-python` 4.11.0 (the runs of one session printed OpenCV 4.8.1); re-runs on Databricks Runtime 15.4 LTS ML with `opencv-python` 4.11.0.86 and `OMP_THREAD_LIMIT=1`, identical to the first tables on 120 test frames |
| 03 frame index | pandas and PySpark on Databricks Runtime 15.4 LTS ML |
| 04 initial prompts | Ultralytics `yolo12x.pt` (COCO), Ultralytics version not recorded |
| 05 tracking | SAMURAI (https://github.com/yangchris11/samurai, cloned at run time, commit not recorded) with the SAM 2.1 `sam2.1_hiera_large.pt` checkpoint and `configs/sam2.1/sam2.1_hiera_l.yaml`; one NVIDIA GPU; PyTorch version not recorded |
| 06 annotations | PySpark (Databricks runtime version not recorded), pandas with openpyxl |
| 07 labels | Databricks Runtime 17.3 LTS ML (Python 3.12.3, pandas 2.2.3) |
| 08 packaging | Databricks Runtime 17.3 LTS ML, `huggingface_hub` 2.0.0, `hf_xet` 1.6.0, openpyxl |
| 09 QC | pandas and NumPy; OCR re-run check as 02 |

Model weights are not included: Ultralytics downloads `yolo12x.pt` automatically, and the SAMURAI repository provides
`checkpoints/download_ckpts.sh` for the SAM 2.1 checkpoints. Put both in `model_root`.

## Citation

If you use the data or this code, please cite the dataset and its Data Descriptor:

> Yang, H., Lesscher, H., Liu, E. & Hostens, M. Video recordings and play behaviour annotations of group-housed dairy
> calves on two Dutch farms. Hugging Face https://doi.org/10.57967/hf/10695 (2026).

> Yang, H., Lesscher, H., Liu, E. & Hostens, M. Video recordings and play behaviour annotations of group-housed dairy
> calves on two Dutch farms. *Scientific Data* (submitted).

```bibtex
@misc{yang2026calfplaydata,
  author    = {Yang, Haiyu and Lesscher, Heidi and Liu, Enhong and Hostens, Miel},
  title     = {Video recordings and play behaviour annotations of group-housed dairy calves on two Dutch farms},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/10695},
  url       = {https://huggingface.co/datasets/Sonam5/Calf-Play-Behavior-Dataset}
}
```

## License

Code: MIT (see `LICENSE`). Data: CC0 1.0 (see the dataset repository).
