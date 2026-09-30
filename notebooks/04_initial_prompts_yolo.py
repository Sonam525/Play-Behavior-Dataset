# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Initial calf boxes for tracking (YOLO12x)
# MAGIC
# MAGIC SAMURAI (notebook 05) needs one box per calf on the **first frame of the first chunk it tracks** (frame index 0 of
# MAGIC that `video<k>` folder). These initial prompts were made as follows:
# MAGIC 1. run the Ultralytics `yolo12x.pt` detector (COCO-pretrained) with default settings on that frame and print every
# MAGIC    detection as `[x, y, w, h]` (top-left corner, width, height, in pixels);
# MAGIC 2. optionally keep only class `cow` with confidence > 0.55 (used for `Tol2_2024-06-11_PM`, frame 0006001);
# MAGIC 3. look at the frame, keep the boxes of the pen's own calves (calves in neighbouring pens are also detected),
# MAGIC    correct or draw boxes by hand where the detector missed a calf, and name each box with the calf's identity;
# MAGIC 4. write one line per calf, `<calf name>: x, y, w, h`, to `BoundingBoxes_<...>.txt`.
# MAGIC
# MAGIC The released prompt files are in the dataset's `tracking/<session_id>.tar` (`<session_id>/initial_prompts/`).
# MAGIC Calf names and their spellings are listed in `metadata/calves.csv`.
# MAGIC
# MAGIC Environment: `ultralytics` was installed unpinned in 2025 (the exact version was not recorded; any Ultralytics
# MAGIC release that supports YOLO12 loads `yolo12x.pt`), with `torch`/`torchvision` of the Databricks ML runtime.
# MAGIC
# MAGIC Output: `<output_root>/initial_prompts/<session_id>/<prompt_file>`.

# COMMAND ----------

# MAGIC %pip install torch torchvision opencv-python ultralytics pyyaml

# COMMAND ----------

# -------------------------------------------------------------------
# 0) CONFIGURATION
# -------------------------------------------------------------------
import os

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
MODEL_ROOT = _param("model_root", CFG["paths"]["model_root"])
LAYOUT = _param("layout", CFG["paths"]["layout"])
SESSION_ID = _param("session_id", "Tol2_2024-06-11_PM")
FIRST_CHUNK = _param("first_chunk", "")                 # default: first tracked chunk of the session
COW_FILTER = _param("cow_filter", "auto")              # 'auto' (from config), 'yes' or 'no'

S = CFG["sessions"][SESSION_ID]
frames_root = CFG["layouts"][LAYOUT]["frames_dir"].format(
    data_root=DATA_ROOT, visit=S["visit"], session_id=SESSION_ID, decoded_folder=S["decode"]["folder"])
chunks = (S.get("tracking") or {}).get("chunks") or [1, S["decode"]["n_chunks"]]
k = int(FIRST_CHUNK) if FIRST_CHUNK else int(chunks[0])
chunk_dir = os.path.join(frames_root, f"video{k}")
first_frame = sorted(f for f in os.listdir(chunk_dir) if f.lower().endswith((".jpg", ".jpeg")))[0]

# 2. DEFINE PATHS TO MODEL WEIGHTS AND IMAGE FRAME
model_weights_path = os.path.join(MODEL_ROOT, "yolo12x.pt")   # downloaded automatically by Ultralytics if missing
image_path = os.path.join(chunk_dir, first_frame)
filt = (S.get("tracking") or {}).get("prompt_detection_filter")
use_cow_filter = (COW_FILTER == "yes") or (COW_FILTER == "auto" and filt is not None)
min_conf = float(filt["min_confidence"]) if filt else 0.55
print(SESSION_ID, image_path, "cow filter:", use_cow_filter)

# COMMAND ----------

# ------------------------------------------------------------------------------
# 1. IMPORT LIBRARIES & SET INLINE PLOTTING
# ------------------------------------------------------------------------------
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------------------
# 3. INITIALIZE THE YOLOv12 MODEL
# ------------------------------------------------------------------------------
model = YOLO(model_weights_path if os.path.exists(model_weights_path) else "yolo12x.pt")
cow_class_idx = next(idx for idx, name in model.names.items() if name.lower() == "cow")

# ------------------------------------------------------------------------------
# 4. RUN THE DETECTOR ON THE IMAGE (default settings)
# ------------------------------------------------------------------------------
results = model(image_path)

# ------------------------------------------------------------------------------
# 5. PRINT OUT THE DETECTIONS (with bounding boxes in [x, y, w, h] format)
#    optional filter: class 'cow' with confidence > 0.55
# ------------------------------------------------------------------------------
detections = []  # tuples (x1, y1, w, h, conf, class_name)
print("Detected Objects:")
for result in results:
    if result.boxes is None:
        continue
    for box in result.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if use_cow_filter and (cls != cow_class_idx or conf <= min_conf):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()   # originally [x1, y1, x2, y2]
        w, h = x2 - x1, y2 - y1
        class_name = model.names[cls] if model.names and cls in model.names else str(cls)
        detections.append((x1, y1, w, h, conf, class_name))

for idx, (x1, y1, w, h, conf, class_name) in enumerate(detections, start=1):
    print(f"Object {idx}: {round(x1,2)}, {round(y1,2)}, {round(w,2)}, {round(h,2)}; "
          f"Class: {class_name}, Confidence: {conf:.2f}")

# ------------------------------------------------------------------------------
# 6. VISUALIZE THE BOUNDING BOXES ON THE ORIGINAL IMAGE
# ------------------------------------------------------------------------------
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load the image from {image_path}")

for idx, (x1, y1, w, h, conf, class_name) in enumerate(detections, start=1):
    x1_i, y1_i = int(x1), int(y1)
    x2_i, y2_i = int(x1 + w), int(y1 + h)
    label = f"{idx} {class_name}: {conf:.2f}"
    # draw bounding box (green rectangle)
    cv2.rectangle(image, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
    # draw label background for better visibility
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1_i, y1_i - th - baseline), (x1_i + tw, y1_i), (0, 255, 0), -1)
    # put the label text above the bounding box
    cv2.putText(image, label, (x1_i, y1_i - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Convert image from BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------------------------
# 7A. DISPLAY THE PLOT USING MATPLOTLIB
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis("on")
plt.title("Detection Results on Original Image")
plt.show()

# ------------------------------------------------------------------------------
# 7B. IF THE MATPLOTLIB PLOT DOES NOT APPEAR, USE THE DATABRICKS display() FUNCTION
# ------------------------------------------------------------------------------
try:
    display(Image.fromarray(image_rgb))  # noqa: F821  (Databricks only)
except NameError:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Name the boxes by hand and write the prompt file
# MAGIC Fill `NAMED_BOXES` with `calf name -> [x, y, w, h]`, taking the numbers of the detections above (or boxes drawn by
# MAGIC hand), then run the cell. The file name defaults to the session's `tracking.prompt_file` in `config/sessions.yaml`.

# COMMAND ----------

NAMED_BOXES = {
    # "Black 2644": [358.15, 132.82, 216.22, 177.67],
}

prompt_file = (S.get("tracking") or {}).get("prompt_file") or f"BoundingBoxes_{SESSION_ID}.txt"
out_dir = os.path.join(OUTPUT_ROOT, CFG["outputs"]["initial_prompts"], SESSION_ID)
if NAMED_BOXES:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, prompt_file), "w") as f:
        f.write("\n".join(f"{name}: {', '.join(str(round(float(v), 2)) for v in box)}"
                          for name, box in NAMED_BOXES.items()))
    print(f"Saved {len(NAMED_BOXES)} prompts to {os.path.join(out_dir, prompt_file)} "
          f"(for frame {first_frame} of video{k})")
else:
    print("NAMED_BOXES is empty: nothing written.")
