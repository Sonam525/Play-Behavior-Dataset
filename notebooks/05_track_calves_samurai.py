# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Track calves with SAMURAI (SAM 2.1)
# MAGIC
# MAGIC Segments and tracks every calf listed in the initial-prompt file (notebook 04) through the decoded frames, one
# MAGIC 3,000-frame chunk (`video<k>` folder) at a time, as for the released `tracking/` files.
# MAGIC
# MAGIC * Tracker: SAMURAI (https://github.com/yangchris11/samurai; its bundled `sam2` package), video predictor built
# MAGIC   from the SAM 2.1 checkpoint `sam2.1_hiera_large.pt` with config `configs/sam2.1/sam2.1_hiera_l.yaml`, on one GPU.
# MAGIC * Each chunk is a separate SAM 2 video: the boxes are added at frame 0 of the chunk and propagated through it.
# MAGIC   The first chunk uses the boxes of the prompt file; every later chunk is prompted with each calf's box on the
# MAGIC   **last frame of the previous chunk** (no identity re-check between chunks).
# MAGIC * Per frame and calf the mask is converted to a bounding box `[x, y, w, h]` (from the mask's pixel extent; `[0, 0, 0, 0]`
# MAGIC   for an empty mask) and to its external contours (`cv2.findContours`, `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`).
# MAGIC
# MAGIC Output per chunk and calf: `<output_root>/tracking/<session_id>/masks/video<k>/annotations_<calf name>.json`
# MAGIC `{"<NNNNNNN>.jpg": {"bounding_box": [x, y, w, h], "contour": [[[x, y], ...], ...], "object_name": "<calf name>"}}`.
# MAGIC (In the released tar the folders are named `video<NNN>`.)
# MAGIC
# MAGIC Environment: SAMURAI cloned at run time (commit not recorded) and installed with `pip install -e sam2`; the SAM 2.1
# MAGIC checkpoint is downloaded with `checkpoints/download_ckpts.sh` of the SAMURAI repository. PyTorch/torchvision versions
# MAGIC of the GPU runtime were not recorded.

# COMMAND ----------

# MAGIC %pip install torch torchvision matplotlib opencv-python lmdb pandas scipy loguru tikzplotlib jpeg4py pyyaml

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf /tmp/samurai
# MAGIC git clone https://github.com/yangchris11/samurai.git /tmp/samurai

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /tmp/samurai/sam2
# MAGIC pip install -e .
# MAGIC pip install -e ".[notebooks]"

# COMMAND ----------

# Databricks: restart Python so that the freshly installed sam2 package is importable
try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass

# COMMAND ----------

try:
    from sam2.build_sam import build_sam2_video_predictor
    print("Successfully imported build_sam_video_predictor from sam2.build_sam!")
except Exception as e:
    print("Error importing SAM2 module:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC Rough Annotations for multiple objects

# COMMAND ----------

import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import json
import time
import shutil

from sam2.build_sam import build_sam2_video_predictor

def load_txt(gt_path):
    """
    Loads bounding box prompts from a text file formatted as:
       <object_name>: x,y,w,h
    Returns a dictionary mapping object names to a tuple: ((x1,y1,x2,y2), 0).
    """
    if not osp.exists(gt_path):
        raise FileNotFoundError(f"File not found: {gt_path}")
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    prompts = {}
    for line in lines:
        parts = line.strip().split(":")
        if len(parts) != 2:
            raise ValueError("Each line must be formatted as <object_name>: x,y,w,h")
        obj_name = parts[0].strip()
        coords = parts[1].strip().split(",")
        if len(coords) != 4:
            raise ValueError("Coordinates must have four comma-separated values")
        x, y, w, h = map(float, coords)
        # Convert to (x1,y1,x2,y2)
        prompts[obj_name] = ((int(x), int(y), int(x+w), int(y+h)), 0)
    return prompts

def determine_model_cfg(model_path):
    """
    Returns the configuration file for the model.
    For example, if "large" is in the model_path, returns:
      "configs/sam2.1/sam2.1_hiera_l.yaml"
    """
    if "large" in model_path:
        return "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    """
    Validates that video_path is either a .mp4 file or a directory of JPEG images.
    """
    if osp.exists(video_path):
        return video_path
    raise ValueError("Invalid video_path format. Should be a .mp4 file or a directory of jpg frames.")

def process_folder(frames_dir, predictor, current_prompt, object_names):
    """
    Processes a single folder of frames.

    Arguments:
      frames_dir: directory of JPEG frames for this batch.
      predictor: the SAM2 predictor (pre-built).
      current_prompt: dict mapping obj_id to prompt bounding box ([x1,y1,x2,y2]) for frame 0.
      object_names: list of object names corresponding to obj_id.

    Returns:
      batch_annotations: dict { obj_id: { frame_name: annotation, ... } }
      new_prompt: dict mapping obj_id to new prompt (bounding box from the last frame) in [x1,y1,x2,y2] format.
    """
    # List and sort frame files in this folder.
    frame_files = sorted([osp.join(frames_dir, f) for f in os.listdir(frames_dir)
                          if f.lower().endswith((".jpg", ".jpeg"))])
    num_frames = len(frame_files)
    print(f"Processing {num_frames} frames in folder: {frames_dir}")

    # Initialize state for this folder.
    state = predictor.init_state(frames_dir, async_loading_frames=True, offload_video_to_cpu=True)

    # For each object, add the initial prompt at frame 0.
    for obj_id, bbox in current_prompt.items():
        predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=obj_id)

    batch_annotations = {obj_id: {} for obj_id in current_prompt.keys()}

    # Process each frame in this folder.
    for local_idx, object_ids, masks in predictor.propagate_in_video(state):
        frame_path = frame_files[local_idx]
        frame_name = osp.basename(frame_path)  # e.g., "0003501.jpg"
        global_frame = frame_name  # using filename as global identifier

        for obj_id, mask in zip(object_ids, masks):
            mask_np = mask[0].cpu().numpy()
            mask_binary = mask_np > 0.0
            nonzero_pixels = np.argwhere(mask_binary)
            if len(nonzero_pixels) == 0:
                bbox_coords = [0, 0, 0, 0]
            else:
                y_min, x_min = nonzero_pixels.min(axis=0).tolist()
                y_max, x_max = nonzero_pixels.max(axis=0).tolist()
                bbox_coords = [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, w, h]
            # Extract contour for precise annotation.
            contours, _ = cv2.findContours((mask_binary*255).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_list = [c.squeeze().tolist() for c in contours if len(c) > 2]
            batch_annotations[obj_id][global_frame] = {
                "bounding_box": bbox_coords,
                "contour": contour_list,
                "object_name": object_names[obj_id]
            }

    # Determine new prompt from the last frame of this folder.
    new_prompt = {}
    for obj_id in current_prompt.keys():
        last_frame = sorted(batch_annotations[obj_id].keys())[-1]
        ann = batch_annotations[obj_id][last_frame]
        bb = ann["bounding_box"]  # [x, y, w, h]
        new_prompt[obj_id] = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])

    del state
    gc.collect()
    torch.cuda.empty_cache()
    return batch_annotations, new_prompt

def main(args):
    # Determine the configuration file.
    model_cfg = determine_model_cfg(args.model_path)

    # Build the predictor (to be reused across folders).
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")

    # Get the list of frame folders from a comma-separated string (empty entries are ignored).
    folders = [folder.strip() for folder in args.frames_dirs.split(",") if folder.strip()]
    folders = [prepare_frames_or_path(folder) for folder in folders]

    # Load initial bounding box prompts.
    prompts = load_txt(args.txt_path)
    object_names = sorted(prompts.keys())
    obj_name_to_id = {name: idx for idx, name in enumerate(object_names)}

    # For the first folder, current_prompt comes from the file.
    current_prompt = {}
    for obj_name, prompt in prompts.items():
        current_prompt[obj_name_to_id[obj_name]] = prompt[0]

    # Process each folder sequentially.
    for folder in folders:
        folder_basename = osp.basename(osp.normpath(folder))
        output_dir = osp.join(args.output_base_dir, folder_basename)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Processing folder: {folder_basename} (Output dir: {output_dir})")

        batch_ann, new_prompt = process_folder(folder, predictor, current_prompt, object_names)
        # Check if we got any annotations
        for obj_id, ann in batch_ann.items():
            print(f"Object '{object_names[obj_id]}' has {len(ann)} frames annotated in folder '{folder_basename}'.")
            # Save JSON for this object in this folder.
            ann_filename = f"annotations_{object_names[obj_id]}.json"
            ann_full_path = osp.join(output_dir, ann_filename)
            with open(ann_full_path, "w") as f:
                json.dump(ann, f)
            print(f"Saved annotations for object '{object_names[obj_id]}' to {ann_full_path}")

        # Update current_prompt for next folder.
        current_prompt = new_prompt

        # Clear GPU memory between folders.
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Completed processing folder {folder_basename}.\n")

    del predictor
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
    print("Batch processing complete. All annotations saved.")

# COMMAND ----------

# -------------------------------------------------------------------
# CONFIGURATION (session, chunk range, prompt file, checkpoint)
# -------------------------------------------------------------------
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
SESSION_ID = _param("session_id", "Eem_2024-06-19")
CHUNKS = _param("chunks", "")          # e.g. "31-108"; default: tracking.chunks of the session
PROMPT_PATH = _param("prompt_path", "")  # default: the session's released prompt file

S = CFG["sessions"][SESSION_ID]
T = S.get("tracking") or {}
frames_root = CFG["layouts"][LAYOUT]["frames_dir"].format(
    data_root=DATA_ROOT, visit=S["visit"], session_id=SESSION_ID, decoded_folder=S["decode"]["folder"])
if CHUNKS:
    lo, hi = (int(x) for x in CHUNKS.split("-"))
elif T.get("chunks"):
    lo, hi = T["chunks"]
else:
    raise ValueError(f"{SESSION_ID}: no tracked chunks in the config; pass chunks='<first>-<last>'")
if not PROMPT_PATH:
    PROMPT_PATH = os.path.join(CFG["layouts"][LAYOUT]["initial_prompts_dir"].format(
        data_root=DATA_ROOT, visit=S["visit"], session_id=SESSION_ID, decoded_folder=S["decode"]["folder"]),
        T["prompt_file"])

args = argparse.Namespace(
    frames_dirs=",".join(os.path.join(frames_root, f"video{k}") for k in range(lo, hi + 1)),
    txt_path=PROMPT_PATH,
    model_path=os.path.join(MODEL_ROOT, "sam2.1_hiera_large.pt"),
    output_base_dir=os.path.join(OUTPUT_ROOT, CFG["outputs"]["tracking"], SESSION_ID, "masks"),
)
print(SESSION_ID, f"chunks video{lo}..video{hi}", "\n prompts:", args.txt_path, "\n ->", args.output_base_dir)

# COMMAND ----------

main(args)
