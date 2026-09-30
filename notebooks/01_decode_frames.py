# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Decode frames
# MAGIC
# MAGIC Decodes the source videos of one session into JPEG frames, exactly as for the released `frames/` folder.
# MAGIC
# MAGIC * Videos are read in the order listed in `config/sessions.yaml` (`sessions.<id>.decode.videos`).
# MAGIC * Every 5th frame read by OpenCV is kept (`stride = 5`); the counter restarts for every video.
# MAGIC * Frames are written with `cv2.imwrite` (OpenCV default JPEG quality, no resizing) as `<NNNNNNN>.jpg`,
# MAGIC   a 7-digit index that runs on across all videos of the session.
# MAGIC * Frames are grouped in sub-folders `video1`, `video2`, ... of 3,000 frames each
# MAGIC   (frame N goes to `video{(N-1)//3000+1}`). The released tars `frames/<session>/<session>_video<NNN>.tar`
# MAGIC   hold one such folder each.
# MAGIC * Numbering resumes after the highest frame index already present in the output folder.
# MAGIC
# MAGIC **Frame times.** The recorder drops frames, so a frame's index (or its position in the video) does not give
# MAGIC its time. Frame times come from the camera clock burned into every frame (notebooks 02 and 03).
# MAGIC
# MAGIC Outputs: `<output_root>/frames/<session_id>/video<k>/<NNNNNNN>.jpg`.

# COMMAND ----------

# MAGIC %pip install opencv-python==4.11.0.86 pyyaml

# COMMAND ----------

# -------------------------------------------------------------------
# 0) CONFIGURATION
# -------------------------------------------------------------------
import glob
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
LAYOUT = _param("layout", CFG["paths"]["layout"])
SESSION_ID = _param("session_id", "Eem_2024-06-19")

S = CFG["sessions"][SESSION_ID]
assert S.get("decoded"), f"{SESSION_ID} is an annotation-only session (no frames)"


def video_path(filename):
    pattern = CFG["layouts"][LAYOUT]["video"].format(data_root=DATA_ROOT, visit=S["visit"], filename=filename)
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else pattern          # a missing file is reported by the decoder


# 1) Define paths and parameters
# The list of video files of this session (order matters!)
annotated_video_files = [video_path(f) for f in S["decode"]["videos"]]

# Root output folder for decoded frames
output_root = os.path.join(OUTPUT_ROOT, CFG["outputs"]["frames"], SESSION_ID)
stride = CFG["decode_defaults"]["stride"]                                # 5: keep 1 frame out of every 5
max_frames_per_folder = CFG["decode_defaults"]["max_frames_per_folder"]  # 3000 frames per sub-folder

print(SESSION_ID, len(annotated_video_files), "videos ->", output_root)

# COMMAND ----------

import os
import cv2
import glob

# 2) Helper functions

def get_last_frame_index(output_root):
    """
    Scans existing subfolders under output_root and finds the highest frame index.
    Assumes frames are named as 7-digit numbers (e.g., 0000001.jpg).
    """
    max_idx = 0
    if not os.path.exists(output_root):
        return max_idx
    # Loop over subfolders (e.g., video1, video2, etc.)
    for subfolder in os.listdir(output_root):
        subfolder_path = os.path.join(output_root, subfolder)
        if os.path.isdir(subfolder_path):
            for file in glob.glob(os.path.join(subfolder_path, "*.jpg")):
                basename = os.path.basename(file)
                name, ext = os.path.splitext(basename)
                try:
                    idx = int(name)
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
    return max_idx

def get_output_folder(frame_index, output_root, max_frames_per_folder):
    """
    Determines the subfolder to use based on the current overall frame index.
    Each folder is named video1, video2, etc.
    """
    folder_num = (frame_index // max_frames_per_folder) + 1
    folder_name = f"video{folder_num}"
    folder_path = os.path.join(output_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# 3) Main decoding function

def decode_videos(video_files, output_root, stride=5, max_frames_per_folder=3000):
    # Ensure the output root directory exists
    os.makedirs(output_root, exist_ok=True)

    # Continue numbering from the last saved frame (if any)
    last_frame = get_last_frame_index(output_root)
    current_frame_index = last_frame  # Will be incremented before saving each new frame

    print(f"Starting at frame: {current_frame_index + 1:07d}.jpg")
    video_frame_mapping = {}  # To record frame ranges for each video

    # Process each video file in the order provided
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        video_start_frame = current_frame_index + 1  # Starting frame number for this video
        frame_count = 0  # Counter for frames read from the current video

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of the video

            # Save the frame based on the stride condition
            if frame_count % stride == 0:
                current_frame_index += 1
                folder_path = get_output_folder(current_frame_index - 1, output_root, max_frames_per_folder)
                filename = os.path.join(folder_path, f"{current_frame_index:07d}.jpg")
                cv2.imwrite(filename, frame)
            frame_count += 1

        video_end_frame = current_frame_index  # Ending frame number for this video
        video_frame_mapping[video_file] = (video_start_frame, video_end_frame)
        print(f"Video {video_file} processed: starts at {video_start_frame:07d}.jpg and ends at {video_end_frame:07d}.jpg")
        cap.release()

    return video_frame_mapping

# COMMAND ----------

# 4) Run the decoder on the session's videos

if not annotated_video_files:
    print("No annotated video files specified.")
else:
    mapping = decode_videos(annotated_video_files, output_root, stride, max_frames_per_folder)
    print("\nFrame range mapping per video:")
    for video, (start, end) in mapping.items():
        print(f"{video}: {start:07d}.jpg to {end:07d}.jpg")

# COMMAND ----------

# MAGIC %md
# MAGIC **Notes on the released sessions** (details in `config/sessions.yaml`):
# MAGIC * `Tol2_2024-06-11_AM`: frames 0000001-0015000 were deleted before release, so the released frames start at 0015001.
# MAGIC   Re-decoding the listed videos regenerates the full numbering (frame 0015001 is the first released frame).
# MAGIC * `Tol2_2024-06-11_PM`: the truncated video `Tol2_ch04_20240611_164646_172030.mp4` cannot be opened and is skipped.
# MAGIC * `Tol3_2025-05-12`: the source is 12 frames/s, so stride 5 gives 2.4 decoded frames/s.
# MAGIC * Two frames are missing from the released sessions (`Tol2_2024-06-11_AM` 0090726, `Eem_2024-06-19` 0095169).
# MAGIC * Decoding uses the video decoder bundled with `opencv-python`; a different OpenCV/FFmpeg build can return a
# MAGIC   different number of frames for damaged files. Compare the result with `metadata/frame_index/` before relying on
# MAGIC   the frame numbers of a re-decode.
