# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 15:54:41 2025

@author: qizixuan
"""

# qwen2vl_openvino_video_qa.py
# Purpose: Video QA/description on CPU using OpenVINO + Qwen2-VL-2B-INT4
# Usage: set VIDEO_PATH and QUESTIONS, then run this script.

from pathlib import Path
import sys, os
import cv2
import numpy as np
from huggingface_hub import snapshot_download

# ========== Configurable parameters ==========
VIDEO_PATH = "./video/CH02-2025-04-27-11-29-35_11-31-20.avi"   # <<< Put your video path here
QUESTIONS = [
    'What is the occupant doing?',
    'What is his metabolic activity level? Show me the answer and confidence in percentage.',
    'What is occupant wearing from top to bottom? Show me the answer and confidence in percentage.',
    'Is window open in boolean? Show me the answer and confidence in percentage.'
]
RESIZED_HEIGHT = 224         # Forced visual height (rounded to multiples of 28)
RESIZED_WIDTH  = 336         # Forced visual width (rounded to multiples of 28)
MAX_NEW_TOKENS = 64         # Max new tokens to generate (smaller = faster)
MAX_PIXELS     = RESIZED_HEIGHT * RESIZED_WIDTH  # Visual resolution cap (smaller = faster)
SEGMENT_SECONDS = 10         # Segment length in seconds for uniform sampling
FRAMES_PER_SEGMENT = 4       # Max frames per segment (uniformly sampled)
# ============================================

# 1) Download preconverted OpenVINO assets from Hugging Face (contains ov_qwen2_vl.py and IR models)
#    Using community 2B INT4 bundle (lightweight and CPU friendly)
REPO_ID = "cydxg/Qwen2-VL-2B-Instruct-OpenVINO-INT4"  # includes ov_qwen2_vl.py / gradio_helper.py / IR
WORKDIR = Path.home() / "ov_qwen2vl_cpu_demo"
LOCAL_REPO = WORKDIR / "qwen2vl_ov_2b_int4"
WORKDIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR = WORKDIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

if not LOCAL_REPO.exists():
    print("⏳ Downloading Qwen2-VL-2B OpenVINO IR & helpers from Hugging Face ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_REPO.as_posix(),
        local_dir_use_symlinks=False
    )
    print("✅ Files downloaded to:", LOCAL_REPO)

# 2) Import helper modules from the downloaded folder
sys.path.insert(0, LOCAL_REPO.as_posix())
from ov_qwen2_vl import OVQwen2VLModel
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# 3) Model directory (IR .xml/.bin files live here)
MODEL_DIR = LOCAL_REPO

# 4) Load OpenVINO model
print("⏳ Loading OpenVINO IR model ...")
model = OVQwen2VLModel(MODEL_DIR, device="CPU")
print("✅ Model loaded.")

# 5) Control visual token budget to speed up CPU inference
processor = AutoProcessor.from_pretrained(
    MODEL_DIR.as_posix(),
    min_pixels=256*28*28,
    max_pixels=MAX_PIXELS
)
model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

def extract_segments_uniform(video_path: str) -> list:
    """
    Read video once, split timeline by SEGMENT_SECONDS; uniformly sample up to
    FRAMES_PER_SEGMENT frames per segment.
    Returns: list[list[str]] of frame file URIs per segment.
    Uniform segment sampling only; no FPS-step sampling or global cap.
    """
    video_path = Path(video_path)
    out_root = FRAMES_DIR / f"segments_{video_path.stem}"
    if out_root.exists():
        for p in out_root.glob("**/*.jpg"):
            p.unlink()
    else:
        out_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path.as_posix())
    assert cap.isOpened(), f"Cannot open video: {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps and fps > 0 and total_frames > 0, f"Cannot get fps/frame_count for: {video_path}"

    duration_s = total_frames / fps
    num_segments = int(np.ceil(duration_s / SEGMENT_SECONDS))

    segments = []
    for seg_idx in range(num_segments):
        seg_start_s = seg_idx * SEGMENT_SECONDS
        seg_end_s = min((seg_idx + 1) * SEGMENT_SECONDS, duration_s)
        seg_start_f = int(round(seg_start_s * fps))
        seg_end_f = int(round(seg_end_s * fps))
        if seg_end_f <= seg_start_f:
            segments.append([])
            continue

        # Frames available in this segment
        frames_in_seg = seg_end_f - seg_start_f
        take_n = min(FRAMES_PER_SEGMENT, frames_in_seg)
        if take_n <= 0:
            segments.append([])
            continue

        # Evenly spaced indices (inclusive of endpoints)
        # Sample take_n indices in [seg_start_f ... seg_end_f-1]
        idxs = np.linspace(seg_start_f, seg_end_f - 1, num=take_n, dtype=int)

        # Seek and save frames
        out_dir = out_root / f"seg_{seg_idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        uris = []
        for j, fidx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok:
                continue
            out_file = out_dir / f"frame_{j:03d}.jpg"
            cv2.imwrite(out_file.as_posix(), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            uris.append(f"file://{out_file.as_posix()}")
        segments.append(uris)

    cap.release()
    return segments

assert os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"
print("🎬 Video:", VIDEO_PATH)

# Segment-based uniform sampling; read once and save frames
segments_frame_uris = extract_segments_uniform(VIDEO_PATH)
for seg_idx, frame_uris in enumerate(segments_frame_uris):
    if not frame_uris:
        continue
    print(f"\n== Segment {seg_idx} (<= {SEGMENT_SECONDS}s) | Frames: {len(frame_uris)} ==")
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        messages = [{
            "role": "user",
            "content": [
                {"type": "video",
                 "video": frame_uris,
                 "fps": FRAMES_PER_SEGMENT / SEGMENT_SECONDS,
                 "max_pixels": MAX_PIXELS,
                 "resized_height": RESIZED_HEIGHT,
                 "resized_width": RESIZED_WIDTH},
                {"type": "text", "text": q},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(output_text)
        print("\n" + "-"*60)

print("\n✅ All done.")