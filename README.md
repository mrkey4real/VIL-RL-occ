VIL-RL-occ: Video QA with Qwen2-VL (OpenVINO, CPU)

This project performs video Q&A/description on CPU using OpenVINO-accelerated Qwen2-VL-2B INT4. It uniformly samples frames per time segment and runs multiple questions per segment.

Repository: https://github.com/mrkey4real/VIL-RL-occ

Features
- CPU-only inference with OpenVINO IR models (Qwen2-VL-2B INT4)
- Uniform, segment-based frame sampling (configurable segment length and frames per segment)
- Simple one-file runner (vil.py) with English-only comments
- Conservative defaults for speed and memory on CPU

Requirements
Install via pip (CPU wheels):

pip install -q "transformers>=4.45" "torch>=2.1" "torchvision" "qwen-vl-utils" "Pillow"
pip install -qU "openvino>=2024.4.0" "nncf>=2.13.0"
# Optional for faster local video decoding
pip install -q decord

Note (Windows): Torch CPU wheels install automatically from PyPI. If you see a message that torchvision video decoding is deprecated, that's fine (we read frames via OpenCV).

How it works
- On first run, vil.py downloads a ready-to-use OpenVINO bundle (IR + helpers) to:
  - %USERPROFILE%/ov_qwen2vl_cpu_demo/qwen2vl_ov_2b_int4
- The script imports the OpenVINO model class from that bundle and runs generation with Transformers.
- Video is read once. We split the timeline into fixed-length segments and uniformly sample up to N frames per segment (saved as JPEGs). We then run QA per segment.

Usage
1) Place a video under video/ (we intentionally ignore large video files in git).
2) Edit the top of vil.py:
   - VIDEO_PATH: path to your local video (e.g. ./video/xxx.avi)
   - QUESTIONS: list of text questions
   - SEGMENT_SECONDS: segment length in seconds
   - FRAMES_PER_SEGMENT: max frames per segment (uniform sampling)
   - RESIZED_HEIGHT/RESIZED_WIDTH: visual resolution fed to the model (smaller = faster)
   - MAX_NEW_TOKENS: max generated tokens (smaller = faster)
3) Run:

python vil.py

You will see per-segment headers and answers printed to stdout.

Key parameters and trade-offs
- SEGMENT_SECONDS: smaller → more segments → more total QA calls; larger → fewer segments.
- FRAMES_PER_SEGMENT: larger → better visual coverage per segment but slower and more memory.
- RESIZED_HEIGHT/RESIZED_WIDTH: controls visual token budget; scale down to speed up.
- MAX_NEW_TOKENS: generation length; reduce for faster responses.

A practical CPU-friendly starter:
- SEGMENT_SECONDS = 10
- FRAMES_PER_SEGMENT = 4
- RESIZED_HEIGHT = 224, RESIZED_WIDTH = 336
- MAX_NEW_TOKENS = 64

Notes
- The model import path is added at runtime via sys.path.insert(0, ...), so your IDE may warn about unresolved imports; runtime is fine.
- Frame JPGs are saved to %USERPROFILE%/ov_qwen2vl_cpu_demo/frames/segments_<video_stem>/... and are not tracked by git.

License
This repository contains glue code. Qwen2-VL and OpenVINO assets follow their original licenses. Refer to the upstream model card and OpenVINO license for details.

