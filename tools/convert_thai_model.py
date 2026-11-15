"""Optional: convert a Thai fine-tuned Whisper (Thonburian Whisper, Mahidol University) for sMeetSec.

Needs internet once and about 6 GB free disk:
    pip install "transformers[torch]" ctranslate2
    python tools/convert_thai_model.py            # distilled turbo version (faster; best for CPU laptops)
    python tools/convert_thai_model.py --large    # full large-v3 version (best accuracy; GPU recommended)
Then set Settings > "Thai model folder" to the printed folder name.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main() -> None:
    large = "--large" in sys.argv
    src = "biodatlab/whisper-th-large-v3-combined" if large else "biodatlab/distill-whisper-th-large-v3"
    out = MODELS / ("thai-large-v3" if large else "thai-distill-large-v3")
    MODELS.mkdir(exist_ok=True)

    from transformers import AutoProcessor
    from ctranslate2.converters import TransformersConverter

    print(f"Converting {src} -> {out}")
    TransformersConverter(src).convert(str(out), quantization="int8_float16", force=True)

    # faster-whisper needs tokenizer.json + preprocessor_config.json inside the folder to stay offline.
    with tempfile.TemporaryDirectory() as tmp:
        proc = AutoProcessor.from_pretrained(src)
        proc.save_pretrained(tmp)
        for name in ("tokenizer.json", "preprocessor_config.json"):
            p = Path(tmp) / name
            if p.exists():
                shutil.copy(p, out / name)
            else:
                print(f"warning: {name} was not produced")
    print(f"\nDone. In sMeetSec Settings set 'Thai model folder' to:  {out.name}")


if __name__ == "__main__":
    main()
