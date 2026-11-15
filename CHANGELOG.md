# sMeetSec — changelog

## 0.3.0 (2026-11-15)
- **No Python, no Ollama needed**: the Windows package contains the engine (built with PyInstaller) and **llama.cpp** as the built-in translation engine. Unzip and run.
- **Models window**: shows what is installed and downloads the missing models with one click (or copy a `models` folder from another PC for offline installs).
- Built automatically on GitHub Actions (Windows), with the automated test and a check that the packaged engine starts.
- Closing sMeetSec also stops the translation engine; the engine frees its memory after 10 idle minutes.
- Ollama is still available as an option (Settings → Translation engine).

## 0.2.0 (2026-11-15)
- **One-click setup**: unzip and start `sMeetSec.exe`; the first start downloads a private Python, packages and models into the sMeetSec folder (no Python install, nothing system-wide).
- **Desktop window**: `sMeetSec.exe` opens the app in its own window (WebView2, same as sUML / sGenko), starts and stops the local engine, asks before closing during a live meeting or running job.
- **Speaker identification** (offline, sherpa-onnx): "Speaker 1, 2, …" after transcription, auto or fixed number of speakers, re-run on any meeting, rename a speaker once for every line, participants listed in reports.
- **Chinese (Mandarin)** as the 4th language; Simplified / Traditional setting for transcription and output.
- **Themes** like sGenko: Simple, Dark, Starry Night, Forest, Burning, Sakura, Snowy.
- **New app icon**; About window with privacy notice and third-party notices.
- Default translation model changed to **Qwen3.5** (4B / 9B, Apache-2.0). Qwen2.5-3B was dropped because its license is non-commercial. Thinking mode is switched off for speed.
- Translation test: 18 cases with en / ja / th / zh references.

## 0.1.0 (2026-11-15)
- First version: batch transcription (faster-whisper), live mode with hardware tiers, CPU guard (thread cap, low priority, 80 % watchdog), translation and reports (Ollama) in ja / en / th, Word / Markdown / HTML export, translation test cases.
