# sMeetSec — release checklist

## ✅ Done (v0.2)
- [x] Batch transcription of audio/video, live mode gated by hardware tier, full-quality pass after live
- [x] CPU guard: thread cap, low priority, 80 % watchdog (throttle / stop), live falls back to record-only
- [x] Speaker identification offline, rename once for all lines
- [x] Translation + report in ja / en / th / zh (Simplified or Traditional), Word / Markdown / printable HTML
- [x] Themes (sGenko family), app icon, desktop window (WebView2), About + privacy + third-party notices
- [x] Automated mock test (`tests/smoke_test.py`) and translation test (`tools/translation_test.py`)

## ⬜ Pack to do on the Dell Latitude 5430
1. `setup_windows.bat` → check that models download (Whisper turbo + speaker models) and `ollama pull qwen3.5:9b` (16 GB) or `qwen3.5:4b` (8 GB).
2. Start `sMeetSec.exe`. First start shows "Starting the local engine…" then the app.
3. Real tests and timings (write the numbers in this file):
   - 10-min Japanese recording: transcription time ____ min, speakers correct? ____
   - 10-min Thai recording: time ____, quality OK? ____ (if poor: `tools\convert_thai_model.py`)
   - 10-min Chinese / mixed recording: time ____
   - Translate & report (4 languages): time ____ min
   - CPU stays under 80 % in Task Manager while a job runs? ____
4. `tools\translation_test.py` → rate the rows; send me the HTML if many are "Wrong".
5. Live mode only matters on a PC with an NVIDIA GPU (on the 5430 it stays off unless forced in Settings).

## Later (ideas)
- Word-level timestamps to split a segment when the speaker changes mid-sentence
- Glossary (company / product names, e.g. ECU part numbers) fed to Whisper and the translator
- Search across all meetings
- Installer with Start-menu shortcut
