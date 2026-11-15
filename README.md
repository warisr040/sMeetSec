# sMeetSec — simply handle IT Meeting Secretary · v0.3

Offline meeting secretary for Windows. Everything runs on your PC; nothing is uploaded.

1. **Transcribe** a meeting recording (audio *or* video), or transcribe **live** from the microphone / Teams-Zoom audio.
2. **Identify speakers** ("Speaker 1, 2, …"), rename them once.
3. **Translate & report** in **Japanese · English · Thai · Chinese**: summary, key points, decisions, action items (with owners), open questions, participants, plus a line-by-line translated transcript.

Outputs per meeting: `transcript.txt / .srt / .json`, `transcript_<lang>.txt`, `report.html` (all languages, printable), `report_<lang>.docx` and `.md`.

---

## 1. Install — no Python, no Ollama

1. Download **`sMeetSec-v0.3.x-windows.zip`** from the GitHub **Releases** page (or the latest *build-windows* run under **Actions**), unzip it anywhere, and double-click **`sMeetSec.exe`**.
2. On first start the **Models** window opens: press **Download missing models** (needs internet once):
   - speech recognition (Whisper, 0.5–3 GB depending on the PC),
   - speaker identification (46 MB),
   - translation & reports (Qwen3.5 4B ≈ 3 GB, or 9B ≈ 6 GB on PCs with 16 GB RAM+).
3. Done — from now on sMeetSec works fully offline.

**PC without internet:** copy the `models` folder from a PC where sMeetSec already downloaded them into the new sMeetSec folder.

Inside the package: `sMeetSec.exe` (window), `backend\` (the engine, Python is built in), `llama\` (llama.cpp translation engine), `models\`, `data\` (your meetings). Needs the Microsoft Edge WebView2 Runtime (already in Windows 10/11).

**Optional — Ollama:** Settings → *Translation engine* → Ollama, if you prefer to use an Ollama installation instead of the built-in engine.

**Developers:** `setup_windows.bat` + `run.bat` run the app from source with a private Python in `runtime\`.

## 2. Use

- **Upload recording** → drop an mp3/m4a/wav/mp4/mkv/mov… → choose the spoken language (Auto, *Mixed languages*, Japanese, English, Thai, Chinese) and the number of speakers (Auto, Off, 2–8) → *Start transcription*. Tick *Translate & make report automatically* to get everything in one go.
- **Transcript tab**: the speaker bar on top renames a speaker on every line at once. Click any line to fix the text, *Save edits*. Set real names **before** the report — they become the owners of action items.
- **👥 Identify speakers** re-runs speaker detection (e.g. with a fixed number of speakers) without losing your text edits.
- **Translate & report**: tick the languages (+ transcript if you want the full translated transcript).
- **Files tab**: download everything. Meeting folders live in `data\meetings\`.
- **Theme** (palette button): Simple, Dark, Starry Night, Forest, Burning, Sakura, Snowy — same set as sGenko / sUML / sERP.
- **About** (click the logo): versions, privacy notice, third-party notices.

## 3. Hardware tiers (auto-detected)

| Tier | Batch model | Live mode | 1-hour meeting (rough) |
|---|---|---|---|
| NVIDIA GPU ≥ 6 GB | large-v3 (float16) | ✅ large-v3-turbo | ~3–6 min |
| NVIDIA GPU < 6 GB | large-v3-turbo | ✅ small | ~6–12 min |
| CPU only (e.g. Dell Latitude 5430) | large-v3-turbo (int8) | ❌ off (experimental option) | ~30–60 min |

Speaker identification adds roughly 2–5 min per hour on CPU. Translation + report adds about 5–20 min on CPU (Qwen3.5 4B is faster, 9B more accurate).

## 4. CPU guard

Settings → *CPU guard*:

- **Thread cap** (default 65 % of CPU threads) for speech recognition, speaker identification *and* the LLM. This is the hard limit.
- **Low priority**: sMeetSec and Ollama run *below normal* so other apps stay responsive.
- **Watchdog** (default **80 %**, 3-second average): when total CPU is over the limit
  - *Throttle*: work pauses between chunks until CPU drops, then continues (gives up after 2 min).
  - *Stop*: the job is cancelled.
  - *Live mode*: preview text pauses first; if CPU stays high for 15 s, live transcription stops but **recording continues**.
- Whisper processes ~30-second windows; the guard acts *between* windows, so a short peak can still happen. The thread cap keeps that peak below 100 %.

## 5. Live mode

- Sources: *Microphone*, *System audio* (Teams/Zoom on this PC — tick **Share system audio**), or both. System-audio capture works best in a browser: use *Open in browser ↗* in the live screen.
- Live audio is always saved as `audio.wav`. After the meeting press **Full-quality transcription**, then **Translate & report**.
- While live mode runs, batch jobs wait, so the two never compete for the CPU/GPU.
- Tell participants that the meeting is recorded (PDPA).

## 6. Languages

- **Thai** is the weakest in standard Whisper. Optional Thonburian Whisper (Mahidol University):
  ```
  runtime\python\python.exe -m pip install "transformers[torch]" ctranslate2
  runtime\python\python.exe tools\convert_thai_model.py          # distilled, recommended for CPU laptops
  runtime\python\python.exe tools\convert_thai_model.py --large  # full large-v3, GPU recommended
  ```
  Then Settings → *Thai model folder* = `thai-distill-large-v3` (used automatically for Thai recordings).
- **Chinese**: Settings → *Chinese characters* = Simplified (简体, default) or Traditional (繁體). Steers both Whisper (which otherwise mixes scripts) and the translation/report output.
- **Glossary** (Settings): one term per line — names, products, part numbers, e.g. `Somchai`, `SecurityAccess`, `量産 = mass production`. Whisper uses them as hints, the translator keeps them exact.

## 7. Translation test

`tests\translation_cases.json`: 18 cases from simple to complicated, no context (dates/numbers, UDS/CAN terms, ambiguous Japanese/Thai/Chinese, code-switching, double negation), each with en / ja / th / zh references and a "what to check" note.

```
runtime\python\python.exe tools\translation_test.py
runtime\python\python.exe tools\translation_test.py --model qwen3.5:9b --model qwen3.5:4b
```

Writes `data\translation_test_<time>.html`: source / model output / reference side by side; red rows lost a code, number or name.

## 8. Translation model and licenses

Default: **Qwen3.5** 4B / 9B (Apache-2.0, fine for company use, 201 languages) run by the built-in **llama.cpp** engine (MIT). The GGUF file is downloaded from Hugging Face (bartowski/Qwen_Qwen3.5-*-GGUF, Q4_K_M). If you put another GGUF into `models\llm\` and choose it in Settings, check its license first (e.g. Qwen2.5-3B is non-commercial). See `legal\THIRD-PARTY-NOTICES.txt`.

## 9. Building

GitHub Actions (`.github/workflows/build-windows.yml`) builds the package on a Windows machine at every push to `main`:
tests → PyInstaller engine → check the frozen engine starts → latest llama.cpp Windows CPU build → `sMeetSec.exe` (Go) → zip.
Each build on `main` publishes (or updates) the GitHub Release for the version in `smeetsec/config.py`.

Local pieces: `desktop\` (Go source of `sMeetSec.exe`), `build_backend.bat` (PyInstaller engine on your own PC).

## 10. Project layout

```
sMeetSec.exe            desktop window (starts the engine)
run.py / run.bat        engine + browser
smeetsec/app.py         local web API (127.0.0.1 only)
smeetsec/hardware.py    hardware detection and tiers
smeetsec/governor.py    CPU guard
smeetsec/transcriber.py faster-whisper transcription
smeetsec/diarize.py     speaker identification (sherpa-onnx)
smeetsec/live.py        live transcription (WebSocket audio stream)
smeetsec/report.py      translation, report, md/html/docx export
smeetsec/llm.py         built-in llama.cpp engine / Ollama client
smeetsec/models.py      model status + one-click download
smeetsec/jobs.py        background job queue (one heavy job at a time)
smeetsec/static/        user interface
desktop/                Go source for sMeetSec.exe
tools/                  model download, Thai conversion, translation test
tests/                  automated mock test, translation cases, screenshots
legal/  logo/  docs/    notices, icons, release checklist
```

Developer mock mode (no models needed, fake transcript/LLM): `set SMEETSEC_MOCK=1` then `python run.py`. Automated test: `python tests\smoke_test.py`.

## 11. Known limits (v0.2)

- Speakers are assigned per transcript segment; if two people speak inside one segment it goes to the main speaker.
- *Mixed languages* mode detects language per section; quality is lower than a single fixed language.
- Translation quality depends on the local LLM; check with the translation test before trusting reports for important meetings.
