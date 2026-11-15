"""Run the ja/en/th translation test cases against the local LLM and write an HTML comparison.

Usage (Ollama must be running):
    python tools/translation_test.py                      # uses the model from settings / auto
    python tools/translation_test.py --model qwen3.5:9b --model qwen3.5:4b   # compare models
Output: data/translation_test_<time>.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smeetsec import config, llm, report  # noqa: E402

L = config.LANG_NAMES


def run(models: list[str]) -> Path:
    cases = json.loads((ROOT / "tests" / "translation_cases.json").read_text(encoding="utf-8"))["cases"]
    results = []
    for model in models:
        if model:
            config.save_settings({"llm_model": model})
        name = model or "auto"
        for c in cases:
            for tgt in [x for x in ("en", "ja", "th", "zh") if x != c["src"]]:
                t0 = time.time()
                try:
                    out = report.translate_text(c["text"], c["src"], tgt)
                except Exception as e:
                    out = f"ERROR: {e}"
                missing = [k for k in c.get("must_keep", []) if k not in out]
                results.append({"model": name, "id": c["id"], "level": c["level_name"], "src": c["src"],
                                "tgt": tgt, "text": c["text"], "ref": c["ref"].get(tgt, ""), "out": out,
                                "check": c["check"], "missing": missing, "sec": round(time.time() - t0, 1)})
                print(f"[{name}] {c['id']} {c['src']}->{tgt} {results[-1]['sec']}s"
                      + (f"  MISSING {missing}" if missing else ""))
    return write_html(results)


def write_html(results: list[dict]) -> Path:
    e = html.escape
    rows = []
    for r in results:
        bad = " class='bad'" if r["missing"] or r["out"].startswith("ERROR") else ""
        rows.append(
            f"<tr{bad}><td>{e(r['model'])}</td><td><b>{e(r['id'])}</b><br><small>{e(r['level'])}</small></td>"
            f"<td>{L[r['src']]} → {L[r['tgt']]}</td><td>{e(r['text'])}</td><td>{e(r['out'])}</td>"
            f"<td>{e(r['ref'])}</td><td><small>{e(r['check'])}</small>"
            + (f"<br><b>Missing: {e(', '.join(r['missing']))}</b>" if r["missing"] else "")
            + f"</td><td>{r['sec']}s</td><td><select><option></option><option>OK</option>"
              "<option>Minor</option><option>Wrong</option></select></td></tr>")
    page = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>sMeetSec translation test</title>
<style>body{{font-family:'Segoe UI','Leelawadee UI','Yu Gothic UI',sans-serif;margin:20px;font-size:14px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccd;padding:6px;vertical-align:top}}
th{{background:#eef2f8;position:sticky;top:0}} tr.bad{{background:#fff1f0}}</style></head><body>
<h2>sMeetSec translation test - {time.strftime('%Y-%m-%d %H:%M')}</h2>
<p>Model output vs. human reference. Rate each row (OK / Minor / Wrong). Red rows lost a code, number or name.</p>
<table><tr><th>Model</th><th>Case</th><th>Direction</th><th>Source</th><th>Model output</th><th>Reference</th>
<th>What to check</th><th>Time</th><th>Rating</th></tr>{''.join(rows)}</table></body></html>"""
    config.ensure_dirs()
    p = config.DATA_DIR / f"translation_test_{time.strftime('%Y%m%d_%H%M%S')}.html"
    p.write_text(page, encoding="utf-8")
    print("Wrote", p)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="Ollama model name (repeat to compare)")
    a = ap.parse_args()
    run(a.model or [""])
