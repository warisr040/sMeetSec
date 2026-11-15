"""Translation (ja/en/th) and meeting report generation with the local LLM, plus file exports."""
from __future__ import annotations

import html
import json
import time

from . import config, llm, meetings
from .governor import governor
from .transcriber import fmt_time

L = config.LANG_NAMES


# ---------------------------------------------------------------- helpers
def transcript_lines(t: dict) -> list[str]:
    out = []
    for s in t.get("segments", []):
        spk = f"{s['speaker']}: " if s.get("speaker") else ""
        out.append(f"[{fmt_time(s['start'])}] {spk}{s['text']}")
    return out


def chunk_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    chunks, cur, n = [], [], 0
    for ln in lines:
        if cur and n + len(ln) > max_chars:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- translation
def translation_system_prompt(source: str, target: str) -> str:
    return (
        f"You are a professional meeting interpreter. Translate meeting transcript lines from "
        f"{config.lang_label(source) if source in L else 'the source language'} into {config.lang_label(target)}. "
        "Keep every line, keep the [mm:ss] timestamp and any 'Speaker:' prefix exactly, one output line per input line. "
        "Keep technical terms, product names, codes, numbers and people's names accurate; do not guess the "
        "spelling of a name in another script. When a sentence is ambiguous, translate it neutrally and do not "
        "add information that was not said. Output only the translated lines, no explanations."
        + glossary_hint()
    )


def glossary_hint(target: str | None = None) -> str:
    terms = config.glossary_terms()
    if not terms:
        return ""
    lines = [f"- {a} -> {b}" if b else f"- {a} (keep as written)" for a, b in terms]
    return "\nGlossary - use these spellings/translations exactly:\n" + "\n".join(lines)


def translate_text(text: str, source: str, target: str) -> str:
    return llm.chat(translation_system_prompt(source, target), "---\n" + text).strip()


def translate_transcript(t: dict, target: str, source: str, progress, cancel) -> list[str]:
    lines = transcript_lines(t)
    chunks = chunk_lines(lines, 1500)
    system = translation_system_prompt(source, target)
    out: list[str] = []
    for i, ch in enumerate(chunks):
        governor.checkpoint(cancel)
        progress(i / max(1, len(chunks)), f"Translating to {L[target]} ({i + 1}/{len(chunks)})")
        prev = "\n".join(out[-3:])
        user = (f"Previous translated lines for context (do not repeat):\n{prev}\n---\n" if prev else "---\n") + "\n".join(ch)
        res = llm.chat(system, user)
        res_lines = [x for x in res.splitlines() if x.strip()]
        out.extend(res_lines)
    return out


# ---------------------------------------------------------------- report
REPORT_KEYS = ["title", "summary", "key_points", "decisions", "action_items", "open_questions"]


def build_report(t: dict, source: str, progress, cancel) -> dict:
    lines = transcript_lines(t)
    chunks = chunk_lines(lines, 6000)
    notes = []
    sys_notes = (
        "You take meeting notes. From this part of a meeting transcript, extract notes as JSON with keys: "
        "points (list of important discussion points), decisions (list), "
        "actions (list of objects with task, owner, due - use empty string if unknown), "
        "questions (list of unresolved questions). Write the notes in English. Be factual, do not invent."
    )
    for i, ch in enumerate(chunks):
        governor.checkpoint(cancel)
        progress(0.1 + 0.5 * i / max(1, len(chunks)), f"Reading the meeting ({i + 1}/{len(chunks)})")
        notes.append(llm.chat_json(sys_notes, "\n".join(ch)))

    governor.checkpoint(cancel)
    progress(0.65, "Writing the report")
    sys_final = (
        "You are a meeting secretary. Combine the partial notes into one final meeting report as JSON with keys: "
        "title (short), summary (3-6 sentences), key_points (list), decisions (list), "
        "action_items (list of objects with task, owner, due), open_questions (list). "
        "Merge duplicates, keep it concise and factual, write in English."
        + glossary_hint()
    )
    rep = llm.chat_json(sys_final, json.dumps(notes, ensure_ascii=False))
    return normalize(rep)


def normalize(rep: dict) -> dict:
    out = {k: rep.get(k, [] if k not in ("title", "summary") else "") for k in REPORT_KEYS}
    out["title"] = str(out["title"] or "Meeting report")
    out["summary"] = str(out["summary"] or "")
    for k in ("key_points", "decisions", "open_questions"):
        out[k] = [str(x) for x in (out[k] or []) if str(x).strip()]
    items = []
    for a in out["action_items"] or []:
        if isinstance(a, dict):
            items.append({"task": str(a.get("task", "")), "owner": str(a.get("owner", "")),
                          "due": str(a.get("due", ""))})
        elif str(a).strip():
            items.append({"task": str(a), "owner": "", "due": ""})
    out["action_items"] = items
    return out


def translate_report(rep: dict, target: str) -> dict:
    if target == "en":
        return rep
    system = (
        f"Translate all text values of this JSON meeting report into {config.lang_label(target)}. "
        "Keep the same JSON keys and structure. Keep people's names and technical terms accurate. "
        "Return only JSON."
        + glossary_hint()
    )
    return normalize(llm.chat_json(system, json.dumps(rep, ensure_ascii=False)))


# ---------------------------------------------------------------- job entry point
def run_report_job(job, mid: str, languages: list[str], translate_transcript_too: bool) -> None:
    meta = meetings.get(mid)
    t = meetings.current_transcript(mid)
    if not t or not t.get("segments"):
        raise ValueError("No transcript yet. Transcribe the recording first.")
    st = llm.status()
    if st["engine"] == "ollama" and not st["running"]:
        raise llm.LLMError("Ollama is not running. Start Ollama, or switch Settings > Translation engine to Built-in.")
    if st["engine"] == "builtin" and not st["installed"]:
        raise llm.LLMError("No translation model installed. Open Models (top bar) and download it.")
    meetings.update(mid, status="reporting", error="")
    source = t.get("language") or "en"
    languages = [x for x in languages if x in L] or ["en"]

    def sub(a, b):
        return lambda f, msg: job.update(a + (b - a) * f, msg)

    base = build_report(t, source, sub(0.0, 0.45), job.cancel)
    people = []
    for s in t["segments"]:
        if s.get("speaker") and s["speaker"] not in people:
            people.append(s["speaker"])
    base["meta"] = {"meeting": meta["title"], "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(meta["created"])),
                    "duration": fmt_time(t.get("duration", 0)), "language": L.get(source, source),
                    "participants": ", ".join(people)}

    reports = {}
    for i, lang in enumerate(languages):
        governor.checkpoint(job.cancel)
        job.update(0.5 + 0.2 * i / len(languages), f"Report in {L[lang]}")
        r = translate_report({k: base[k] for k in REPORT_KEYS}, lang)
        r["meta"] = base["meta"]
        reports[lang] = r
        meetings.write_json(mid, f"report_{lang}.json", r)

    translations = []
    if translate_transcript_too:
        targets = [x for x in languages if x != source]
        for i, lang in enumerate(targets):
            a = 0.7 + 0.25 * i / max(1, len(targets))
            b = 0.7 + 0.25 * (i + 1) / max(1, len(targets))
            lines = translate_transcript(t, lang, source, sub(a, b), job.cancel)
            meetings.path(mid, f"transcript_{lang}.txt").write_text("\n".join(lines), encoding="utf-8")
            translations.append(lang)

    job.update(0.97, "Saving files")
    export_reports(mid, meta, reports, languages)
    meetings.update(mid, status="done", reports=languages, translations=translations, error="")


# ---------------------------------------------------------------- exports
UI_TEXT = {
    "en": {"summary": "Summary", "key_points": "Key points", "decisions": "Decisions",
           "action_items": "Action items", "open_questions": "Open questions", "task": "Task",
           "owner": "Owner", "due": "Due", "date": "Date", "duration": "Duration",
           "language": "Spoken language", "none": "None", "participants": "Participants"},
    "th": {"summary": "สรุป", "key_points": "ประเด็นสำคัญ", "decisions": "ข้อตัดสินใจ",
           "action_items": "สิ่งที่ต้องดำเนินการ", "open_questions": "คำถามที่ยังค้างอยู่", "task": "งาน",
           "owner": "ผู้รับผิดชอบ", "due": "กำหนดเสร็จ", "date": "วันที่", "duration": "ระยะเวลา",
           "language": "ภาษาที่ใช้ในการประชุม", "none": "ไม่มี", "participants": "ผู้เข้าร่วม"},
    "ja": {"summary": "概要", "key_points": "主な論点", "decisions": "決定事項",
           "action_items": "アクションアイテム", "open_questions": "未解決の課題", "task": "タスク",
           "owner": "担当者", "due": "期限", "date": "日付", "duration": "時間",
           "language": "会議の言語", "none": "なし", "participants": "参加者"},
    "zh": {"summary": "摘要", "key_points": "要点", "decisions": "决定事项",
           "action_items": "待办事项", "open_questions": "未解决的问题", "task": "任务",
           "owner": "负责人", "due": "截止日期", "date": "日期", "duration": "时长",
           "language": "会议语言", "none": "无", "participants": "参会人员"},
    "zh-Hant": {"summary": "摘要", "key_points": "要點", "decisions": "決定事項",
                "action_items": "待辦事項", "open_questions": "未解決的問題", "task": "任務",
                "owner": "負責人", "due": "截止日期", "date": "日期", "duration": "時長",
                "language": "會議語言", "none": "無", "participants": "與會人員"},
}


def ui_text(lang: str) -> dict:
    if lang == "zh" and config.load_settings().get("chinese_script") == "traditional":
        return UI_TEXT["zh-Hant"]
    return UI_TEXT[lang]


def html_lang(lang: str) -> str:
    if lang == "zh":
        return "zh-Hant" if config.load_settings().get("chinese_script") == "traditional" else "zh-Hans"
    return lang


EAST_ASIA_FONT = {"ja": "Yu Gothic UI", "zh": "Microsoft YaHei", "zh-Hant": "Microsoft JhengHei"}


def to_markdown(r: dict, lang: str) -> str:
    T = ui_text(lang)
    m = r.get("meta", {})
    out = [f"# {r['title']}", "",
           f"- {T['date']}: {m.get('date', '')}", f"- {T['duration']}: {m.get('duration', '')}",
           f"- {T['language']}: {m.get('language', '')}"]
    if m.get("participants"):
        out.append(f"- {T['participants']}: {m['participants']}")
    out += ["", f"## {T['summary']}", "", r["summary"], ""]
    for k in ("key_points", "decisions"):
        out += [f"## {T[k]}", ""] + ([f"- {x}" for x in r[k]] or [f"- {T['none']}"]) + [""]
    out += [f"## {T['action_items']}", ""]
    if r["action_items"]:
        out += [f"| {T['task']} | {T['owner']} | {T['due']} |", "|---|---|---|"]
        out += [f"| {a['task']} | {a['owner']} | {a['due']} |" for a in r["action_items"]]
    else:
        out.append(f"- {T['none']}")
    out += ["", f"## {T['open_questions']}", ""] + ([f"- {x}" for x in r["open_questions"]] or [f"- {T['none']}"])
    return "\n".join(out) + "\n"


def to_html(reports: dict, languages: list[str], title: str) -> str:
    e = html.escape
    tabs, panes = [], []
    for i, lang in enumerate(languages):
        r = reports[lang]
        T = ui_text(lang)
        m = r.get("meta", {})
        lst = lambda xs: "<ul>" + "".join(f"<li>{e(x)}</li>" for x in xs) + "</ul>" if xs else f"<p class='muted'>{T['none']}</p>"
        acts = ("<table><tr><th>" + "</th><th>".join([T['task'], T['owner'], T['due']]) + "</th></tr>" +
                "".join(f"<tr><td>{e(a['task'])}</td><td>{e(a['owner'])}</td><td>{e(a['due'])}</td></tr>"
                        for a in r["action_items"]) + "</table>") if r["action_items"] else f"<p class='muted'>{T['none']}</p>"
        tabs.append(f"<button class='tab{' on' if i == 0 else ''}' onclick=\"show('{lang}',this)\">{L[lang]}</button>")
        panes.append(f"""<section id='p-{lang}' lang='{html_lang(lang)}' style='display:{'block' if i == 0 else 'none'}'>
<h1>{e(r['title'])}</h1>
<p class='meta'>{T['date']}: {e(m.get('date', ''))} · {T['duration']}: {e(m.get('duration', ''))} · {T['language']}: {e(m.get('language', ''))}{(' · ' + T['participants'] + ': ' + e(m['participants'])) if m.get('participants') else ''}</p>
<h2>{T['summary']}</h2><p>{e(r['summary'])}</p>
<h2>{T['key_points']}</h2>{lst(r['key_points'])}
<h2>{T['decisions']}</h2>{lst(r['decisions'])}
<h2>{T['action_items']}</h2>{acts}
<h2>{T['open_questions']}</h2>{lst(r['open_questions'])}
</section>""")
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{e(title)}</title>
<style>
body{{font-family:'Segoe UI','Leelawadee UI','Yu Gothic UI','Microsoft YaHei',sans-serif;}}
[lang=zh-Hans]{{font-family:'Segoe UI','Microsoft YaHei',sans-serif}} [lang=zh-Hant]{{font-family:'Segoe UI','Microsoft JhengHei',sans-serif}} [lang=ja]{{font-family:'Segoe UI','Yu Gothic UI',sans-serif}}
body{{max-width:860px;margin:32px auto;padding:0 20px;color:#1d2433;line-height:1.6}}
.tab{{border:1px solid #cfd6e4;background:#fff;padding:6px 16px;border-radius:20px;cursor:pointer;margin-right:6px}}
.tab.on{{background:#2457d6;color:#fff;border-color:#2457d6}}
h1{{margin:20px 0 4px}} h2{{font-size:1.1rem;margin-top:26px;border-bottom:2px solid #e8ecf4;padding-bottom:4px}}
.meta,.muted{{color:#6b7488}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #dde3ee;padding:6px 10px;text-align:left}} th{{background:#f3f6fb}}
.foot{{margin-top:40px;color:#9aa3b5;font-size:.8rem}}
@media print{{.tab{{display:none}} section{{display:block!important;page-break-after:always}}}}
</style></head><body>
<div>{''.join(tabs)}</div>{''.join(panes)}
<p class='foot'>Generated offline by sMeetSec · simply handle IT</p>
<script>function show(l,b){{document.querySelectorAll('section').forEach(s=>s.style.display='none');document.getElementById('p-'+l).style.display='block';document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));b.classList.add('on')}}</script>
</body></html>"""


def to_docx(r: dict, lang: str, path) -> None:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    T = ui_text(lang)
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Leelawadee UI"
    st.font.size = Pt(11)
    rpr = st.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(fonts)
    fonts.set(qn("w:eastAsia"), EAST_ASIA_FONT.get(html_lang(lang).replace("zh-Hans", "zh"), "Yu Gothic UI"))
    fonts.set(qn("w:cs"), "Leelawadee UI")

    m = r.get("meta", {})
    doc.add_heading(r["title"], 0)
    doc.add_paragraph(f"{T['date']}: {m.get('date', '')}    {T['duration']}: {m.get('duration', '')}    "
                      f"{T['language']}: {m.get('language', '')}")
    if m.get("participants"):
        doc.add_paragraph(f"{T['participants']}: {m['participants']}")
    doc.add_heading(T["summary"], 1)
    doc.add_paragraph(r["summary"])
    for k in ("key_points", "decisions"):
        doc.add_heading(T[k], 1)
        for x in r[k] or [T["none"]]:
            doc.add_paragraph(x, style="List Bullet")
    doc.add_heading(T["action_items"], 1)
    if r["action_items"]:
        tb = doc.add_table(rows=1, cols=3)
        tb.style = "Light Grid Accent 1"
        for c, h in zip(tb.rows[0].cells, (T["task"], T["owner"], T["due"])):
            c.text = h
        for a in r["action_items"]:
            row = tb.add_row().cells
            row[0].text, row[1].text, row[2].text = a["task"], a["owner"], a["due"]
    else:
        doc.add_paragraph(T["none"])
    doc.add_heading(T["open_questions"], 1)
    for x in r["open_questions"] or [T["none"]]:
        doc.add_paragraph(x, style="List Bullet")
    doc.save(str(path))


def export_reports(mid: str, meta: dict, reports: dict, languages: list[str]) -> None:
    for lang in languages:
        r = reports[lang]
        meetings.path(mid, f"report_{lang}.md").write_text(to_markdown(r, lang), encoding="utf-8")
        to_docx(r, lang, meetings.path(mid, f"report_{lang}.docx"))
    meetings.path(mid, "report.html").write_text(to_html(reports, languages, meta["title"]), encoding="utf-8")


def export_transcript(mid: str, t: dict) -> None:
    """Write transcript.txt and transcript.srt next to the JSON."""
    lines = transcript_lines(t)
    meetings.path(mid, "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    srt = []
    for i, s in enumerate(t.get("segments", []), 1):
        spk = f"{s['speaker']}: " if s.get("speaker") else ""
        srt += [str(i), f"{fmt_time(s['start'], True)} --> {fmt_time(s['end'], True)}", spk + s["text"], ""]
    meetings.path(mid, "transcript.srt").write_text("\n".join(srt), encoding="utf-8")
