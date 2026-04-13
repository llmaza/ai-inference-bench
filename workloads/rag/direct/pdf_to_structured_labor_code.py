#!/usr/bin/env python3
"""
Извлечение Трудового кодекса из PDF в иерархический текст:
Документ → Часть → Раздел → Глава → Статья → Пункт (1., 2., … или 1), 2), …).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz

RE_RAZDEL = re.compile(r"^РАЗДЕЛ\s+\d+\.")
RE_GLAVA = re.compile(r"^Глава\s+\d+\.")
RE_STATYA = re.compile(r"^Статья\s+(\d+)\.\s*(.*)$")
RE_CHAST = re.compile(r"^(ОБЩАЯ ЧАСТЬ|ОСОБАЯ ЧАСТЬ)$")
RE_BODY_START = re.compile(r"^\s*\d+\.\s")
# Пункт статьи: «1. Текст» в начале строки (не «1)»).
RE_PUNKT_LINE = re.compile(r"^(\d{1,3})\.\s+(\S.*)$")
# Подпункт, если в статье нет пунктов с точкой — только «1) …».
RE_PODPUNKT_LINE = re.compile(r"^(\d{1,3})\)\s+(\S.*)$")


def merge_wrapped_title(title: str, body: str) -> tuple[str, str]:
    """Склеить перенос заголовка статьи (хвост на следующей строке PDF)."""
    if not body:
        return title, body
    first, sep, rest = body.partition("\n")
    frag = first.strip()
    if (
        frag
        and len(frag) < 90
        and not RE_BODY_START.match(frag)
        and frag[:1].islower()
    ):
        return f"{title} {frag}".strip(), (rest.lstrip("\n") if sep else "")
    return title, body


def _point_split_indices(
    lines: list[str], pattern: re.Pattern[str]
) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = pattern.match(line.strip())
        if m:
            hits.append((i, m.group(1)))
    return hits


def split_article_into_points(body: str) -> list[dict[str, str]]:
    """
    Разбить текст статьи на пункты.
    Сначала по строкам «N. …», иначе по «N) …», иначе один блок.
    """
    b = body.strip()
    if not b:
        return []

    lines = body.split("\n")
    punkt_idx = _point_split_indices(lines, RE_PUNKT_LINE)
    use_podpunkt = False
    if not punkt_idx:
        punkt_idx = _point_split_indices(lines, RE_PODPUNKT_LINE)
        use_podpunkt = bool(punkt_idx)

    if not punkt_idx:
        return [{"kind": "text", "label": "", "text": b}]

    kind_p = "podpunkt" if use_podpunkt else "punkt"
    out: list[dict[str, str]] = []
    first_i = punkt_idx[0][0]
    intro = "\n".join(lines[:first_i]).strip()
    if intro:
        out.append({"kind": "intro", "label": "", "text": intro})

    for j, (start, label) in enumerate(punkt_idx):
        end = punkt_idx[j + 1][0] if j + 1 < len(punkt_idx) else len(lines)
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            out.append({"kind": kind_p, "label": label, "text": chunk})
    return out


def extract_lines(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(doc[i].get_text() for i in range(doc.page_count)).split("\n")
    finally:
        doc.close()


def collect_header_lines(
    lines: list[str],
    start_i: int,
    stop_at: tuple[re.Pattern, ...],
) -> tuple[str, int]:
    """Собрать многострочный заголовок до первой строки, подходящей под stop_at."""
    parts: list[str] = [lines[start_i].strip()]
    i = start_i + 1
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        if not s:
            i += 1
            continue
        if any(p.match(s) for p in stop_at):
            break
        parts.append(s)
        i += 1
    return " ".join(parts), i


def parse_labor_code(lines: list[str], doc_title: str) -> list[dict]:
    chast: str | None = None
    razdel: str | None = None
    glava: str | None = None
    articles: list[dict] = []

    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue

        if RE_CHAST.match(s):
            chast = s
            i += 1
            continue

        if RE_RAZDEL.match(s):
            title, i = collect_header_lines(lines, i, (RE_GLAVA, RE_STATYA))
            razdel = title
            continue

        if RE_GLAVA.match(s):
            title, i = collect_header_lines(lines, i, (RE_STATYA, RE_RAZDEL))
            glava = title
            continue

        m = RE_STATYA.match(s)
        if m:
            block_lines = [lines[i]]
            i += 1
            while i < n:
                ts = lines[i].strip()
                if RE_STATYA.match(ts):
                    break
                if RE_RAZDEL.match(ts) or RE_GLAVA.match(ts) or RE_CHAST.match(ts):
                    break
                block_lines.append(lines[i])
                i += 1
            full = "\n".join(block_lines).strip()
            first_nl = full.find("\n")
            if first_nl == -1:
                head, body = full, ""
            else:
                head, body = full[:first_nl].strip(), full[first_nl + 1 :].strip()

            mm = RE_STATYA.match(head)
            num = mm.group(1) if mm else m.group(1)
            title_hint = (mm.group(2) if mm else m.group(2)).strip()
            title_hint, body = merge_wrapped_title(title_hint, body)

            articles.append(
                {
                    "document": doc_title,
                    "part": chast or "",
                    "section": razdel or "",
                    "chapter": glava or "",
                    "article_number": int(num),
                    "article_title": title_hint,
                    "text": body,
                    "points": split_article_into_points(body),
                    "full_heading_line": head,
                }
            )
            continue

        i += 1

    return articles


def write_markdown(articles: list[dict], path: Path) -> None:
    lines: list[str] = []
    doc = articles[0]["document"] if articles else "Документ"
    lines.append(f"# {doc}")
    lines.append("")

    prev: tuple[str, str, str] = ("", "", "")
    for a in articles:
        key = (a["part"], a["section"], a["chapter"])
        if key != prev:
            if a["part"] and (not prev[0] or a["part"] != prev[0]):
                lines.append(f"## {a['part']}")
                lines.append("")
            if a["section"] and a["section"] != prev[1]:
                lines.append(f"### {a['section']}")
                lines.append("")
            if a["chapter"] and a["chapter"] != prev[2]:
                lines.append(f"#### {a['chapter']}")
                lines.append("")
            prev = key

        lines.append(f"##### Статья {a['article_number']}. {a['article_title']}")
        lines.append("")
        for p in a.get("points") or []:
            pk = p["kind"]
            if pk == "intro":
                lines.append("###### Вводная часть")
                lines.append("")
                lines.append(p["text"])
                lines.append("")
            elif pk == "punkt":
                lines.append(f"###### Пункт {p['label']}.")
                lines.append("")
                lines.append(p["text"])
                lines.append("")
            elif pk == "podpunkt":
                lines.append(f"###### Подпункт {p['label']})")
                lines.append("")
                lines.append(p["text"])
                lines.append("")
            else:
                lines.append(p["text"])
                lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_jsonl(articles: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for a in articles:
            row = {k: v for k, v in a.items() if k != "full_heading_line"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_pretty(articles: list[dict], path: Path) -> None:
    """Один массив JSON с отступами — удобно смотреть в редакторе (поле points видно сразу)."""
    out = [{k: v for k, v in a.items() if k != "full_heading_line"} for a in articles]
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent
        / "assets"
        / "Трудовой Кодекс Актуальный.pdf",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "assets",
    )
    args = ap.parse_args()

    lines = extract_lines(args.pdf)
    doc_title = "Трудовой кодекс Республики Казахстан"
    for line in lines[:5]:
        t = line.strip()
        if t and not t.startswith("Кодекс") and "кодекс" in t.lower():
            doc_title = t
            break
        if t.startswith("Трудовой кодекс"):
            doc_title = t
            break

    articles = parse_labor_code(lines, doc_title)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = (args.out_dir / "Трудовой_Кодекс_структурированный.md").resolve()
    jl_path = (args.out_dir / "Трудовой_Кодекс_структурированный.jsonl").resolve()
    json_path = (args.out_dir / "Трудовой_Кодекс_структурированный.json").resolve()

    write_markdown(articles, md_path)
    write_jsonl(articles, jl_path)
    write_json_pretty(articles, json_path)

    nums = sorted(a["article_number"] for a in articles)
    print(f"Статей: {len(articles)} (номера {nums[0]}…{nums[-1]})")
    print(f"Markdown: {md_path}")
    print(f"JSONL:    {jl_path}")
    print(f"JSON:     {json_path}")
    print(
        "(В .jsonl каждая статья — одна длинная строка: поле «points» в конце. "
        "Откройте .json для просмотра с отступами.)"
    )


if __name__ == "__main__":
    main()
