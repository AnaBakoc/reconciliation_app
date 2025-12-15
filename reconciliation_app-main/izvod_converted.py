# parser_raiffeisen.py
from __future__ import annotations

import re, json
from pathlib import Path
from collections import OrderedDict
from typing import Optional

# ------------------- Patterns & helpers -------------------

HEADER_KILLWORDS = {
    "raiffeisen banka", "đorđa stanojevića", "telefon", "fax",
    "http://www.raiffeisenbank.rs", "broj izvoda", "strana"
}

DATE_RE = r"\d{2}\.\d{2}\.\d{2}"
MONEY_GENERIC = r"[+-]?\s?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d{2})"

UK_POT_RE = re.compile(
    rf"Ukupna\s+potro[šs]nja[:\s]*({MONEY_GENERIC})\s*RSD",
    re.I
)

# One whole-transaction block regardless of line wraps
BLOCK_RE = re.compile(
    rf"""
    (?P<dt1>{DATE_RE})?                                   # datum_transakcije (optional)
    \s*
    (?P<dt2>{DATE_RE})                                    # datum_valute
    \s+
    (?P<card>\d{{3,6}})                                   # br. kartice
    \s+
    (?P<merchant>.+?)                                     # greedy merchant until next amount
    \s+
    (?P<orig>{MONEY_GENERIC}\s+[A-Z]{{3}})                # iznos u orig. valuti
    (?:\s+(?P<ref>{MONEY_GENERIC}\s+[A-Z]{{3}}))?         # iznos u ref. valuti (optional)
    \s+
    (?P<rsd>{MONEY_GENERIC})                              # iznos RSD
    \s+
    (?P<refno>\d{{6,}}(?:\.\d+)?)                         # Ref. broj
    """,
    re.X | re.I
)

NAME_CAND_RE = re.compile(
    r"\b([A-ZČĆŽŠĐ][A-Za-zČĆŽŠĐčćžšđ.'\-]+(?:\s+[A-ZČĆŽŠĐ][A-Za-zČĆŽŠĐčćžšđ.'\-]+){1,3})\b"
)

def _normalize_number(s: str) -> float | None:
    """Accept '1.234,56' / '1,234.56' / '1234.56', keep sign."""
    s = s.strip().replace(" ", "")
    if "," in s and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def _load_pages_text(path: Path) -> list[str]:
    """PDF -> list of page texts; image -> single OCR page."""
    if path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
        except Exception as e:
            raise RuntimeError("pip install pdfplumber") from e
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                txt = p.extract_text() or ""
                if not txt.strip():
                    # OCR fallback (best-effort)
                    try:
                        import pytesseract
                        im = p.to_image(resolution=300).original
                        txt = pytesseract.image_to_string(im, lang="srp+eng")
                    except Exception:
                        pass
                pages.append(txt.strip())
        return pages
    else:
        # image file → OCR
        try:
            from PIL import Image
            import pytesseract
        except Exception as e:
            raise RuntimeError("pip install pillow pytesseract; install tesseract-ocr") from e
        return [pytesseract.image_to_string(Image.open(path), lang="srp+eng").strip()]

def _strip_header_noise(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not any(k in ln.lower() for k in HEADER_KILLWORDS)]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()

def _find_ukupna(collapsed: str) -> float | None:
    m = UK_POT_RE.search(collapsed)
    return _normalize_number(m.group(1)) if m else None

def _find_name(collapsed: str) -> str:
    """
    Try to find a name-like 2–4 token sequence near 'Ukupna potrošnja' first,
    else take the first plausible candidate on the page.
    """
    up = re.search(r"Ukupna\s+potro[šs]nja", collapsed, re.I)
    if up:
        L = max(0, up.start() - 150)
        R = min(len(collapsed), up.end() + 150)
        win = collapsed[L:R]
        m = NAME_CAND_RE.search(win)
        if m:
            name = m.group(1).strip()
            name = re.sub(r"\bUkupna(?:\s+potro[šs]nja)?\b", "", name, flags=re.I).strip()
            return name

    m2 = NAME_CAND_RE.search(collapsed)
    if m2:
        name = m2.group(1).strip()
        name = re.sub(r"\bUkupna(?:\s+potro[šs]nja)?\b", "", name, flags=re.I).strip()
        return name
    return ""

def _parse_transactions_blob(collapsed: str):
    txs = []
    for m in BLOCK_RE.finditer(collapsed):
        dt1 = m.group("dt1") or ""
        dt2 = m.group("dt2")
        card = m.group("card")
        merchant = " ".join(m.group("merchant").split())
        orig = " ".join(m.group("orig").split())
        refv = m.group("ref")
        refv = " ".join(refv.split()) if refv else None
        rsd_raw = m.group("rsd")
        rsd = _normalize_number(rsd_raw)
        refno = m.group("refno")

        txs.append({
            "datum_transakcije": dt1,
            "datum_valute": dt2,
            "br_kartice": card,
            "mesto_i_naziv_trgovca": merchant,
            "iznos_orig_valuta": orig,
            "iznos_ref_valuta": refv,    # null if missing
            "iznos_rsd": rsd,
            "ref_broj": refno
        })
    return txs

def _parse_page(raw_text: str):
    collapsed = _strip_header_noise(raw_text)
    name = _find_name(collapsed)       # may be ""
    ukupna = _find_ukupna(collapsed)   # may be None
    txs = _parse_transactions_blob(collapsed)
    return name, ukupna, txs

# ------------------- Public API -------------------

def parse_bank_statement(
    input_path: str | Path,
    save_path: Optional[str | Path] = None,
    *,
    skip_first_page: bool = True,
    debug_dump_dir: Optional[str | Path] = None,
) -> list[dict]:
    """
    Parse a Bank RS statement (multi-person, multi-page) into a list of people dicts.

    Args:
        input_path: PDF (applymapeferred) or an image file path.
        save_path: If provided, the resulting JSON is saved here. If None, no file is written.
        skip_first_page: Some statements have a cover/summary page; set False to include it.
        debug_dump_dir: If provided, dumps raw extracted text per page into this directory.

    Returns:
        A list[dict] with entries like:
        {
          "name_surname": str,
          "ukupna_potrosnja_rsd": float | None,
          "transactions": [ ... ]
        }
    """
    input_path = Path(input_path)
    pages = _load_pages_text(input_path)

    # Select pages
    if skip_first_page and pages:
        pages_to_parse = list(enumerate(pages[1:], start=2))  # (page_number, text), 1-based
    else:
        pages_to_parse = list(enumerate(pages, start=1))

    # Optional debug dumps
    if debug_dump_dir:
        dbg_dir = Path(debug_dump_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)

    people: "OrderedDict[str, dict]" = OrderedDict()
    current_person_key: str | None = None
    unknown_counter = 1

    for page_no, page_text in pages_to_parse:
        if debug_dump_dir:
            dbg = Path(debug_dump_dir) / f"{input_path.stem}.debug_page_{page_no:03d}.txt"
            dbg.write_text(page_text, encoding="utf-8")

        name, ukupna, txs = _parse_page(page_text)

        # Decide which person this page belongs to
        if name:
            # If we've seen this name before, continue appending
            if name in people:
                current_person_key = name
            else:
                # New person starts here
                people[name] = {
                    "name_surname": name,
                    "ukupna_potrosnja_rsd": None,
                    "transactions": []
                }
                current_person_key = name
        else:
            # Continuation page (no name on the page)
            if current_person_key is None:
                # Start an UNKNOWN record that we keep separate
                placeholder = f"UNKNOWN_{unknown_counter}"
                unknown_counter += 1
                people[placeholder] = {
                    "name_surname": "",
                    "ukupna_potrosnja_rsd": None,
                    "transactions": []
                }
                current_person_key = placeholder

        # Merge total: keep the last non-None encountered
        if ukupna is not None:
            people[current_person_key]["ukupna_potrosnja_rsd"] = ukupna

        # Append transactions
        if txs:
            people[current_person_key]["transactions"].extend(txs)

    # Final polish: drop entirely empty UNKNOWNs
    people = OrderedDict(
        (k, v) for k, v in people.items()
        if v["transactions"] or v["ukupna_potrosnja_rsd"] is not None or v["name_surname"]
    )

    out_list = list(people.values())

    # Optional save
    if save_path is not None:
        save_path = Path(save_path)
        save_path.write_text(json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_list
