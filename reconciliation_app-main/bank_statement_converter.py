# pip install pdfplumber pandas openpyxl

from pathlib import Path
import re
from typing import Optional, Union, Iterable
import pandas as pd
import pdfplumber

def extract_bank_pdf(
    pdf_path: Union[str, Path],
    *,
    target_columns: Iterable[str] = (
        "RB",
        "Datum valute / Datum knjiž.",
        "Korisnik kartice",
        "Opis / Referenca",
        "Iznos transakcije",
        "Iznos zaduženja",
    ),
    min_cols: int = 6,
    header_keywords: Iterable[str] = (
        "datum", "valute", "knji", "korisnik", "kartice",
        "opis", "referenca", "iznos", "tranzakcije", "zaduzenja", "zaduženja"
    ),
    out_xlsx: Optional[Union[str, Path]] = None,
    out_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Extracts transaction tables from a bank PDF and returns a clean DataFrame with 6 fixed columns:
        RB | Datum valute / Datum knjiž. | Korisnik kartice | Opis / Referenca | Iznos transakcije | Iznos zaduženja

    Behavior:
    - Detects & skips a header row if present; otherwise keeps the first row (no data loss).
    - Takes exactly the first 6 columns from each detected table (pads rows if shorter).
    - Preserves raw amount strings in the two amount columns.
    - Trims whitespace and drops fully empty rows.
    - Optionally saves to Excel and/or CSV.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF statement.
    target_columns : iterable of str
        Column names to enforce (length must be 6).
    min_cols : int
        Minimum number of columns expected per table (rows shorter than this are padded with None).
    header_keywords : iterable of str
        Keywords to detect header rows (case/diacritic-insensitive).

    Returns
    -------
    pandas.DataFrame
        Cleaned statement rows with exactly the 6 requested columns.
    """
    target_columns = tuple(target_columns)
    assert len(target_columns) == 6, "target_columns must contain exactly 6 names."

    def _norm(s: str) -> str:
        s = ("" if s is None else str(s))
        s = re.sub(r"\s+", " ", s).strip().lower()
        # normalize diacritics roughly for matching
        s = (s.replace("ž", "z").replace("đ", "dj")
               .replace("č", "c").replace("ć", "c").replace("š", "s"))
        return s

    header_keywords = tuple(_norm(k) for k in header_keywords)

    def _looks_like_header(cells) -> bool:
        txt = " ".join(_norm(c) for c in cells if c is not None)
        hits = sum(1 for kw in header_keywords if kw in txt)
        return hits >= 2

    def _normalize_ws(x):
        if isinstance(x, str):
            return re.sub(r"\s+", " ", x).strip()
        return x

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path.resolve()}")

    collected = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for t in tables:
                if not t or len(t) < 1:
                    continue

                first_row = t[0]
                has_header = _looks_like_header(first_row)
                rows = t[1:] if has_header and len(t) > 1 else t[:]
                if not rows:
                    continue

                trimmed = []
                for r in rows:
                    if r is None:
                        continue
                    r = list(r)
                    if len(r) < min_cols:
                        r += [None] * (min_cols - len(r))
                    trimmed.append(r[:6])

                if not trimmed:
                    continue

                df = pd.DataFrame(trimmed, columns=target_columns)
                df = df.map(_normalize_ws)

                # drop fully empty rows
                nonempty = df.apply(
                    lambda row: any(
                        (isinstance(v, str) and v.strip() != "") or pd.notna(v)
                        for v in row
                    ),
                    axis=1,
                )
                df = df[nonempty]

                if not df.empty:
                    collected.append(df)

    if not collected:
        result = pd.DataFrame(columns=target_columns)
    else:
        result = pd.concat(collected, ignore_index=True)

    # enforce final column order & keep only the 6 wanted columns
    result = result.loc[:, target_columns].reset_index(drop=True)

    # optional exports
    if out_xlsx:
        Path(out_xlsx).parent.mkdir(parents=True, exist_ok=True)
        result.to_excel(out_xlsx, index=False)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_csv, index=False)

    return result


"""
pdf_path = Path("./EBanking sept_okt test.pdf")

df = extract_bank_pdf(
    pdf_path,
    out_xlsx="./EBanking_converted.xlsx",
    out_csv="./EBanking_converted.csv",
)

print(df.head(10))
"""