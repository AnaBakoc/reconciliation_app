# run_reconciliation.py
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
import pandas as pd

from izvod_converted import parse_bank_statement
from bank_statement_converter import extract_bank_pdf
from reconcile_expenses import match_expenses


def _ensure_path_list(x: str | Path | Sequence[str | Path]) -> list[Path]:
    """
    Helper: accept either a single path or a sequence of paths,
    and always return a list[Path].
    """
    if isinstance(x, (str, Path)):
        return [Path(x)]
    # assume it's already a sequence
    return [Path(p) for p in x]


def reconcile_and_save(
    *,
    # now accepts either a single path OR a list/tuple of paths
    bank_pdf_path: str | Path | Sequence[str | Path],
    ebanking_pdf_path: str | Path | Sequence[str | Path],
    caaps_path: str | Path,
    expense_path: str | Path,
    out_path: str | Path = "A_TRIAL1.xlsx",
    skip_first_page: bool = True,
    debug_dump_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Full pipeline:
      1) Parse one or more bank card statement PDFs to a combined JSON-like structure.
      2) Extract one or more eBanking PDFs into a single concatenated DataFrame.
      3) Run matching (CAAPS + Expense Detail) and save to Excel.
      4) Return the final DataFrame.
    """
    bank_pdf_paths = _ensure_path_list(bank_pdf_path)
    ebanking_pdf_paths = _ensure_path_list(ebanking_pdf_path)
    caaps_path = Path(caaps_path)
    expense_path = Path(expense_path)
    out_path = Path(out_path)

    # 1) IBM Bank statements -> combined JSON-like structure
    all_izvod = []
    for p in bank_pdf_paths:
        izvod_part = parse_bank_statement(
            p,
            save_path=None,
            skip_first_page=skip_first_page,
            debug_dump_dir=debug_dump_dir,
        )
        # If your parser returns a list-of-persons, extend. If it returns a dict, append.
        if isinstance(izvod_part, list):
            all_izvod.extend(izvod_part)
        else:
            all_izvod.append(izvod_part)

    IBM_izvod_json = all_izvod

    # 2) eBanking statements -> concatenated DataFrame
    ebanking_dfs = []
    for p in ebanking_pdf_paths:
        df_part = extract_bank_pdf(p)
        if df_part is not None and not getattr(df_part, "empty", True):
            ebanking_dfs.append(df_part)

    if ebanking_dfs:
        ebanking_izvod = pd.concat(ebanking_dfs, ignore_index=True)
    else:
        # fallback to empty df if nothing parsed
        ebanking_izvod = pd.DataFrame()

    # 3) Match + classify expenses; save to Excel
    final_df = match_expenses(
        eBanking_df=ebanking_izvod,
        IBM_izvod_json=IBM_izvod_json,
        caaps_path=caaps_path,
        expense_path=expense_path,
        save_to=out_path,  # writes the file
    )

    # 4) Ensure file exists even if match_expenses saving is changed later
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_excel(out_path, index=False)

    return final_df


if __name__ == "__main__":
    out_path = "A_TRIAL2.xlsx"
    df = reconcile_and_save(
        bank_pdf_path=[
            "/Users/anabakoc/Desktop/Danica/data/IBM-INTERNATIONAL BUSINESS_Izvod Oktobar.pdf",
            # add more PDFs here if you want:
            # "/Users/anabakoc/Desktop/Danica/data/IBM-INTERNATIONAL BUSINESS_Izvod Novembar.pdf",
        ],
        ebanking_pdf_path=[
            "/Users/anabakoc/Desktop/Danica/data/Oktobar_Novembar mbanking.pdf",
            # "/Users/anabakoc/Desktop/Danica/data/Decembar mbanking.pdf",
        ],
        caaps_path="/Users/anabakoc/Desktop/Danica/data/CAAPS 14 novembar.xlsx",
        expense_path="/Users/anabakoc/Desktop/Danica/data/RP0004 - Expense Detail Report 17. novembar.xlsx",
        out_path=out_path,
        skip_first_page=True,
        debug_dump_dir=None,
    )
    print(f"\n✅ Saved: {out_path}")
