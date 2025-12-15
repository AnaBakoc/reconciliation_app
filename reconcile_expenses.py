# expense_matcher.py
from __future__ import annotations

import os
import re
import json
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, Sequence

import pandas as pd
from special_grouping import process_hotel_stays

# =========================
# Helpers 
# =========================
def parse_amount_to_cents(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    # Grab ONLY the first numeric block (with optional minus)
    # Examples:
    #   "-20.66 EUR"          -> "-20.66"
    #   "2,86 978/EUR"        -> "2,86"
    #   "-3,520.00 RSD"       -> "-3,520.00"
    m = re.search(r'-?\d[\d\.,]*', s)
    if not m:
        return None

    num = m.group(0)

    # Decide decimal vs thousand separators
    if ',' in num and '.' in num:
        last_comma = num.rfind(',')
        last_dot = num.rfind('.')
        if last_dot > last_comma:
            # pattern like "-3,520.00" -> decimal is '.', comma is thousands
            num = num.replace(',', '')
        else:
            # pattern like "1.234,56" -> decimal is ',', dot is thousands
            num = num.replace('.', '').replace(',', '.')
    elif ',' in num:
        # Only comma present -> treat comma as decimal
        num = num.replace(',', '.')

    try:
        v = float(num)
        return int(round(v * 100))
    except Exception:
        return None

def canonical_name(name: str) -> str:
    if name is None:
        return ''

    s = str(name)

    # --- Serbian-specific digraph/ligature normalization ---
    # đ/Đ -> dj/DJ
    s = re.sub(r'đ', 'dj', s)
    s = re.sub(r'Đ', 'DJ', s)

    # --- Strip accents/diacritics after the above mappings ---
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))

    # Keep only letters/spaces, uppercase, dedupe short junk
    s = re.sub(r'[^A-Za-z\s]', ' ', s)
    tokens = [t for t in s.upper().split() if len(t) > 1]

    # Order-insensitive canonical form
    return ' '.join(sorted(set(tokens)))


def find_column(columns, exact_lower=None, must_contain_all=None):
    cols = list(columns)
    if exact_lower:
        for c in cols:
            if c.strip().lower() == exact_lower:
                return c
    if must_contain_all:
        for c in cols:
            low = c.lower()
            if all(sub in low for sub in must_contain_all):
                return c
    return None

# ---- Date parsing ----
EXCEL_ORIGIN = datetime(1899, 12, 30)  # Excel's 1900 system

def to_iso(d: datetime) -> str:
    return d.date().isoformat()

def parse_excel_serial(n):
    # Excel serial date to datetime (handles floats)
    try:
        return EXCEL_ORIGIN + timedelta(days=float(n))
    except Exception:
        return None

def parse_any_date(value):
    """
    Returns YYYY-MM-DD or None.
    Handles pandas Timestamps, Excel serials, and many string formats.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    if isinstance(value, (pd.Timestamp, datetime)):
        return to_iso(pd.to_datetime(value).to_pydatetime())

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        d = parse_excel_serial(value)
        if d:
            return to_iso(d)

    s = str(value).strip()
    if not s:
        return None

    s_norm = s.replace('\n', ' ').replace('\t', ' ').replace('/', ' ').replace('|', ' ')
    s_norm = re.sub(r'\s+', ' ', s_norm)

    for fmt in ['%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y']:
        try:
            return datetime.strptime(s_norm, fmt).date().isoformat()
        except Exception:
            pass

    for fmt in ['%d.%m.%Y', '%d.%m.%y', '%Y-%m-%d', '%d-%m-%Y', '%d-%m-%y', '%m/%d/%Y', '%m/%d/%y']:
        try:
            return datetime.strptime(s_norm, fmt).date().isoformat()
        except Exception:
            pass

    token = None
    m = re.search(r'\b\d{2}\.\d{2}\.\d{2,4}\b', s_norm)
    if m:
        token = m.group(0)
        for fmt in ['%d.%m.%Y', '%d.%m.%y']:
            try:
                d = datetime.strptime(token, fmt).date()
                if d.year < 1970:
                    d = d.replace(year=d.year + 2000)
                return d.isoformat()
            except Exception:
                pass

    try:
        return pd.to_datetime(s_norm, errors='coerce', dayfirst=True).date().isoformat()
    except Exception:
        return None

def parse_ebanking_dual_date(value):
    """
    Takes a field like '01.09.2025 29.08.2025' and returns the SECOND date as YYYY-MM-DD.
    Accepts newlines or slashes between dates. If only one date is present, returns that one.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).replace('\n', ' ').replace('/', ' ').strip()
    tokens = re.findall(r'\b\d{2}\.\d{2}\.\d{2,4}\b', s)
    if not tokens:
        return parse_any_date(value)
    chosen = tokens[-1] if len(tokens) > 1 else tokens[0]
    for fmt in ['%d.%m.%Y', '%d.%m.%y']:
        try:
            d = datetime.strptime(chosen, fmt).date()
            if d.year < 1970:
                d = d.replace(year=d.year + 2000)
            return d.isoformat()
        except Exception:
            continue
    return None

def amounts_match(a, b, tolerance_ratio=0.000):
    """
    Returns True if |a - b| <= tolerance_ratio * max(|a|, |b|).
    a and b are amounts in cents (can be int, float, or None).
    """
    # Handle None / NaN / non-numeric
    try:
        if a is None or b is None:
            return False
        a_int = int(a)
        b_int = int(b)
    except (TypeError, ValueError):
        return False

    diff = abs(a_int - b_int)
    base = max(abs(a_int), abs(b_int))
    if base == 0:
        return diff == 0
    allowed = tolerance_ratio * base
    return diff <= allowed
def match_expenses(
    eBanking_df,   # DataFrame from extract_bank_pdf(...)
    IBM_izvod_json,    # list/dict from parse_bank_statement(...)
    caaps_path,
    expense_path,
    *,
    save_to=None
):
    import pandas as pd
    from pathlib import Path
    from itertools import combinations

    # ---------- Helpers specific to this adapter ----------
    def _triplets_from_izvod_json(json_like):
        persons = json_like if isinstance(json_like, list) else [json_like]
        triplets, pairs = set(), set()
        for p in persons:
            name_canon = canonical_name(p.get('name_surname', ''))
            for t in (p.get('transactions') or []):
                raw_amt = (
                    t.get('Iznos u orig. valuti') or
                    t.get('iznos_orig_valuta') or
                    t.get('iznos_rsd')
                )
                cents = parse_amount_to_cents(raw_amt)
                if cents is None:
                    continue
                a = abs(cents)
                pairs.add((name_canon, a))
                d = parse_any_date(t.get('datum_transakcije'))
                if d:
                    triplets.add((name_canon, a, d))
        return triplets, pairs

    def _triplets_from_bank_df(df):
        if df is None or getattr(df, "empty", True):
            return set(), set()

        df = df.copy()
        df.columns = df.columns.astype(str).str.replace('\n', ' ').str.strip()

        name_col = (find_column(df.columns, exact_lower='korisnik kartice') or
                    find_column(df.columns, must_contain_all=['korisnik']) or
                    find_column(df.columns, must_contain_all=['kartic']))
        amt_col = (
            find_column(df.columns, exact_lower='iznos transakcije') or
            find_column(df.columns, must_contain_all=['transakc']) or
            find_column(df.columns, exact_lower='iznos zaduženja') or
            find_column(df.columns, must_contain_all=['zadu'])
        )

        date_col = (find_column(df.columns, exact_lower='datum valute / datum knjiž.') or
                    find_column(df.columns, must_contain_all=['datum', 'valute']) or
                    find_column(df.columns, must_contain_all=['datum', 'knji']))

        if not all([name_col, amt_col, date_col]):
            raise ValueError("Bank statement DataFrame must contain cardholder, debit amount, and value/posting date columns.")

        df['name_canon'] = df[name_col].apply(canonical_name)
        df['amount_cents_abs'] = df[amt_col].apply(parse_amount_to_cents).abs()
        df['date_iso'] = df[date_col].apply(parse_ebanking_dual_date)

        triplets = set(
            df.dropna(subset=['amount_cents_abs', 'date_iso'])[['name_canon', 'amount_cents_abs', 'date_iso']]
              .itertuples(index=False, name=None)
        )
        pairs = set(
            df.dropna(subset=['amount_cents_abs'])[['name_canon', 'amount_cents_abs']]
              .itertuples(index=False, name=None)
        )
        return triplets, pairs

    # ---------- 1) CAAPS ----------
    caaps_path = Path(caaps_path)
    df_raw = pd.read_excel(caaps_path, header=None)
    section2_row = df_raw[df_raw.iloc[:, 0].astype(str).str.contains("SECTION 2", case=False, na=False)].index[0]
    header_row = section2_row + 1

    caaps_df = pd.read_excel(caaps_path, header=header_row)
    caaps_df.columns = caaps_df.columns.str.replace('\n', ' ').str.strip()

    report_key_col = find_column(caaps_df.columns, exact_lower='report key') or \
                     find_column(caaps_df.columns, must_contain_all=['report', 'key'])
    pay_amt_col = find_column(caaps_df.columns, exact_lower='payment amt') or \
                  find_column(caaps_df.columns, must_contain_all=['payment', 'amt'])
    if not report_key_col or not pay_amt_col:
        raise ValueError("Missing 'REPORT KEY' or 'PAYMENT AMT' in CAAPS.")

    ctr_pattern = r"^CTR\d+$"
    caaps_valid = caaps_df[caaps_df[report_key_col].astype(str).str.match(ctr_pattern, na=False)]
    caaps_result = caaps_valid[[report_key_col, pay_amt_col]].copy()
    caaps_result.rename(columns={report_key_col: 'REPORT KEY',
                                pay_amt_col: 'PAYMENT AMT'}, inplace=True)

    # Preserve RPTKEY order from CAAPS
    report_key_series = caaps_result['REPORT KEY'].astype(str).str.strip()
    report_key_order = list(dict.fromkeys(report_key_series))
    report_keys = set(report_key_order)

    # ---------- 2) Expense Detail ----------
    expense_raw = pd.read_excel(expense_path, header=None)
    row_lengths = expense_raw.notna().sum(axis=1)
    header_row_expense = row_lengths[row_lengths > 10].index[0]

    expense_df = pd.read_excel(expense_path, header=header_row_expense)
    expense_df.columns = expense_df.columns.str.replace('\n', ' ').str.strip()

    rptkey_col = find_column(expense_df.columns, exact_lower='rptkey') or \
                 find_column(expense_df.columns, must_contain_all=['rpt', 'key'])
    emp_name_col = find_column(expense_df.columns, exact_lower='emp name') or \
                   find_column(expense_df.columns, must_contain_all=['name'])

    exp_type_col = find_column(expense_df.columns, exact_lower='expense type') or \
                   find_column(expense_df.columns, must_contain_all=['expense', 'type'])

    exp_amt_txn_col = find_column(expense_df.columns, exact_lower='expense amt (txn currency)') or \
                      find_column(expense_df.columns, must_contain_all=['expense', 'amt', 'txn'])

    exp_amt_reimb_col = find_column(expense_df.columns, exact_lower='expense amt (reimbursement currency)') or \
                        find_column(expense_df.columns, must_contain_all=['expense', 'amt', 'reimbursement'])

    txn_date_col = (find_column(expense_df.columns, exact_lower='txn date') or
                    find_column(expense_df.columns, must_contain_all=['txn', 'date']) or
                    find_column(expense_df.columns, must_contain_all=['trans', 'date']))

    txn_curr_col = (
        find_column(expense_df.columns, exact_lower='txn currency') or
        find_column(expense_df.columns, must_contain_all=['txn', 'curr'])
    )

    if not all([rptkey_col, emp_name_col, exp_type_col, exp_amt_txn_col, exp_amt_reimb_col, txn_date_col]):
        raise ValueError(
            "Missing one of RPTKEY / Emp Name / Expense Type / "
            "Expense Amt (txn currency) / Expense Amt (reimbursement currency) / Txn Date in Expense Detail."
        )

    expense_df[rptkey_col] = expense_df[rptkey_col].astype(str).str.strip()

    base_cols = [
        rptkey_col, emp_name_col, exp_type_col,
        exp_amt_txn_col, exp_amt_reimb_col, txn_date_col
    ]
    if txn_curr_col:
        base_cols.append(txn_curr_col)

    matched = expense_df[
        expense_df[rptkey_col].isin(report_keys)
    ][base_cols].copy()

    rename_map = {
        rptkey_col: 'RPTKEY',
        emp_name_col: 'Emp Name',
        exp_type_col: 'Expense Type',
        exp_amt_txn_col: 'Expense Amt (txn currency)',
        exp_amt_reimb_col: 'Expense Amt (RSD)',
        txn_date_col: 'Txn Date',
    }
    if txn_curr_col:
        rename_map[txn_curr_col] = 'Txn Currency'

    matched.rename(columns=rename_map, inplace=True)
    # ensure RPTKEY follows CAAPS order
    matched['RPTKEY'] = matched['RPTKEY'].astype(str).str.strip()
    matched['RPTKEY'] = pd.Categorical(
        matched['RPTKEY'],
        categories=report_key_order,
        ordered=True
    )
    matched = matched.sort_values('RPTKEY', kind='stable').reset_index(drop=True)

    # ---- internal helper columns for matching ----
    matched['name_canon'] = matched['Emp Name'].apply(canonical_name)
    matched['amount_cents_abs'] = matched['Expense Amt (txn currency)'].apply(parse_amount_to_cents).abs()
    matched['Transaction Date'] = matched['Txn Date'].apply(parse_any_date)

    # ---------- 2b) Identify hotel-related lines ----------
    HOTEL_BASE_TYPES = {
        'HOTEL',
        'HOTEL TAX',
        'HOTEL INVOICE BREAKFAST',
    }
    PARKING_TYPE = 'PARKING'

    matched['Expense Type Canon'] = (
        matched['Expense Type']
        .fillna('')
        .str.upper()
        .str.strip()
    )

    # Mark hotel-related rows
    matched['is_hotel_base'] = matched['Expense Type Canon'].isin(HOTEL_BASE_TYPES)
    matched['is_parking'] = matched['Expense Type Canon'] == PARKING_TYPE
    matched['hotel_stay_id'] = pd.NA

    # ---------- 3) Build lookup sets ----------
    json_triplet_set, json_pair_set = _triplets_from_izvod_json(IBM_izvod_json)
    bank_triplet_set, bank_pair_set = _triplets_from_bank_df(eBanking_df)

    # ---------- 4) Classification ----------
    def classify_row(row):
        # For hotel items, skip individual classification - will be handled in hotel matching
        if row.get('is_hotel_base', False) or row.get('is_parking', False):
            return pd.Series({
                'Payment Method': 'cash',
                'Matched In': 'none',
                'Match Level': 'none',
                'Human Error Suspected': False,
            })

        amount_for_match = row['amount_cents_abs']

        if pd.isna(amount_for_match):
            return pd.Series({
                'Payment Method': 'cash',
                'Matched In': 'none',
                'Match Level': 'none',
                'Human Error Suspected': False,
            })

        where = []
        level = 'none'
        human_error = False

        # ----- Pass 1: name + date + amount -----
        hit_json_date = False
        hit_bank_date = False

        if row['Transaction Date'] is not None:
            # JSON triplets
            for (nm, amt, dt) in json_triplet_set:
                if nm == row['name_canon'] and dt == row['Transaction Date']:
                    if amounts_match(amt, amount_for_match):
                        hit_json_date = True
                        if int(amt) != int(amount_for_match):
                            human_error = True
                        break

            # eBanking triplets
            for (nm, amt, dt) in bank_triplet_set:
                if nm == row['name_canon'] and dt == row['Transaction Date']:
                    if amounts_match(amt, amount_for_match):
                        hit_bank_date = True
                        if int(amt) != int(amount_for_match):
                            human_error = True
                        break

        if hit_json_date or hit_bank_date:
            if hit_json_date:
                where.append('IBM-izvod')
            if hit_bank_date:
                where.append('eBanking')
            level = 'date+amount'
            return pd.Series({
                'Payment Method': 'company card',
                'Matched In': ','.join(where),
                'Match Level': level,
                'Human Error Suspected': human_error,
            })

        # ----- Pass 2: name + amount only -----
        hit_json_amt = False
        hit_bank_amt = False

        for (nm, amt) in json_pair_set:
            if nm == row['name_canon'] and amounts_match(amt, amount_for_match):
                hit_json_amt = True
                if int(amt) != int(amount_for_match):
                    human_error = True
                break

        for (nm, amt) in bank_pair_set:
            if nm == row['name_canon'] and amounts_match(amt, amount_for_match):
                hit_bank_amt = True
                if int(amt) != int(amount_for_match):
                    human_error = True
                break

        if hit_json_amt or hit_bank_amt:
            if hit_json_amt:
                where.append('IBM-izvod')
            if hit_bank_amt:
                where.append('eBanking')
            level = 'amount-only'
            return pd.Series({
                'Payment Method': 'company card',
                'Matched In': ','.join(where),
                'Match Level': level,
                'Human Error Suspected': human_error,
            })

        # ----- No match -----
        return pd.Series({
            'Payment Method': 'cash',
            'Matched In': 'none',
            'Match Level': level,
            'Human Error Suspected': False,
        })

    classified = matched.apply(classify_row, axis=1)

    # ---------- 4b) Special rule: Airfare + Airline Fee grouping ----------
    df_all = matched.join(classified)
    if 'Human Error Suspected' not in df_all.columns:
        df_all['Human Error Suspected'] = False

    df_all["AirfareFeeGroup"] = False
    df_all["HotelParkingGroup"] = False

    used_fee_idx = set()

    for idx, row in df_all.iterrows():
        if row['Payment Method'] != 'cash':
            continue
        if row['Transaction Date'] is None:
            continue

        if 'AIRFARE' not in row['Expense Type Canon']:
            continue

        airline_fee_part = (
            df_all['Expense Type Canon'].str.contains('AIRLINE', na=False) &
            df_all['Expense Type Canon'].str.contains('FEE', na=False)
        )

        airport_fee_part = (
            df_all['Expense Type Canon'].str.contains('AIRPORT', na=False) &
            (
                df_all['Expense Type Canon'].str.contains('FEE', na=False) |
                df_all['Expense Type Canon'].str.contains('FEES', na=False) |
                df_all['Expense Type Canon'].str.contains('TAX', na=False) |
                df_all['Expense Type Canon'].str.contains('TAXES', na=False)
            )
        )

        mask = (
            (df_all['Payment Method'] == 'cash') &
            (df_all.index != idx) &
            (~df_all.index.isin(used_fee_idx)) &
            (df_all['RPTKEY'] == row['RPTKEY']) &
            (df_all['Emp Name'] == row['Emp Name']) &
            (df_all['Transaction Date'] == row['Transaction Date']) &
            (airline_fee_part | airport_fee_part)
        )

        fee_candidates = df_all[mask]
        if fee_candidates.empty:
            continue

        fee_idx = fee_candidates.index[0]
        fee_row = df_all.loc[fee_idx]

        sum_amt = row['amount_cents_abs'] + fee_row['amount_cents_abs']

        where = []
        hit_json_date = False
        hit_bank_date = False
        human_error_airfare = False

        if row['Transaction Date'] is not None:
            for (nm, amt, dt) in json_triplet_set:
                if nm == row['name_canon'] and dt == row['Transaction Date']:
                    if amounts_match(amt, sum_amt):
                        hit_json_date = True
                        if int(amt) != int(sum_amt):
                            human_error_airfare = True
                        break

            for (nm, amt, dt) in bank_triplet_set:
                if nm == row['name_canon'] and dt == row['Transaction Date']:
                    if amounts_match(amt, sum_amt):
                        hit_bank_date = True
                        if int(amt) != int(sum_amt):
                            human_error_airfare = True
                        break

        if not (hit_json_date or hit_bank_date):
            continue

        if hit_json_date:
            where.append('IBM-izvod')
        if hit_bank_date:
            where.append('eBanking')

        new_matched_in = ','.join(where)
        new_level = 'date+amount (airfare+fee sum)'

        for j in (idx, fee_idx):
            df_all.at[j, 'Payment Method'] = 'company card'
            df_all.at[j, 'Matched In'] = new_matched_in
            df_all.at[j, 'Match Level'] = new_level
            df_all.at[j, 'AirfareFeeGroup'] = True
            if human_error_airfare:
                df_all.at[j, 'Human Error Suspected'] = True

        used_fee_idx.add(fee_idx)

    # ---------- 4c) Special rule: Hotel stays with date-based grouping ----------
    df_all, hotel_stay_count = process_hotel_stays(df_all, json_pair_set, bank_pair_set)

    # ---------- rebuild classification from df_all ----------
    classified = df_all[
        [
            "Payment Method",
            "Matched In",
            "Match Level",
            "AirfareFeeGroup",
            "HotelParkingGroup",
            "Human Error Suspected",
        ]
    ]

    # ---------- 5) Build final output ----------
    visible_cols = [
        'RPTKEY',
        'Emp Name',
        'Expense Type',
        'Expense Amt (txn currency)',
        'Expense Amt (RSD)',
    ]
    extra_cols = ['Transaction Date']

    final = pd.concat(
        [
            matched[visible_cols],
            classified,
            matched[extra_cols],
        ],
        axis=1,
    )

    if 'Human Error Suspected' in final.columns and not final['Human Error Suspected'].any():
        final = final.drop(columns=['Human Error Suspected'])

    if 'Txn Currency' in matched.columns:
        curr = matched['Txn Currency'].fillna('')
        final['Expense Amt (txn currency)'] = (
            final['Expense Amt (txn currency)'].astype(str) +
            (' ' + curr).where(curr != '', '')
        )

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        final.to_excel(Path(save_to), index=False)

    return final