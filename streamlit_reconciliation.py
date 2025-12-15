from pathlib import Path
import tempfile

import streamlit as st
from run_reconciliation import reconcile_and_save
from reconcile_expenses import parse_amount_to_cents
import pandas as pd

st.set_page_config(page_title="Bank Reconciliation", layout="centered")

st.title("Bank Reconciliation Tool")
st.write(
    "Upload your bank statement, eBanking PDF, CAAPS report and Expense Detail report, "
    "then generate the reconciled Excel file."
)

# --- File uploaders ---
st.subheader("Upload input files")

bank_pdf_files = st.file_uploader(
    "Bank card statement PDF(s)", type=["pdf"], key="bank_pdf", accept_multiple_files=True
)
ebanking_pdf_files = st.file_uploader(
    "eBanking PDF(s)", type=["pdf"], key="ebanking_pdf", accept_multiple_files=True
)

caaps_file = st.file_uploader(
    "CAAPS Excel file", type=["xls", "xlsx"], key="caaps"
)
expense_file = st.file_uploader(
    "Expense Detail Excel file", type=["xls", "xlsx"], key="expense"
)

# Show what has been uploaded so far (so the page is never "empty")
def status_icon(uploaded):
    return "✅" if uploaded else "❌"

st.markdown("### Upload Status")

col1, col2 = st.columns(2)

with col1:
    st.write(f"{status_icon(bank_pdf_files)} Bank PDF(s)")
    st.write(f"{status_icon(ebanking_pdf_files)} eBanking PDF(s)")


with col2:
    st.write(f"{status_icon(caaps_file)} CAAPS File")
    st.write(f"{status_icon(expense_file)} Expense File")

st.markdown("---")

# --- Options ---
st.subheader("Generate output")

default_out_name = "output_result.xlsx"
out_name = st.text_input("Output Excel file name", value=default_out_name)
skip_first_page = st.checkbox("Skip first page of bank statement PDF", value=True)


run_btn = st.button("▶ Run reconciliation")


def save_uploaded_to_temp(uploaded_file, base_dir: Path) -> Path:
    """Save an UploadedFile to a temporary directory and return its path."""
    dest_path = base_dir / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path

def save_many_uploaded_to_temp(uploaded_files, base_dir: Path) -> list[Path]:
    return [save_uploaded_to_temp(f, base_dir) for f in uploaded_files]


# --- Main action ---
if run_btn:
    # This ensures you see at least *something* on click
    st.info("Starting validation...")

    if not all([bank_pdf_files, ebanking_pdf_files, caaps_file, expense_file]):
        st.error("❌ Please upload all required files before running the reconciliation.")

    else:
        try:
            with st.spinner("⏳ Running reconciliation pipeline..."):
                # Create a temp directory for this run
                temp_dir = Path(tempfile.mkdtemp(prefix="reconcile_"))
                #st.write(f"Temporary working directory: `{temp_dir}`")

                bank_pdf_paths = save_many_uploaded_to_temp(bank_pdf_files, temp_dir)
                ebanking_pdf_paths = save_many_uploaded_to_temp(ebanking_pdf_files, temp_dir)
                caaps_path = save_uploaded_to_temp(caaps_file, temp_dir)
                expense_path = save_uploaded_to_temp(expense_file, temp_dir)

                out_path = temp_dir / out_name


                # Call your existing pipeline
                #st.write("Calling `reconcile_and_save`...")
                final_df = reconcile_and_save(
                    bank_pdf_path=bank_pdf_paths,
                    ebanking_pdf_path=ebanking_pdf_paths,
                    caaps_path=caaps_path,
                    expense_path=expense_path,
                    out_path=out_path,
                    skip_first_page=skip_first_page,
                )


            st.success(f"✅ Reconciliation completed. File created: `{out_path.name}`")

            st.subheader("Result")

            if final_df is None or final_df.empty:
                st.warning("The resulting DataFrame is empty.")
            else:
                # Keep flag for styling, but don't show it as a column
                if "AirfareFeeGroup" in final_df.columns:
                    airfare_flags = final_df["AirfareFeeGroup"].astype(bool)
                    hotel_parking_flags = (
                        final_df["HotelParkingGroup"].astype(bool)
                        if "HotelParkingGroup" in final_df.columns
                        else None
                    )
                    display_df = final_df.drop(
                            columns=[c for c in ["AirfareFeeGroup", "HotelParkingGroup"] if c in final_df.columns]
                    )
                else:
                    airfare_flags = None
                    display_df = final_df
 
                # ---- HIGHLIGHT HOTEL EXPENSE TYPES ----
                def highlight_hotel(col):
                    etype = col.astype(str).str.upper()
                    # Base: all HOTEL* rows
                    mask = etype.str.contains("HOTEL", na=False)

                    # Add PARKING rows that belong to a hotel group
                    if 'hotel_parking_flags' in globals() or 'hotel_parking_flags' in locals():
                        if hotel_parking_flags is not None:
                            flags = hotel_parking_flags.reindex(col.index, fill_value=False)
                            parking_mask = etype.str.contains("PARKING", na=False) & flags
                            mask = mask | parking_mask

                    return ['background-color: #FFF2CC' if x else '' for x in mask]

                # ---- HIGHLIGHT AIRFARE + AIRLINE FEE ON *Expense Type* ONLY ----
                def highlight_airfare_fee(col):
                    # col is the 'Expense Type' column because of subset=['Expense Type']
                    if airfare_flags is None:
                        return [''] * len(col)
                    flags = airfare_flags.reindex(col.index, fill_value=False)
                    return ['background-color: #CCECFF' if flag else '' for flag in flags]
                def highlight_daily_allow(col):
                    # col will be the 'Expense Type' column when used with subset=['Expense Type']
                    mask = col.astype(str).str.upper().str.contains("DAILY ALLOW", na=False)
                    return ['background-color: #EAD9FF' if x else '' for x in mask]  # light purple

                # ---- cash (red) ----
                def highlight_allowance_and_cash(row):
                    styles = [''] * len(row)
                    etype = str(row.get('Expense Type', '')).upper()
                    payment = str(row.get('Payment Method', '')).lower()

                    # ❌ Do NOT highlight cash if it's a Daily Allowance row
                    if payment == 'cash' and 'DAILY ALLOW' not in etype:
                        return ['background-color: #FFD6D6'] * len(row)  # light red

                    return styles


            
                styled_df = (
                    display_df
                    .style
                    .apply(highlight_hotel, subset=['Expense Type'])
                    .apply(highlight_airfare_fee, subset=['Expense Type'])
                    .apply(highlight_daily_allow, subset=['Expense Type'])
                    .apply(highlight_allowance_and_cash, axis=1)
                    .format({
                        "Expense Amt (RSD)": lambda x: f"{float(x):.2f}"
                    })
                )

                
                st.dataframe(
                    styled_df,
                    width='stretch',
                    hide_index=True
                )

                st.markdown("""
                **Legend:**
                - 🟨 Hotel-related expenses grouped
                - 🟦 Airfare + Airline Fee combined match
                - 🟪 Daily Allowance
                - 🟥 Cash payment 
                """)
                
                #st.subheader("Summary")

                # ----- Base summary from RESULT (display_df) -----
                summary_df = display_df.copy()
                summary_df["RPTKEY"] = summary_df["RPTKEY"].astype(str)

                # Order of RPTKEYs that actually appear in the result
                result_rptkey_order = summary_df["RPTKEY"].drop_duplicates().tolist()

                # One employee name per RPTKEY (from result, if present)
                name_map = (
                    summary_df
                    .groupby("RPTKEY", sort=False)["Emp Name"]
                    .agg(lambda s: s.iloc[0])
                    .reset_index()
                )

                # Convert amounts to numeric cents
                summary_df["amount_rsd_cents"] = summary_df["Expense Amt (RSD)"].apply(
                    parse_amount_to_cents
                )
                summary_df = summary_df.dropna(subset=["amount_rsd_cents"])

                # Totals per RPTKEY + Payment Method (from result)
                pivot = (
                    summary_df
                    .groupby(["RPTKEY", "Payment Method"], sort=False)["amount_rsd_cents"]
                    .sum()
                    .unstack(fill_value=0)
                )

                for col in ["company card", "cash"]:
                    if col not in pivot.columns:
                        pivot[col] = 0

                summary_out = pivot[["company card", "cash"]] / 100.0
                summary_out = summary_out.rename(columns={
                    "company card": "Card Total (RSD)",
                    "cash": "Cash Total (RSD)",
                })
                summary_out["Total (RSD)"] = (
                    summary_out["Card Total (RSD)"] +
                    summary_out["Cash Total (RSD)"]
                )

                summary_out_reset = summary_out.reset_index()

                # Attach Emp Name from result where available
                summary_out_reset = summary_out_reset.merge(name_map, on="RPTKEY", how="left")

                # ----- Now bring in ALL RPTKEYs from CAAPS -----
                import pandas as pd

                df_raw_caaps = pd.read_excel(caaps_path, header=None)
                section2_row = df_raw_caaps[
                    df_raw_caaps.iloc[:, 0]
                    .astype(str)
                    .str.contains("SECTION 2", case=False, na=False)
                ].index[0]
                header_row_caaps = section2_row + 1

                caaps_df = pd.read_excel(caaps_path, header=header_row_caaps)
                caaps_df.columns = caaps_df.columns.str.replace('\n', ' ').str.strip()

                # Simple column finder for CAAPS
                def find_col(cols, keyword):
                    keyword = keyword.lower()
                    for c in cols:
                        if keyword in c.lower():
                            return c
                    return None

                caaps_rptkey_col = find_col(caaps_df.columns, "report key")
                caaps_name_col = find_col(caaps_df.columns, "name")  # if there is an employee name column

                # Raw RPTKEY + strip
                caaps_df["RPTKEY"] = caaps_df[caaps_rptkey_col].astype(str).str.strip()

                # ✅ Keep ONLY proper CTR keys like CTR0123456789
                ctr_mask = caaps_df["RPTKEY"].str.match(r"^CTR\d+$", na=False)
                caaps_valid = caaps_df[ctr_mask].copy()

                if caaps_name_col:
                    caaps_valid["Emp Name CAAPS"] = caaps_valid[caaps_name_col].astype(str)
                else:
                    caaps_valid["Emp Name CAAPS"] = ""

                # unique valid RPTKEYs from CAAPS, in CAAPS order
                caaps_unique = (
                    caaps_valid[["RPTKEY", "Emp Name CAAPS"]]
                    .drop_duplicates("RPTKEY", keep="first")
                )
                caaps_rptkey_order = caaps_unique["RPTKEY"].tolist()


                # ----- Identify CAAPS RPTKEYs missing in the result summary -----
                existing_keys = set(summary_out_reset["RPTKEY"].astype(str))
                missing_caaps = caaps_unique[~caaps_unique["RPTKEY"].isin(existing_keys)]

                # Build rows for CAAPS-only entries (no expenses/bank data)
                # They get 0 totals and a status message.
                if not missing_caaps.empty:
                    missing_rows = pd.DataFrame({
                        "RPTKEY": missing_caaps["RPTKEY"],
                        "Card Total (RSD)": 0.0,
                        "Cash Total (RSD)": 0.0,
                        "Total (RSD)": 0.0,
                        "Emp Name": missing_caaps["Emp Name CAAPS"],
                        "Status": "No match"
                    })
                else:
                    missing_rows = pd.DataFrame(columns=["RPTKEY", "Card Total (RSD)", "Cash Total (RSD)", "Total (RSD)", "Emp Name", "Status"])

                # Existing summary rows get a status (empty/OK)
                summary_out_reset["Status"] = ""

                # Align column order (we will remove Status later if not needed)
                base_cols = ["RPTKEY", "Emp Name", "Card Total (RSD)", "Cash Total (RSD)", "Total (RSD)", "Status"]
                summary_out_reset = summary_out_reset[base_cols]

                # Concatenate existing + missing
                full_summary = pd.concat(
                    [summary_out_reset, missing_rows[base_cols]],
                    ignore_index=True
                )

                # If NO missing rows → remove the Status column entirely
                if missing_rows.empty:
                    full_summary = full_summary.drop(columns=["Status"])
                else:
                    # Otherwise keep Status and sort by CAAPS order
                    full_summary["RPTKEY"] = full_summary["RPTKEY"].astype(str)
                    full_summary["RPTKEY"] = pd.Categorical(
                        full_summary["RPTKEY"],
                        categories=caaps_rptkey_order,
                        ordered=True
                    )
                    full_summary = full_summary.sort_values("RPTKEY", kind="stable")


                #st.dataframe(
                #    full_summary,
                #    use_container_width=True,
                #    hide_index=True
                #)
            # ----- Extra: Per-RPTKEY Cash Breakdown -----
            st.subheader("Cash Breakdown (RSD)")

            # Work from visible result table
            res_df = display_df.copy()
            res_df["RPTKEY"] = res_df["RPTKEY"].astype(str)

            # Cash-only rows
            cash_rows = res_df[res_df["Payment Method"] == "cash"].copy()

            if cash_rows.empty:
                st.info("No cash payments found for any RPTKEY.")
            else:
                # Flag daily allowance
                cash_rows["is_daily_allow"] = (
                    cash_rows["Expense Type"]
                    .astype(str)
                    .str.upper()
                    .str.contains("DAILY ALLOW", na=False)
                )
                # Flag personal car mileage
                cash_rows["is_car_mileage"] = (
                    cash_rows["Expense Type"]
                    .astype(str)
                    .str.upper()
                    .str.contains("PERSONAL CAR MILEAGE", na=False)
                )
                # Parse amounts
                cash_rows["amount_rsd_cents"] = cash_rows["Expense Amt (RSD)"].apply(
                    parse_amount_to_cents
                )
                cash_rows = cash_rows.dropna(subset=["amount_rsd_cents"])

                # Emp Name per RPTKEY
                emp_map = (
                    res_df.groupby("RPTKEY")["Emp Name"]
                    .agg(lambda s: s.iloc[0])
                    .to_dict()
                )

                # Compute per-category totals per RPTKEY
                breakdown = []

                for rptkey, group in cash_rows.groupby("RPTKEY"):
                    daily = group.loc[group["is_daily_allow"], "amount_rsd_cents"].sum() / 100.0
                    mileage = group.loc[group["is_car_mileage"], "amount_rsd_cents"].sum() / 100.0
                    other = group.loc[
                        ~(group["is_daily_allow"] | group["is_car_mileage"]),
                        "amount_rsd_cents"
                    ].sum() / 100.0

                    total = daily + mileage + other

                    breakdown.append({
                        "RPTKEY": rptkey,
                        "Emp Name": emp_map.get(rptkey, ""),
                        "Daily Allowance ": daily,
                        "Personal Car Mileage": mileage,
                        "Other": other,
                        "Total": total
                    })

                rptkey_breakdown = pd.DataFrame(breakdown)

                # Only keep RPTKEYs with non-zero cash total in summary
                summary_with_cash = full_summary[
                    full_summary["Cash Total (RSD)"] > 0
                ]["RPTKEY"].astype(str)

                rptkey_breakdown = rptkey_breakdown[
                    rptkey_breakdown["RPTKEY"].isin(summary_with_cash)
                ]

                if rptkey_breakdown.empty:
                    st.info("No RPTKEYs with non-zero cash spending.")
                else:
                    # Apply CAAPS order
                    rptkey_breakdown["RPTKEY"] = pd.Categorical(
                        rptkey_breakdown["RPTKEY"],
                        categories=caaps_rptkey_order,
                        ordered=True
                    )
                    rptkey_breakdown = rptkey_breakdown.sort_values("RPTKEY", kind="stable")

                    # Display without index
                    st.dataframe(
                        rptkey_breakdown,
                        width='stretch',
                        hide_index=True
                    )

            # Download button for the Excel file
            st.subheader("Download")
            from pandas import ExcelWriter

            try:
                # Write styled DataFrame with colors to Excel
                with ExcelWriter(out_path, engine="openpyxl") as writer:
                    styled_df.to_excel(writer, sheet_name="Result", index=False)

                # Offer it for download
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Excel file",
                        data=f.read(),
                        file_name=out_path.name,
                        mime=(
                            "application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"
                        ),
                    )

            except FileNotFoundError:
                st.error(
                    "The Excel file was not found on disk after writing. "
                    "Please check that the path is valid and writable."
                )
            except Exception as e:
                st.error(f"Error while creating or reading the Excel file: `{e}`")


        except Exception as e:
            # This guarantees the error appears on the page
            st.error(f"💥 An error occurred while running reconciliation:\n\n`{e}`")
