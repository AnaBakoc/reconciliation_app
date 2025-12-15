import pandas as pd
from itertools import combinations
def process_hotel_stays(df_all, json_pair_set, bank_pair_set):
    """
    Process hotel stays with date-based grouping and smart parking assignment.
    
    Parameters:
    -----------
    df_all : DataFrame
        The main dataframe containing all expense items
    json_pair_set : set
        Set of (name, amount_cents) tuples from JSON payment records
    bank_pair_set : set
        Set of (name, amount_cents) tuples from bank statement records
        
    Returns:
    --------
    DataFrame
        Updated dataframe with hotel stay matching
    int
        Total number of hotel stays processed
    """
    
    # Helper functions first
    def find_parking_for_hotel(hotel_date, hotel_currency, parking_items, used_parkings, hotel_base_items):
        """Find the best parking for a hotel date without double-assignment"""
        # First try same day parking
        same_day_parking = parking_items[
            (parking_items['Transaction Date'] == hotel_date) &
            (parking_items.get('Currency', '') == hotel_currency) &
            (~parking_items.index.isin(used_parkings))
        ]
        
        if not same_day_parking.empty:
            # Take the first matching parking
            return same_day_parking.index[0]
        
        # Try next day parking (for overnight stays)
        next_day = hotel_date + pd.Timedelta(days=1)
        next_day_parking = parking_items[
            (parking_items['Transaction Date'] == next_day) &
            (parking_items.get('Currency', '') == hotel_currency) &
            (~parking_items.index.isin(used_parkings))
        ]
        
        if not next_day_parking.empty:
            # Check if there's another hotel the next day
            hotels_next_day = hotel_base_items[
                (hotel_base_items['Transaction Date'] == next_day) &
                (hotel_base_items['Expense Type Canon'] == 'HOTEL')
            ]
            
            # If no hotel next day, this parking likely belongs to current hotel
            if hotels_next_day.empty:
                return next_day_parking.index[0]
        
        return None
    
    def mark_items_as_matched(df, indices, stay_id, matched_where, stay_counter):
        """Mark items as matched with company card"""
        for idx in indices:
            df.at[idx, 'Payment Method'] = 'company card'
            df.at[idx, 'Matched In'] = matched_where
            df.at[idx, 'Match Level'] = 'amount-only (hotel stay)' if stay_id == 'single' else 'amount-only (multi-night hotel stay)'
            df.at[idx, 'HotelParkingGroup'] = True
            df.at[idx, 'hotel_stay_id'] = stay_counter
        
        return df
    
    def try_match_combination(name_canon, amount_cents, json_pair_set, bank_pair_set):
        """Try to match a combination amount with payment records"""
        hit_json = (name_canon, amount_cents) in json_pair_set
        hit_bank = (name_canon, amount_cents) in bank_pair_set
        
        if hit_json or hit_bank:
            where = []
            if hit_json:
                where.append('IBM-izvod')
            if hit_bank:
                where.append('eBanking')
            return ','.join(where)
        return None
    
    # Main processing function for one employee group
    def process_employee_group(df_all, sub_df, first_name_canon, stay_counter, json_pair_set, bank_pair_set):
        """Process hotel stays for a single employee"""
        hotel_items = sub_df[sub_df['is_hotel_base'] | sub_df['is_parking']].copy()
        
        if hotel_items.empty:
            return df_all, stay_counter
        
        # Ensure Transaction Date is datetime
        hotel_items['Transaction Date'] = pd.to_datetime(hotel_items['Transaction Date'], errors='coerce')
        
        # Get currency information
        hotel_items['Currency'] = hotel_items['Txn Currency'].fillna('') if 'Txn Currency' in hotel_items.columns else ''
        
        # Get hotel and parking items
        hotel_base_items = hotel_items[hotel_items['is_hotel_base']].copy()
        parking_items = hotel_items[hotel_items['is_parking']].copy()
        
        if hotel_base_items.empty:
            return df_all, stay_counter
        
        # Step 1: Create stay candidates with proper parking assignment
        stay_candidates = []
        assigned_parking_indices = set()
        
        # Get all hotel charges
        hotel_charges = hotel_base_items[hotel_base_items['Expense Type Canon'] == 'HOTEL'].copy()
        
        for idx, hotel_row in hotel_charges.iterrows():
            hotel_date = hotel_row['Transaction Date']
            hotel_currency = hotel_row.get('Currency', '')
            
            if pd.isna(hotel_date):
                continue
            
            # 1. Hotel tax on same date
            same_day_taxes = hotel_base_items[
                (hotel_base_items['Expense Type Canon'] == 'HOTEL TAX') &
                (hotel_base_items['Transaction Date'] == hotel_date) &
                (hotel_base_items.get('Currency', '') == hotel_currency)
            ]
            
            # 2. Breakfast next day
            next_day = hotel_date + pd.Timedelta(days=1)
            breakfasts = hotel_base_items[
                (hotel_base_items['Expense Type Canon'] == 'HOTEL INVOICE BREAKFAST') &
                (hotel_base_items['Transaction Date'] == next_day) &
                (hotel_base_items.get('Currency', '') == hotel_currency)
            ]
            
            # 3. Find appropriate parking
            parking_idx = find_parking_for_hotel(
                hotel_date, hotel_currency, parking_items, 
                assigned_parking_indices, hotel_base_items
            )
            
            # Create candidate
            candidate_items = [idx]
            candidate_items.extend(same_day_taxes.index.tolist())
            candidate_items.extend(breakfasts.index.tolist())
            
            if parking_idx:
                candidate_items.append(parking_idx)
                assigned_parking_indices.add(parking_idx)
            
            stay_candidates.append({
                'hotel_idx': idx,
                'hotel_date': hotel_date,
                'items': list(set(candidate_items)),
                'currency': hotel_currency,
                'total_amount': sum(int(df_all.at[i, 'amount_cents_abs']) for i in candidate_items)
            })
        
        if not stay_candidates:
            return df_all, stay_counter
        
        # Step 2: Sort and process candidates
        stay_candidates.sort(key=lambda x: x['hotel_date'])
        matched_indices = set()
        
        # First pass: Individual candidates
        for candidate in stay_candidates:
            if any(idx in matched_indices for idx in candidate['items']):
                continue
            
            candidate_items = [idx for idx in candidate['items'] if idx not in matched_indices]
            if not candidate_items:
                continue
                
            candidate_sum = sum(int(df_all.at[idx, 'amount_cents_abs']) for idx in candidate_items)
            matched_where = try_match_combination(first_name_canon, candidate_sum, json_pair_set, bank_pair_set)
            
            if matched_where:
                stay_counter += 1
                df_all = mark_items_as_matched(df_all, candidate_items, 'single', matched_where, stay_counter)
                matched_indices.update(candidate_items)
        
        # Second pass: Multi-night stays
        remaining_candidates = [c for c in stay_candidates if not any(idx in matched_indices for idx in c['items'])]
        remaining_candidates.sort(key=lambda x: x['hotel_date'])
        
        i = 0
        while i < len(remaining_candidates):
            candidate = remaining_candidates[i]
            candidate_items = [idx for idx in candidate['items'] if idx not in matched_indices]
            
            if not candidate_items:
                i += 1
                continue
            
            # Build multi-night stay
            multi_night_items = candidate_items.copy()
            multi_night_total = sum(int(df_all.at[idx, 'amount_cents_abs']) for idx in multi_night_items)
            current_date = candidate['hotel_date']
            nights_included = 1
            
            # Look for consecutive nights
            j = i + 1
            while j < len(remaining_candidates):
                other_candidate = remaining_candidates[j]
                other_items = [idx for idx in other_candidate['items'] if idx not in matched_indices]
                
                if not other_items:
                    j += 1
                    continue
                
                # Check if consecutive night with same currency
                if (other_candidate['hotel_date'] == current_date + pd.Timedelta(days=1) and
                    other_candidate['currency'] == candidate['currency']):
                    
                    multi_night_items.extend(other_items)
                    multi_night_total += sum(int(df_all.at[idx, 'amount_cents_abs']) for idx in other_items)
                    current_date = other_candidate['hotel_date']
                    nights_included += 1
                    j += 1
                else:
                    break
            
            if nights_included > 1:
                matched_where = try_match_combination(first_name_canon, multi_night_total, json_pair_set, bank_pair_set)
                if matched_where:
                    stay_counter += 1
                    df_all = mark_items_as_matched(df_all, multi_night_items, 'multi', matched_where, stay_counter)
                    matched_indices.update(multi_night_items)
                    i += nights_included
                else:
                    i += 1
            else:
                i += 1
        
        # Third pass: Greedy combinations for leftovers
        all_hotel_indices = hotel_items.index.tolist()
        unmatched_indices = [idx for idx in all_hotel_indices if idx not in matched_indices]
        
        if unmatched_indices:
            # Group by currency
            currency_groups = {}
            for idx in unmatched_indices:
                currency = df_all.at[idx, 'Txn Currency'] if 'Txn Currency' in df_all.columns else ''
                currency = '' if pd.isna(currency) else currency
                currency_groups.setdefault(currency, []).append(idx)
            
            for currency, currency_indices in currency_groups.items():
                if not currency_indices:
                    continue
                
                # Need at least one hotel charge
                if not any(df_all.at[idx, 'Expense Type Canon'] in {'HOTEL', 'HOTEL TAX'} for idx in currency_indices):
                    continue
                
                # Try combinations
                found_match = False
                for size in range(1, len(currency_indices) + 1):
                    if found_match:
                        break
                    
                    for combo in combinations(currency_indices, size):
                        # Check if combo has hotel charge
                        if not any(df_all.at[idx, 'Expense Type Canon'] in {'HOTEL', 'HOTEL TAX'} for idx in combo):
                            continue
                        
                        combo_sum = sum(int(df_all.at[idx, 'amount_cents_abs']) for idx in combo)
                        matched_where = try_match_combination(first_name_canon, combo_sum, json_pair_set, bank_pair_set)
                        
                        if matched_where:
                            stay_counter += 1
                            df_all = mark_items_as_matched(df_all, list(combo), 'combo', matched_where, stay_counter)
                            matched_indices.update(combo)
                            found_match = True
                            break
        
        return df_all, stay_counter
    
    # ----- Main execution starts here -----
    
    # First, reset all hotel-related items
    hotel_related_mask = df_all['is_hotel_base'] | df_all['is_parking']
    df_all.loc[hotel_related_mask, 'Payment Method'] = 'cash'
    df_all.loc[hotel_related_mask, 'Matched In'] = 'none'
    df_all.loc[hotel_related_mask, 'Match Level'] = 'none'
    df_all.loc[hotel_related_mask, 'hotel_stay_id'] = pd.NA
    df_all.loc[hotel_related_mask, 'HotelParkingGroup'] = False
    
    total_stays = 0
    
    # Process each employee group
    for (rptkey, emp), sub_df in df_all.groupby(['RPTKEY', 'Emp Name'], sort=False):
        if sub_df.empty:
            continue
        
        first_name_canon = sub_df.iloc[0]['name_canon']
        df_all, total_stays = process_employee_group(
            df_all, sub_df, first_name_canon, total_stays, 
            json_pair_set, bank_pair_set
        )
    
    return df_all, total_stays


# Usage in your main code:
# df_all, hotel_stay_count = process_hotel_stays(df_all, json_pair_set, bank_pair_set)


