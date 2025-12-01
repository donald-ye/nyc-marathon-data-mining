import asyncio
import aiohttp
import pandas as pd
import time
import sys
from typing import Dict

LIST_URL = 'https://rmsprodapi.nyrr.org/api/v2/runners/finishers-filter'
HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'content-type': 'application/json;charset=UTF-8',
    'origin': 'https://results.nyrr.org',
    'referer': 'https://results.nyrr.org/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

all_unique_runners: Dict[int, dict] = {}

async def fetch_runners_with_params(session, strategy_name: str, payload_base: dict, max_pages: int = 5) -> int:
    new_runners_count = 0
    last_checkpoint = (len(all_unique_runners) // 10000) * 10000
    for page_idx in range(1, max_pages + 1):
        payload = {**payload_base, 'pageIndex': page_idx, 'pageSize': 100}
        try:
            async with session.post(LIST_URL, headers=HEADERS, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                data = await response.json()
            runners = data.get('items', [])
            if not runners:
                break
            page_new = 0
            for runner in runners:
                runner_id = runner['runnerId']
                if runner_id not in all_unique_runners:
                    all_unique_runners[runner_id] = runner
                    page_new += 1
            new_runners_count += page_new
            if page_new > 0 or page_idx == 1:
                print(f"  {strategy_name[:50]:50s} P{page_idx}: {page_new:3d} new | Total: {len(all_unique_runners):,}")
            current_total = len(all_unique_runners)
            current_checkpoint = (current_total // 10000) * 10000
            if current_checkpoint > last_checkpoint and current_checkpoint > 0:
                year = payload_base.get('eventCode', 'M2025').replace('M', '')
                runners_df = pd.DataFrame(list(all_unique_runners.values()))
                checkpoint_file = f'checkpoint_runners_{year}_{current_checkpoint}.csv'
                runners_df.to_csv(checkpoint_file, index=False)
                print(f"\n  CHECKPOINT: {checkpoint_file} ({current_total:,} runners)\n")
                last_checkpoint = current_checkpoint
            if page_new == 0:
                break
            await asyncio.sleep(0.15)
        except:
            break
    return new_runners_count

async def strategy_all_three_letter_combos(session, event_code):
    print("\n" + "="*80)
    print("STRATEGY: ALL THREE-LETTER COMBINATIONS (17,576 combos)")
    print("="*80)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    priority_first = 'JMCSABDEKLRNTPW'
    other_first = ''.join(c for c in letters if c not in priority_first)
    all_first = priority_first + other_first
    total_new = 0
    for first in all_first:
        for second in letters:
            for third in letters:
                search_term = first + second + third
                payload_base = {'eventCode': event_code, 'searchString': search_term, 'sortColumn': 'overallPlace', 'sortDescending': False}
                new = await fetch_runners_with_params(session, f"'{search_term}'", payload_base, max_pages=5)
                total_new += new
    print(f"\nThree-letter combos added: {total_new:,} new runners")
    return total_new

async def strategy_all_two_letter_all_sorts(session, event_code):
    print("\n" + "="*80)
    print("STRATEGY: ALL TWO-LETTER x ALL SORTS (8,112 queries)")
    print("="*80)
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sort_columns = ['overallPlace', 'ageGradePlace', 'genderPlace', 'bib', 'lastName', 'firstName']
    total_new = 0
    for first in letters:
        for second in letters:
            combo = first + second
            for sort_col in sort_columns:
                for sort_desc in [False, True]:
                    payload_base = {'eventCode': event_code, 'searchString': combo, 'sortColumn': sort_col, 'sortDescending': sort_desc}
                    new = await fetch_runners_with_params(session, f"'{combo}'+{sort_col[:3]}", payload_base, max_pages=5)
                    total_new += new
    print(f"\nTwo-letter x all sorts added: {total_new:,} new runners")
    return total_new

async def strategy_empty_search_all_sorts(session, event_code):
    print("\n" + "="*80)
    print("STRATEGY: EMPTY SEARCH x ALL SORTS")
    print("="*80)
    sort_columns = ['overallPlace', 'ageGradePlace', 'genderPlace', 'bib', 'lastName', 'firstName']
    total_new = 0
    for sort_col in sort_columns:
        for sort_desc in [False, True]:
            payload_base = {'eventCode': event_code, 'searchString': '', 'sortColumn': sort_col, 'sortDescending': sort_desc}
            new = await fetch_runners_with_params(session, f"Empty+{sort_col}", payload_base, max_pages=5)
            total_new += new
    print(f"\nEmpty search x sorts added: {total_new:,} new runners")
    return total_new

async def strategy_single_chars_extended(session, event_code):
    print("\n" + "="*80)
    print("STRATEGY: SINGLE CHARACTERS x ALL SORTS")
    print("="*80)
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    sort_columns = ['overallPlace', 'ageGradePlace', 'genderPlace', 'bib', 'lastName', 'firstName']
    total_new = 0
    for char in chars:
        for sort_col in sort_columns:
            for sort_desc in [False, True]:
                payload_base = {'eventCode': event_code, 'searchString': char, 'sortColumn': sort_col, 'sortDescending': sort_desc}
                new = await fetch_runners_with_params(session, f"'{char}'+{sort_col[:3]}", payload_base, max_pages=5)
                total_new += new
    print(f"\nSingle chars x sorts added: {total_new:,} new runners")
    return total_new

async def strategy_all_states(session, event_code):
    print("\n" + "="*80)
    print("STRATEGY: ALL US STATE CODES")
    print("="*80)
    states = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
    total_new = 0
    for state in states:
        payload_base = {'eventCode': event_code, 'searchString': state, 'sortColumn': 'overallPlace', 'sortDescending': False}
        new = await fetch_runners_with_params(session, f"State:{state}", payload_base, max_pages=5)
        total_new += new
    print(f"\nState searches added: {total_new:,} new runners")
    return total_new

async def strategy_all_countries(session, event_code):
    print("\n" + "="*80)
    print("STRATEGY: EXTENDED COUNTRY CODES")
    print("="*80)
    countries = ['USA','CAN','MEX','GBR','FRA','DEU','ITA','ESP','NLD','BEL','CHE','AUT','POL','CZE','HUN','ROU','BGR','GRC','PRT','IRL','DNK','NOR','SWE','FIN','ISL','RUS','UKR','BRA','ARG','CHL','COL','PER','VEN','ECU','CHN','JPN','KOR','IND','PAK','BGD','IDN','THA','VNM','PHL','MYS','SGP','AUS','NZL','ZAF','KEN','ETH','NGA','EGY','MAR','TUN','ISR','TUR','SAU','ARE','QAT']
    total_new = 0
    for country in countries:
        payload_base = {'eventCode': event_code, 'searchString': country, 'sortColumn': 'overallPlace', 'sortDescending': False}
        new = await fetch_runners_with_params(session, f"Country:{country}", payload_base, max_pages=5)
        total_new += new
    print(f"\nCountry searches added: {total_new:,} new runners")
    return total_new

async def main():
    if len(sys.argv) < 2:
        print("Usage: python nyrr_ultra_aggressive_fetch.py <year>")
        return
    year = sys.argv[1]
    event_code = f'M{year}'
    start_time = time.time()
    print("\n" + "="*80)
    print(f"NYC MARATHON {year} - ULTRA AGGRESSIVE COLLECTION")
    print("="*80)
    print(f"Target: ~59,600 runners (100% coverage)")
    print(f"This will take 1-2 hours")
    print("="*80)
    input("\nPress Enter to start...")
    async with aiohttp.ClientSession() as session:
        print("\nPHASE 1: Foundation queries...")
        await strategy_empty_search_all_sorts(session, event_code)
        await strategy_single_chars_extended(session, event_code)
        print("\nPHASE 2: Two-letter coverage...")
        await strategy_all_two_letter_all_sorts(session, event_code)
        print("\nPHASE 3: Geographic coverage...")
        await strategy_all_states(session, event_code)
        await strategy_all_countries(session, event_code)
        print("\nPHASE 4: Three-letter exhaustive (30-60 min)...")
        await strategy_all_three_letter_combos(session, event_code)
        end_time = time.time()
        print("\n" + "="*80)
        print("COLLECTION COMPLETE")
        print("="*80)
        print(f"Total: {len(all_unique_runners):,}")
        print(f"Coverage: {len(all_unique_runners)/59600*100:.2f}%")
        print(f"Time: {(end_time-start_time)/60:.1f} min")
        if len(all_unique_runners) > 0:
            runners_df = pd.DataFrame(list(all_unique_runners.values()))
            summary_filename = f'nyrr_marathon_{year}_summary_{len(all_unique_runners)}_runners.csv'
            runners_df.to_csv(summary_filename, index=False)
            print(f"\nSaved: {summary_filename}")

if __name__ == '__main__':
    asyncio.run(main())
