import asyncio
import aiohttp
import pandas as pd
import random
import time
from pathlib import Path

TEST_EVENT_CODE = 'M2025'
DETAIL_URL = 'https://rmsprodapi.nyrr.org/api/v2/runners/resultDetails'
HEADERS = {
    'accept': 'application/json, text/plain, */*',
    'content-type': 'application/json;charset=UTF-8',
    'origin': 'https://results.nyrr.org',
    'referer': 'https://results.nyrr.org/',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}
CONCURRENT_LIMIT = 3
MIN_DELAY_SECONDS = 1.0
MAX_RETRIES = 3
BACKOFF_DELAY = 5.0

async def fetch_runner_details_with_retry(session, runner_summary, rate_limit_semaphore, semaphore):
    runner_id = runner_summary.get('runnerId')
    name = f"{runner_summary.get('firstName')} {runner_summary.get('lastName')}"
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            async with rate_limit_semaphore:
                wait_time = random.uniform(MIN_DELAY_SECONDS, MIN_DELAY_SECONDS + 0.5)
                await asyncio.sleep(wait_time)
                try:
                    payload_detail = {'eventCode': TEST_EVENT_CODE, 'runnerId': runner_id}
                    async with session.post(DETAIL_URL, headers=HEADERS, json=payload_detail, timeout=aiohttp.ClientTimeout(total=15)) as response:
                        if response.status == 429:
                            await asyncio.sleep(BACKOFF_DELAY)
                            continue
                        response.raise_for_status()
                        detail_data = await response.json()
                    runner_details = detail_data.get('details', {})
                    splits = runner_details.get('splitResults', [])
                    if splits:
                        df_splits = pd.DataFrame(splits)
                        df_splits['RunnerName'] = name
                        df_splits['RunnerID'] = runner_id
                        df_splits['OverallTime'] = runner_summary.get('overallTime')
                        df_splits['OverallPlace'] = runner_summary.get('overallPlace')
                        df_splits['Gender'] = runner_summary.get('gender')
                        df_splits['Age'] = runner_summary.get('age')
                        df_splits['City'] = runner_summary.get('city')
                        df_splits['Country'] = runner_summary.get('countryCode')
                        df_splits['Bib'] = runner_summary.get('bib')
                        return df_splits
                    return None
                except:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2)
                        continue
                    return None
    return None

async def main():
    print("="*80)
    print("RESUME SPLIT FETCHER")
    print("="*80)
    checkpoint_file = 'checkpoint_splits_35000.csv'
    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint_df = pd.read_csv(checkpoint_file)
    completed_runners = set(checkpoint_df['RunnerID'].unique())
    print(f"Already completed: {len(completed_runners):,} runners")
    summary_file = 'nyrr_marathon_2025_summary_45885_runners_CLEAN.csv'
    print(f"Loading summary: {summary_file}")
    runners_df = pd.read_csv(summary_file)
    remaining_runners = runners_df[~runners_df['runnerId'].isin(completed_runners)]
    runners_list = remaining_runners.to_dict('records')
    print(f"Remaining: {len(runners_list):,} runners")
    print(f"Est time: ~{len(runners_list) * 1.25 / 60:.0f} min")
    input("\nPress Enter to resume...")
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        all_splits_data = []
        rate_limit_semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
        global_semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
        tasks = [fetch_runner_details_with_retry(session, r, rate_limit_semaphore, global_semaphore) for r in runners_list]
        print(f"\nResuming...\n")
        batch_size = 100
        successful = 0
        failed = 0
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) or result is None:
                    failed += 1
                else:
                    all_splits_data.append(result)
                    successful += 1
            completed = min(i + batch_size, len(tasks))
            total_completed = len(completed_runners) + completed
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining_time = (len(tasks) - completed) / rate if rate > 0 else 0
            print(f"[{total_completed:5d}/45885] Success: {successful:5d} | Failed: {failed:4d} | ETA: {remaining_time/60:.1f}min")
            if (len(completed_runners) + completed) % 5000 == 0 and all_splits_data:
                new_splits = pd.concat(all_splits_data, ignore_index=True)
                combined = pd.concat([checkpoint_df, new_splits], ignore_index=True)
                checkpoint_file_new = f'checkpoint_splits_{len(completed_runners) + completed}.csv'
                combined.to_csv(checkpoint_file_new, index=False)
                print(f"    Checkpoint: {checkpoint_file_new}")
        if all_splits_data:
            new_splits_df = pd.concat(all_splits_data, ignore_index=True)
            final_df = pd.concat([checkpoint_df, new_splits_df], ignore_index=True)
            output_filename = 'nyrr_marathon_2025_summary_45885_runners_CLEAN_WITH_SPLITS.csv'
            final_df.to_csv(output_filename, index=False)
            print(f"\nSaved: {output_filename}")
            print(f"Total: {final_df['RunnerID'].nunique():,} runners")

if __name__ == '__main__':
    asyncio.run(main())
