import pandas as pd

# 2025 one hot encoded bib numbers
wheelchair_bibs = [201, 202, 203, 205, 206, 250, 212, 204, 207, 208, 213, 209, 214, 251, 254, 252, 211, 217, 215, 255, 260, 253, 262, 261, 258, 210, 216, 265, 263, 257, 266, 278, 267, 268, 269, 272, 270, 276, 274, 291]
handcycle_bibs = [300, 309, 334, 329, 301, 316, 306, 307, 325, 321, 304, 322, 318, 341, 311, 350, 324, 315, 226, 327, 332, 317, 338, 337, 314, 1252, 386, 378, 308, 303, 310, 312, 397, 340, 385, 320, 328, 394, 384, 333, 395, 39, 388, 398 , 383, 387, 391, 377, 319, 396, 393, 389]

df = pd.read_csv('/content/nyrr_marathon_2025_summary_56480_runners_WITH_SPLITS.csv')

# label the participant types
df['participant_type'] = 'Runner'

df.loc[df['Bib'].isin(wheelchair_bibs), 'participant_type'] = 'Wheelchair'
df.loc[df['Bib'].isin(handcycle_bibs), 'participant_type'] = 'Handcycle'

df.to_csv('/content/nyrr_marathon_2025_summary_56480_runners_WITH_SPLITS.csv', index=False)
print(df['participant_type'].value_counts())

df_runners = df[df['participant_type'] == 'Runner'].copy()
print(f"Runners remaining: {len(df_runners):,}")

# convert time into seconds
def convert_pace_to_seconds(pace_str):
    if pd.isna(pace_str):
        return None
    try:
        parts = str(pace_str).split(':')
        if len(parts) == 3: # format HH:MM:SS (e.g., overall time)
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2: # format MM:SS (e.g., split pace)
            m, s = map(int, parts)
            return m * 60 + s
        else:
            return None
    except ValueError:
        return None

df_runners['pace_seconds'] = df_runners['pace'].apply(convert_pace_to_seconds)
df_runners = df_runners.dropna(subset=['pace_seconds'])
df_runners['OverallTime_seconds'] = df['OverallTime'].apply(convert_pace_to_seconds)
df_runners['split_time_seconds'] = df['time'].apply(convert_pace_to_seconds)
df_runners.drop(columns=['pace','OverallTime','time'], inplace=True)
df_runners = df_runners.rename(columns={'pace_seconds': 'pace', 'OverallTime_seconds': 'OverallTime', 'split_time_seconds' : 'time'})

# create two pivots, one for time and one for pace
pivot_time = df_runners.pivot_table(
    index='RunnerID',
    columns='splitCode',
    values='time',
    aggfunc='first'
)

pivot_time = pivot_time.add_prefix('time_')

pivot_pace = df_runners.pivot_table(
    index='RunnerID',
    columns='splitCode',
    values='pace',
    aggfunc='first'
)
pivot_pace = pivot_pace.add_prefix('pace_')
     
pivot_all = pd.concat([pivot_time, pivot_pace], axis=1)
# create a mapping so we have the times in the right order
distance_map = (
    df_runners[['splitCode', 'distance']]
    .drop_duplicates()
    .set_index('splitCode')['distance']
    .to_dict()
)

def sort_key(col):
    # column name looks like "time_3M" or "pace_5K"
    split = col.split('_')[1]           # e.g. "3M"
    return distance_map[split]

pivot_all = pivot_all.reindex(
    sorted(pivot_all.columns, key=sort_key),
    axis=1
)

# join the runner info so we have name, gender, overallplace, age, city, country...
runner_info = df_runners.drop_duplicates(subset='RunnerID').set_index('RunnerID')
final_df = runner_info.join(pivot_all, how='left')
final_df = final_df.rename(columns=lambda x: f"split_{x}" if x in pivot_all.columns else x)

final_df.dropna(subset='OverallTime', inplace=True)
final_df.drop(columns=['splitName'], inplace=True)
final_df.to_csv('/content/all_runners_2025.csv')
