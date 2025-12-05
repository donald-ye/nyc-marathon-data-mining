import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re

def get_distance(col_name):
    clean = col_name.replace('split_pace_', '').replace('split_time_', '')

    if clean == 'MAR':
        return 26.2
    if clean == 'HALF':
        return 13.1

    match_k = re.search(r'(\d+(\.\d+)?)K', clean)
    if match_k:
        km = float(match_k.group(1))
        return km * 0.621371  # convert km to miles

    match_m = re.search(r'(\d+(\.\d+)?)M', clean)
    if match_m:
        return float(match_m.group(1))

    return 0

df = pd.read_csv('/content/all_runners_2025.csv')

pace_cols = [c for c in df.columns if 'split_pace' in c]
pace_cols_sorted = sorted(pace_cols, key=get_distance)
distances = [get_distance(c) for c in pace_cols_sorted]

print(f"found {len(pace_cols_sorted)} split points")

pace_data = df[pace_cols_sorted].copy()

# fill gaps between known splits
pace_data = pace_data.interpolate(method='linear', axis=1, limit_direction='both')
pace_data = pace_data.dropna()
df = df.loc[pace_data.index].copy()

runner_start_pace = pace_data.iloc[:, 0]

# normalize
pace_normalized = pace_data.div(runner_start_pace, axis=0)

reliable_splits = []
reliable_distances = []

for col, dist in zip(pace_cols_sorted, distances):
    missing_pct = (df[col].isna().sum() / len(df))
    avg_normalized = (df[col] / runner_start_pace).mean()

    if missing_pct < 0.4 and 0.5 < avg_normalized < 2.0:
        reliable_splits.append(col)
        reliable_distances.append(dist)
    else:
        if missing_pct >= 0.4:
            reason = f"{missing_pct*100:.1f}% missing data"
        else:
            reason = f"anomalous pace factor: {avg_normalized:.3f}"
        print(f"{col:30s} ({dist:5.2f} mi): {reason}")

pace_cols_sorted = reliable_splits
distances = reliable_distances

# re-select the normalized data using only the reliable columns
X_data = pace_normalized[pace_cols_sorted].copy()
X_data = X_data.fillna(X_data.mean())

print(f"X_data ready with shape: {X_data.shape}")

# can skip this block; we know optimal clusters is 5
# elbow and silhouette analysis
# find optimal number of clusters
inertias = []
silhouettes = []
K_range = range(2, 11)

print("Testing different numbers of clusters...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_data)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_data, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.3f}")

# plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# fit k-means
k = 5

print(f"\nfitting k-means with k={k} clusters...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster_old'] = kmeans.fit_predict(X_data)

# pca
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_data)
df['pc1'] = pca_coords[:, 0]
df['pc2'] = pca_coords[:, 1]

print(f"pca explained variance: pc1={pca.explained_variance_ratio_[0]:.1%}, pc2={pca.explained_variance_ratio_[1]:.1%}")
print(f"total variance captured: {sum(pca.explained_variance_ratio_):.1%}")

# analyze and rank clusters
cluster_info = []

print(f"\n{'old cluster':<12} | {'end pace':<10} | {'fade %':<10}")
print("-" * 50)

for i in range(k):
    cluster_profile = X_data[df['cluster_old'] == i].mean(axis=0)
    end_pace = cluster_profile.iloc[-1]
    fade_pct = (end_pace - 1.0) * 100
    cluster_info.append((i, end_pace, fade_pct))
    print(f"{i:<12} | {end_pace:<10.3f} | {fade_pct:<10.1f}%")

cluster_info.sort(key=lambda x: x[1])

old_to_new = {}
new_to_label = {
    1: "cluster 1: even pacing",
    2: "cluster 2: mild slowing",
    3: "cluster 3: moderate slowing",
    4: "cluster 4: significant slowing",
    5: "cluster 5: severe crash"
}

print(f"\n{'new #':<8} | {'old #':<8} | {'end pace':<10} | {'fade %':<10} | {'label'}")
print("-" * 80)

for new_num, (old_num, end_pace, fade_pct) in enumerate(cluster_info, start=1):
    old_to_new[old_num] = new_num
    label = new_to_label[new_num]
    print(f"{new_num:<8} | {old_num:<8} | {end_pace:<10.3f} | {fade_pct:<10.1f}% | {label}")

# apply renumbering
df['cluster'] = df['cluster_old'].map(old_to_new)
df['cluster_name'] = df['cluster'].map(new_to_label)

print("\nfinal distribution (new numbering):")
for i in range(1, 6):
    count = (df['cluster'] == i).sum()
    pct = (count / len(df)) * 100
    print(f"  {new_to_label[i]}: {count:,} ({pct:.1f}%)")
