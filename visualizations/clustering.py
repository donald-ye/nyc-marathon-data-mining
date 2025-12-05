color_map = {
    'cluster 1: even pacing': 'forestgreen',
    'cluster 2: mild slowing': 'dodgerblue',
    'cluster 3: moderate slowing': 'gold',
    'cluster 4: significant slowing': 'darkorange',
    'cluster 5: severe crash': 'firebrick'
}

plt.figure(figsize=(14, 9))
sns.scatterplot(
    x='pc1', y='pc2',
    hue='cluster_name',
    data=df,
    palette=color_map,
    alpha=0.6,
    s=20,
    edgecolor=None
)

plt.title('nyc marathon pacing strategies', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('pc1: how much you faded (91.7%)', fontsize=13, fontweight='bold')
plt.ylabel('pc2: timing of fade (6.0%)', fontsize=13, fontweight='bold')
plt.legend(title='pacing strategy', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 9))

cluster_order = sorted(df['cluster'].unique())

for new_num in cluster_order:
    label = new_to_label[new_num]
    color = color_map.get(label, 'gray')

    profile = X_data[df['cluster'] == new_num].mean(axis=0)
    cluster_size = (df['cluster'] == new_num).sum()
    cluster_pct = (cluster_size / len(df)) * 100

    # label with percentage
    display_label = f"{label} ({cluster_pct:.1f}%)"

    plt.plot(distances, profile, label=display_label, color=color, linewidth=3.5, alpha=0.9)

    plt.text(distances[-1] + 0.3, profile.iloc[-1],
             f"C{new_num}",
             fontsize=10, color=color, fontweight='bold',
             va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

plt.axhline(1.0, color='black', linestyle='--', linewidth=2.5, alpha=0.6,
            label='perfect even pace', zorder=0)

# shaded zones
plt.axhspan(0.95, 1.05, alpha=0.1, color='green', zorder=0, label='optimal zone (±5%)')
plt.axhspan(1.20, 1.50, alpha=0.1, color='red', zorder=0, label='crisis zone (20%+ fade)')

plt.title('nyc marathon pacing strategy profiles: how runners fade',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('distance (miles)', fontsize=14, fontweight='bold')
plt.ylabel('pace factor (1.0 = starting pace)', fontsize=14, fontweight='bold')

ax = plt.gca()
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_ylabel('percent change from start pace', fontsize=14, fontweight='bold')

y_ticks = ax.get_yticks()
ax2.set_yticks(y_ticks)
ax2.set_yticklabels([f'{int((y-1)*100):+d}%' for y in y_ticks])

plt.grid(True, alpha=0.25, linestyle='--')
plt.ylim(0.95, 1.40)
ax.set_ylim(0.95, 1.40)

ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True,
          title='pacing strategy (% of runners)', title_fontsize=12)

plt.tight_layout()
plt.show()

# can skip this block; we know optimal clusters is 5

from sklearn.metrics import silhouette_score

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
