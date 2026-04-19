import matplotlib.pyplot as plt
from data_processing import metrics_df

# metrics for plotting
k_values = [3, 5, 10]
precision = metrics_df['Precision@K'].values
recall = metrics_df['Recall@K'].values
ndcg = metrics_df['NDCG@K'].values

# creates visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(k_values, precision, marker='o', linewidth=2, markersize=8, label='Precision@K')
ax.plot(k_values, recall, marker='s', linewidth=2, markersize=8, label='Recall@K')
ax.plot(k_values, ndcg, marker='^', linewidth=2, markersize=8, label='NDCG@K')
ax.set_xlabel('K (Number of Recommendations)', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Recommendation System Performance Metrics', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(k_values)

plt.tight_layout()
plt.savefig('metrics_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'metrics_visualization.png'")
