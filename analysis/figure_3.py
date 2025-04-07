import os
import json
import matplotlib.pyplot as plt
import pandas as pd

# benchmark
bench = "mt-bench"  # alpacaeval, mt-bench, arena-hard, wildbench
# Paths
json_path = os.path.expanduser(f'config/rankings/{bench}_test.json')
tsv_dir = os.path.expanduser(f'data/ranking/{bench}')

# Load Elo-based model ranking
with open(json_path, 'r') as f:
    model_data = json.load(f)

# Map normalized model names to their Elo rating for sorting
elo_ranking = {}
model_name_mapping = {}

for entry in model_data:
    norm_name = entry['name'].strip().lower().replace('@together', '').replace('-hf', '')
    elo_ranking[norm_name] = entry['rating']
    model_name_mapping[norm_name] = entry['name']

# Sort models by Elo rating (descending)
sorted_models = sorted(elo_ranking, key=lambda k: -elo_ranking[k])

# Collect scores from each TSV file
linescores = {}

for filename in os.listdir(tsv_dir):
    if filename.endswith('.tsv'):
        path = os.path.join(tsv_dir, filename)
        df = pd.read_csv(path, sep='\t')
        scores = {}
        for _, row in df.iterrows():
            model_name = str(row['model_test']).strip().lower().replace('@together', '').replace('-hf', '')
            scores[model_name] = float(row[df.columns[2]])
        linescores[filename] = scores

# Plotting
plt.figure(figsize=(12, 7))  # Slightly taller for legend space

x_labels = [model_name_mapping[name] for name in sorted_models]
x_pos = range(len(sorted_models))

for filename, score_dict in linescores.items():
    y_vals = [score_dict.get(name, None) for name in sorted_models]
    plt.plot(x_pos, y_vals, marker='o', label=filename)

plt.xticks(x_pos, x_labels, rotation=45, ha='right')
plt.xlabel('Models (ranked by Elo)', labelpad=20)
plt.ylabel('Score from TSV files')
plt.title('Model Scores by Elo Ranking')
plt.grid(True)

# Clean, vertically stacked legend below the chart
plt.legend(
    title='TSV Files',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.3),
    ncol=1,
    frameon=False
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.45)  # Space for vertical legend and labels

# Save the figure or show it
plt.savefig(os.path.expanduser(f'analysis/figure_3/{bench}.png'), dpi=300, bbox_inches='tight')
plt.show()