import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from corr import spearman_correlation

# benchmark
bench = "mt-bench"  # alpacaeval, mt-bench, arena-hard, wildbench
week_timestamp = "240416"

# Paths
json_path = os.path.expanduser(f'config/rankings/{bench}_test.json')
tsv_dir = os.path.expanduser(f'data/ranking/{bench}/{week_timestamp}_meeting')

# Load Elo-based model ranking
with open(json_path, 'r') as f:
    model_data = json.load(f)

# Map normalized model names to their Elo rating for sorting
elo_rating = {}
model_name_mapping = {}

for entry in model_data:
    norm_name = entry['name'].strip().lower().replace('@together', '').replace('-hf', '')
    elo_rating[norm_name] = entry['rating']  # {"gpt-4o-mini": 1120, "llama-3-70B-instruct":1320, ...}
    model_name_mapping[norm_name] = entry['name']

# Sort models by Elo rating (descending)
sorted_models = sorted(elo_rating, key=lambda k: -elo_rating[k])
print(f'model_rankings_based_on_ELO_Ratings:{sorted_models}')
baseline_ranking = {}
for i in range(len(sorted_models)):
    baseline_ranking[sorted_models[i]] = i+1
print(f'baseline ranking dict: {baseline_ranking}')

# Collect scores from each TSV file
linescores = {}
ranking_correlations = {}

for filename in os.listdir(tsv_dir):
    if filename.endswith('.tsv'):
        path = os.path.join(tsv_dir, filename)
        df = pd.read_csv(path, sep='\t')
        scores = {}
        llm_judge_ranking = {}
        for _, row in df.iterrows():
            model_name = str(row['model_test']).strip().lower().replace('@together', '').replace('-hf', '')
            scores[model_name] = float(row[df.columns[2]])
            llm_judge_ranking[model_name] = row['Rank']
        linescores[filename] = scores
        ranking_correlations[filename] = spearman_correlation(dict1=baseline_ranking, dict2=llm_judge_ranking)
# linescores['Elo-Rating'] = elo_rating
# ranking_correlations['Elo-Rating'] = spearman_correlation(dict1=baseline_ranking, dict2=baseline_ranking)
print(f'calculated line scores: {linescores}')
print(f'calculated ranking correlations: {ranking_correlations}')

# Plotting
plt.figure(figsize=(12, 9))  # Slightly taller for legend space
x_labels = [model_name_mapping[name] for name in sorted_models]
x_pos = range(len(sorted_models))

# Store line info for custom legend
legend_lines = []
legend_labels = []

colors = plt.cm.tab10.colors
color_map = {}

for i, (filename, score_dict) in enumerate(linescores.items()):
    y_vals = [score_dict.get(name, None) for name in sorted_models]
    color = colors[i%len(colors)]
    color_map[filename] = color
    line, = plt.plot(x_pos, y_vals, marker='o', label=filename[:-4], color=color)
    legend_lines.append(line)
    corr = ranking_correlations[filename]
    legend_labels.append(f"{filename[:-4]} -> Ranking Correlation: {corr:.3f}")

# Main Chart Formatting
plt.xticks(x_pos, x_labels, rotation=45, ha='right')
plt.xlabel('M')
plt.xlabel('LLM Rankings (by Elo Ratings)', labelpad=20)
plt.ylabel('Judged Scores (by Judge LLMs)')
plt.title('Scores & Rankings')
plt.grid(True)

# Custom Legend Below Chart
for i, (line, label) in enumerate(zip(legend_lines, legend_labels)):
    plt.text(
        0.25, -0.5 -i * 0.045,  # Adjust Y-step and starting point as needed
        label,
        color=line.get_color(),
        transform=plt.gca().transAxes,
        fontsize=9,
        fontweight='bold'
    )

plt.tight_layout()
plt.subplots_adjust(bottom=0.45)  # Leave space for custom legend
plt.savefig(os.path.expanduser(f'analysis/figure_3/{week_timestamp}_meeting/{bench}.png'), dpi=300, bbox_inches='tight')
plt.show()

"""for filename, score_dict in linescores.items():
    y_vals = [score_dict.get(name, None) for name in sorted_models]
    plt.plot(x_pos, y_vals, marker='o', label=filename)

plt.xticks(x_pos, x_labels, rotation=45, ha='right')
plt.xlabel('LLM Rankings (by Elo Ratings)', labelpad=20)
plt.ylabel('Judged Scores (by Judge LLMs)')
plt.title('Scores & Rankings')
plt.grid(True)

# Clean, vertically stacked legend below the chart
plt.legend(
    title='Judge LLM (Checklist Generator LLM)',
    loc='upper center',
    bbox_to_anchor=(0.5, -0.3),
    ncol=1,
    frameon=False
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.45)  # Space for vertical legend and labels

# Save the figure or show it
plt.savefig(os.path.expanduser(f'analysis/figure_3/{bench}.png'), dpi=300, bbox_inches='tight')
plt.show()"""