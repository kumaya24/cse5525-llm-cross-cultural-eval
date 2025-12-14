import pandas as pd
import numpy as np

INPUT_FILE = "datasets/combined_similarity_scores.csv"

df = pd.read_csv(INPUT_FILE)
score_df = df.iloc[:, 2:]

for col in score_df.columns:
    score_df.loc[:, col] = pd.to_numeric(score_df[col], errors='coerce')

all = pd.DataFrame({
    'Mean Score': score_df.mean(),
    'Std Dev': score_df.std(),
    'Mean |Score|': score_df.abs().mean()
})

all.index.name = 'FileName'
all = all.reset_index()

split_names = all['FileName'].str.split('_', n=1, expand=True)
all['Model'] = split_names[0]

prompt_part = split_names[1]

all['Prompt'] = prompt_part.str.replace('.csv', '', regex=False).str.replace(' ', '_', regex=False).str.lower()
mean_table = all.pivot(index='Model', columns='Prompt', values='Mean Score').round(4)

std_table = all.pivot(index='Model', columns='Prompt', values='Std Dev').round(4)

#  Mean Score
print(mean_table)
print("-" * 70)

#  Standard Deviation
print(std_table)
print("-" * 70)

best_mean = all.loc[all['Mean Score'].idxmax()]
best_std = all.loc[all['Std Dev'].idxmin()]

print(
    f"1. Highest Mean Score: "
    f"'{best_mean['Model']}_{best_mean['Prompt']}' "
)

print(
    f"2. Lowest Standard Deviation: "
    f"'{best_std['Model']}_{best_std['Prompt']}' "
)
