import pandas as pd
import numpy as np

INPUT_FILE = "datasets/combined_psr_scores.csv" 
df = pd.read_csv(INPUT_FILE)

score_df = df.iloc[:, 2:]

for col in score_df.columns:
    score_df.loc[:, col] = pd.to_numeric(score_df[col], errors='coerce')

all = pd.DataFrame({
    'Mean PSR': score_df.mean(),
    'Std Dev': score_df.std(),
})

all.index.name = 'FileName'
all = all.reset_index()

split_names = all['FileName'].str.split('_', n=1, expand=True)
all['Model'] = split_names[0]
all['Prompt'] = split_names[1].str.replace('.csv', '', regex=False).str.replace(' ', '_', regex=False).str.lower()

mean_table = all.pivot(index='Model', columns='Prompt', values='Mean PSR').round(4)
std_table = all.pivot(index='Model', columns='Prompt', values='Std Dev').round(4)

# Mean 
print(mean_table)
print("-" * 70)

# Standard Deviation
print(std_table)
print("-" * 70)

avg_mean = mean_table.mean(axis=1)
avg_std = std_table.mean(axis=1)

top_mean_model = avg_mean.idxmax()
top_mean_value = avg_mean.max()

lowest_std_model = avg_std.idxmin()
lowest_std_value = avg_std.min()

print(f"The model with highest mean is '{top_mean_model}'")
print(f"The model with lowest Standard Deviatioonsistent is '{lowest_std_model}'")
