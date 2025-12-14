import csv
import os
import collections
from fuzzywuzzy import fuzz 

FILE_NAMES = [
    'datasets/Gemini_Culture-aware.csv', 'datasets/Gemini_Title_and_Synopsis.csv', 'datasets/Gemini_Title-only.csv', 
    'datasets/gpt_Culture-aware.csv', 'datasets/gpt_Title_and_Synopsis.csv',
    'datasets/gpt_Title-only.csv', 'datasets/Llama_Culture-aware.csv', 
    'datasets/Llama_Title_and_Synopsis.csv', 'datasets/Llama_Title-only.csv'
]
OUTPUT_FILE = "datasets/combined_psr_scores.csv"

COLUMN_INDEX_REFERENCE = 0
COLUMN_INDEX_GENERATED = 3
KEY_COLUMN_INDICES = [0, 1] 
CSI_TERMS = [] 

def calculate_psr_for_row(reference, generated, csi_terms):
    if not reference or not generated:
        return 0.0

    comp = csi_terms if csi_terms else [reference]
    max_psr = 0.0

    for target in comp:
        if not target: 
            continue
        psr_score = fuzz.partial_ratio(str(target).lower(), str(generated).lower()) / 100.0

    return max(max_psr, psr_score)

agg_scores = collections.defaultdict(lambda: {name: '' for name in FILE_NAMES})
total = 0

for file_name in FILE_NAMES:
    with open(file_name, 'r', newline='', encoding='utf-8-sig') as f_in:
        reader = csv.reader(f_in)
        next(reader)

        file_rows = list(reader)
        total += len(file_rows)

        for row in file_rows:
            key = (row[KEY_COLUMN_INDICES[0]], row[KEY_COLUMN_INDICES[1]])
            reference = row[COLUMN_INDEX_REFERENCE] # Column 1
            generated = row[COLUMN_INDEX_GENERATED] # Column 4

            psr_score = calculate_psr_for_row(reference, generated, CSI_TERMS)
            agg_scores[key][file_name] = f"{psr_score:.4f}"

output_header = [
    'Column 1', 
    'Column 2',
] + FILE_NAMES

output_rows = []
for key, scores in agg_scores.items():
    row = list(key)
    for file_name in FILE_NAMES:
        row.append(scores.get(file_name, '')) 

    output_rows.append(row)

if output_rows:
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(output_header)
        writer.writerows(output_rows)
else:
    print("No data.")