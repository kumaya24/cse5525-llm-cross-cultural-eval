import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df_cs = pd.read_csv('datasets/combined_similarity_scores.csv')
df_csi = pd.read_csv('datasets/combined_psr_scores.csv')

def process_and_anova(df, score_col_name):
    id_vars = df.columns[:2]
    value_vars = df.columns[2:]
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Condition', value_name='Score')

    def parse_condition(condition):
        clean_name = condition.replace('.csv', '').replace('datasets/', '')
        
        if clean_name.lower().startswith('gpt'):
            model = 'GPT'
            rest = clean_name[4:]
        elif clean_name.lower().startswith('llama'):
            model = 'Llama'
            rest = clean_name[6:]
        elif clean_name.lower().startswith('gemini'):
            model = 'Gemini'
            rest = clean_name[7:] if clean_name.startswith('Gemini_') else clean_name.replace('Gemini', '').strip('_')
        else:
            model = 'Unknown'
            rest = clean_name
            
        rest_lower = rest.lower().replace('_', ' ').replace('-', ' ')
        
        if 'culture aware' in rest_lower:
            prompt = 'Culture-aware'
        elif 'title and synopsis' in rest_lower:
            prompt = 'Title & Synopsis'
        elif 'title only' in rest_lower:
            prompt = 'Title-only'
        else:
            prompt = 'Unknown'
            
        return pd.Series([model, prompt])

    df_long[['Model', 'Prompt']] = df_long['Condition'].apply(parse_condition)
    
    df_long = df_long.dropna(subset=['Score'])
    
    df_long['Model'] = pd.Categorical(df_long['Model'])
    df_long['Prompt'] = pd.Categorical(df_long['Prompt'])
    
    lm = ols('Score ~ C(Model) * C(Prompt)', data=df_long).fit()
    anova_table = sm.stats.anova_lm(lm, typ=2)
    anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
    
    return anova_table

# Compute ANOVAs
anova_cs = process_and_anova(df_cs, 'CS_Score')
anova_csi = process_and_anova(df_csi, 'CSI_Score')

results = {
    'Effect': ['Model', 'Prompt Condition', 'Model × Prompt Condition'],
    'CSI-Match p-value': [
        f"{anova_csi.loc['C(Model)', 'PR(>F)']:.2e}",
        f"{anova_csi.loc['C(Prompt)', 'PR(>F)']:.2e}",
        f"{anova_csi.loc['C(Model):C(Prompt)', 'PR(>F)']:.2e}"
    ],
    'Cosine Similarity p-value': [
        f"{anova_cs.loc['C(Model)', 'PR(>F)']:.2e}",
        f"{anova_cs.loc['C(Prompt)', 'PR(>F)']:.2e}",
        f"{anova_cs.loc['C(Model):C(Prompt)', 'PR(>F)']:.2e}"
    ]
}

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

