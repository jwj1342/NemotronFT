import pandas as pd
import os

df = pd.read_csv('data/train_cleaned.csv')

import pandas as pd
import os

df = pd.read_csv('data/train_cleaned.csv')

# Extract category from cleaned prompt
def categorize(prompt):
    if 'Bit Manipulation:' in prompt:
        return 'bit_ops'
    elif 'Text Cipher:' in prompt:
        return 'cipher'
    elif 'Numeral System:' in prompt:
        return 'numeral'
    elif 'Unit Conversion:' in prompt:
        return 'unit_conv'
    elif 'Gravity Constant:' in prompt:
        return 'gravity'
    elif 'Equation Numeric:' in prompt:
        return 'equation_numeric'
    elif 'Equation Symbolic:' in prompt:
        return 'equation_symbolic'
    else:
        return 'other'

df['category'] = df['prompt'].apply(categorize)

# 创建目录
os.makedirs('data/cleaned_by_category', exist_ok=True)

# 按类别保存
for category in df['category'].unique():
    cat_df = df[df['category'] == category][['id', 'prompt', 'answer']]
    cat_df.to_csv(f'data/cleaned_by_category/{category}.csv', index=False)
    print(f"{category}: {len(cat_df)} 条")

# 从每类采样136条
sampled_dfs = []
for category in df['category'].unique():
    cat_df = df[df['category'] == category]
    sample_size = min(136, len(cat_df))
    sampled = cat_df.sample(n=sample_size, random_state=42)
    sampled_dfs.append(sampled)
    print(f"采样 {category}: {sample_size} 条")

# 合并并保存
distill_df = pd.concat(sampled_dfs)[['id', 'prompt', 'answer']]
distill_df.to_csv('data/cleaned_data_for_distillation.csv', index=False)
print(f"\n总计采样: {len(distill_df)} 条")

