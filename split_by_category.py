import pandas as pd
import os

df = pd.read_csv('data/train_cleaned.csv')

# 分类函数
def categorize(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p or 'bit' in p[:100]:
        return 'bit_ops'
    elif 'encryption' in p or 'decrypt' in p or 'cipher' in p:
        return 'cipher'
    elif 'numeral system' in p or 'roman' in p:
        return 'numeral'
    elif 'unit conversion' in p:
        return 'unit_conv'
    elif 'gravitational' in p or 'gravity' in p:
        return 'gravity'
    elif 'transformation rules is applied to equations' in p:
        # 检查示例中是否主要是数字
        lines = prompt.split('\n')
        has_numbers = any(any(c.isdigit() for c in line) for line in lines[2:5] if line.strip())
        has_symbols = any(any(c in '`!@#$%^&*(){}[]<>?~' for c in line) for line in lines[2:5] if line.strip())
        if has_numbers and not has_symbols:
            return 'equation_numeric'
        else:
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

