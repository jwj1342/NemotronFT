import pandas as pd
import re

df = pd.read_csv('data/train.csv')

def extract_core_question(prompt):
    # 只去除"In Alice's Wonderland,"开头
    # 保留问题类型说明、示例和问题
    if prompt.startswith("In Alice's Wonderland, "):
        return prompt[len("In Alice's Wonderland, "):]
    return prompt

df['prompt_cleaned'] = df['prompt'].apply(extract_core_question)

# 查看清洗效果
print("原始prompt示例:")
print(df['prompt'].iloc[0][:200])
print("\n清洗后prompt示例:")
print(df['prompt_cleaned'].iloc[0][:200])

# 保存清洗后的数据
df_cleaned = df[['id', 'prompt_cleaned', 'answer']].copy()
df_cleaned.columns = ['id', 'prompt', 'answer']
df_cleaned.to_csv('data/train_cleaned.csv', index=False)
print(f"\n已保存 {len(df_cleaned)} 条清洗后的数据到 train_cleaned.csv")
