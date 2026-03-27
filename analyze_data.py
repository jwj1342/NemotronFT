import pandas as pd
import re
import sys

# 同时输出到终端和文件
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

f = open('data_analyze.md', 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)

df = pd.read_csv('data/train.csv')

# 详细题型分类（合并细分）
def classify_detailed_type(prompt):
    p = prompt.lower()
    if 'bit manipulation' in p:
        return 'bit_manipulation'
    elif 'encryption' in p or 'decrypt' in p:
        return 'encryption'
    elif 'numeral system' in p:
        return 'numeral_conversion'
    elif 'unit conversion' in p:
        return 'unit_conversion'
    elif 'gravitational constant' in p:
        return 'physics'
    elif any(x in p for x in ['mathematical', 'equation', 'formula']):
        return 'mathematics'
    else:
        return 'other'

df['type'] = df['prompt'].apply(classify_detailed_type)
df['prompt_len'] = df['prompt'].str.len()
df['answer_len'] = df['answer'].astype(str).str.len()

# 1. 详细题型分布
print("="*60)
print("1. 详细题型分布")
print("="*60)
type_counts = df['type'].value_counts()
print(type_counts)
print(f"\n总计: {len(df)} 条\n")

# 2. 各题型提示词长度统计、答案长度统计
print("="*60)
print("2. 各题型提示词长度统计")
print("="*60)
print(df.groupby('type')['prompt_len'].describe())

print("\n" + "="*60)
print("2. 各题型答案长度统计")
print("="*60)
print(df.groupby('type')['answer_len'].describe())

# 3. 题型-答案格式交叉分布表
def answer_format(ans):
    ans = str(ans)
    if re.match(r'^[01]+$', ans):
        return 'binary'
    elif re.match(r'^[0-9]+$', ans):
        return 'numeric'
    elif re.match(r'^[IVXLCDM]+$', ans):
        return 'roman'
    elif re.match(r'^[0-9.]+$', ans):
        return 'decimal'
    else:
        return 'text'

df['answer_format'] = df['answer'].apply(answer_format)

print("\n" + "="*60)
print("3. 题型-答案格式交叉分布表")
print("="*60)
print(pd.crosstab(df['type'], df['answer_format']))

# 4. 常见提示词开头及常见提示词开头-题型交叉分布表
df['prompt_start'] = df['prompt'].str[:80]

print("\n" + "="*60)
print("4. 常见提示词开头")
print("="*60)
start_counts = df['prompt_start'].value_counts().head(10)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
label_map = {}
for i, (start, count) in enumerate(start_counts.items()):
    label = labels[i]
    label_map[start] = label
    print(f"[{label}] {count:4d}条: {start}...")

print("\n" + "="*60)
print("4. 常见提示词开头-题型交叉分布表")
print("="*60)
top_starts = df['prompt_start'].value_counts().head(10).index
df_filtered = df[df['prompt_start'].isin(top_starts)].copy()
df_filtered['label'] = df_filtered['prompt_start'].map(label_map)
print(pd.crosstab(df_filtered['label'], df_filtered['type']))

sys.stdout = original_stdout
f.close()
print("\n分析结果已保存至 data_analyze.md")
