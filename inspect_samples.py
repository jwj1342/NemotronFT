import pandas as pd

df = pd.read_csv('data/train.csv')
print('列名:', df.columns.tolist())
print(f'\n总数据量: {len(df)}')

print('\n=== 样本1 (位运算) ===')
print(df['prompt'].iloc[0])

print('\n=== 样本2 (加密) ===')
print(df['prompt'].iloc[100])

print('\n=== 样本3 (单位转换) ===')
print(df['prompt'].iloc[200])

print('\n=== 样本4 ===')
print(df['prompt'].iloc[300])

print('\n=== 样本5 ===')
print(df['prompt'].iloc[400])
