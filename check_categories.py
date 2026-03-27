import pandas as pd
import os

categories = ['bit_ops', 'cipher', 'equation_numeric', 'equation_symbolic', 'gravity', 'numeral', 'unit_conv']

for cat in categories:
    df = pd.read_csv(f'data/cleaned_by_category/{cat}.csv')
    print(f'\n{"="*60}')
    print(f'Category: {cat} ({len(df)} samples)')
    print(f'{"="*60}')
    print(df['prompt'].iloc[0])
    print()
