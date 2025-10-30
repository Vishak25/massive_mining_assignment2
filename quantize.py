q = df['SA_score'].quantile(0.8)   # top 20 % ≈ “harder” molecules
df['imb_label'] = (df['SA_score'] > q).astype(int)
print(df['imb_label'].value_counts(normalize=True))q