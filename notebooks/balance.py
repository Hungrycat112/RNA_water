import pandas as pd

df = pd.read_csv("test_combine_2.csv")
df = df[~(df == 0).all(axis=1)]

# Shuffle the original DataFrame
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into positive and negative after shuffle
df_pos = df_shuffled[df_shuffled['label'] == 1]
df_neg = df_shuffled[df_shuffled['label'] == 0]

# Determine the smaller group size (in case you want to make it flexible)
n_samples = len(df_pos)

# Sample equal amounts
df_neg_sampled = df_neg.sample(n=n_samples, random_state=42)

# Combine and reshuffle
df_balanced = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Check class balance
print(df_balanced['label'].value_counts())

# Save the balanced DataFrame to a CSV file
df_balanced.to_csv('balanced_dataset.csv', index=False)

