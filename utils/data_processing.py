import pandas as pd

WORD_LIST = "data/CSW24defs.csv"
REGEX_PATTERN = r'India|Hindu|\(Hindi\)|\(Tamil\)|\(Telugu\)|\(Bengali\)|\(Marathi\)|\(Gujarati\)|\(Kannada\)|\(Malayalam\)|\(Punjabi\)|\(Odia\)|\(Assamese\)|\(Urdu\)|\(Sanskrit\)'

with open(WORD_LIST, 'r') as f:
    lines = f.readlines()

data = [line.strip().split(',', 1) for line in lines]
df = pd.DataFrame(data, columns=['word', 'definition'])

print(df.shape)

# Filter Indian and non-Indian words
indian_mask = df['definition'].str.contains(REGEX_PATTERN, regex=True, case=False)
indian_origin_words = df[indian_mask]
non_indian_words = df[~indian_mask]

print("Indian origin words:", len(indian_origin_words))
print("Non-Indian words:", len(non_indian_words))

# Write to CSV
indian_origin_words.to_csv('data/indian_words.csv', index=False)
non_indian_words.to_csv('data/non_indian_words.csv', index=False)