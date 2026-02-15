import pandas as pd

try:
    df = pd.read_csv('dataset/archive/BBCNews.csv')
    print("Columns:", df.columns)
    print("First 5 tags:\n", df['tags'].head())
    
    # Check for keywords in tags
    keywords = ['sport', 'politics', 'tech', 'business', 'entertainment']
    for k in keywords:
        count = df['tags'].str.contains(k, case=False).sum()
        print(f"Contains '{k}': {count}")
        
    # Check if we can form categories from tags
    # Maybe the 'tags' are the categories?
    # Let's see value counts of tags if possible, or just sample more
    print("Top 10 most common tags:\n", df['tags'].value_counts().head(10))

except Exception as e:
    print(e)
