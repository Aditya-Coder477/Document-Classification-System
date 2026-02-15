import pandas as pd

try:
    df = pd.read_csv('dataset/archive/BBCNews.csv')
    print("Total rows:", len(df))
    
    categories = {
        'sport': 'Sports',
        'politics': 'Politics',
        'tech': 'Technology',
        'business': 'Business',
        'entertainment': 'Entertainment'
    }
    
    match_counts = {k: 0 for k in categories.values()}
    unmatched = 0
    
    for _, row in df.iterrows():
        tags = str(row['tags']).lower()
        found = False
        for key, val in categories.items():
            if key in tags:
                match_counts[val] += 1
                found = True
                break # Count only once per row
        if not found:
            unmatched += 1
            
    print("Category text matches:", match_counts)
    print("Unmatched rows:", unmatched)

except Exception as e:
    print(e)
