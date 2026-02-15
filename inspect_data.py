import pandas as pd
import os

try:
    df = pd.read_csv('dataset/archive/BBCNews.csv')
    print("BBCNews.csv columns:", df.columns)
    print("BBCNews.csv shape:", df.shape)
    # Check if 'tags' or 'descr' can be mapped to categories
    # Maybe the 'tags' column contains the category as a substring?
    print("Sample tags:", df['tags'].head(10).tolist())
except Exception as e:
    print("Error reading BBCNews.csv:", e)

# Check archive (1)
try:
    files = os.listdir('dataset/archive (1)')
    print("Files in archive (1):", files)
except Exception as e:
    print("Error reading archive (1):", e)
