import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = str(Path(os.getcwd()).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our utilities
from src.eda_utils import (
    display_basic_info,
    display_summary_stats,
    count_articles_per_publisher,
    analyze_publication_dates,
    publication_time_distribution,
    extract_unique_publisher_domains,
    simple_keyword_count
)

def time_operation(name, func):
    """Helper function to time operations"""
    print(f"\n=== Running: {name} ===")
    start = time.time()
    result = func()
    end = time.time()
    print(f"✓ Completed in {end - start:.2f} seconds")
    return result

def main():
    os.makedirs('../data', exist_ok=True)

    # Load only needed columns to save memory
    needed_columns = ['date', 'publisher', 'headline']
    
    print("Loading data...")
    start = time.time()
    df = pd.read_csv('data/raw_analyst_ratings.csv', usecols=needed_columns)
    end = time.time()
    print(f"✓ Data loaded in {end - start:.2f} seconds")
    print(f"✓ Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")

    # Run basic analysis with timing
    time_operation("Basic Info", lambda: display_basic_info(df))
    time_operation("Summary Stats", lambda: display_summary_stats(df))

    # Publisher analysis
    publisher_counts = time_operation("Publisher Analysis", 
        lambda: count_articles_per_publisher(df))
    
    # Plot top 10 publishers
    plt.figure(figsize=(12, 6))
    publisher_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Publishers by Article Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../data/top_publishers.png')
    plt.close()

    # Time analysis
    pub_dates = time_operation("Publication Date Analysis", 
        lambda: analyze_publication_dates(df))
    
    # Plot publication dates over time
    plt.figure(figsize=(15, 6))
    pub_dates.plot(kind='line')
    plt.title('Articles Published Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig('../data/publication_dates.png')
    plt.close()

    # Time distribution
    time_dist = time_operation("Publication Time Distribution", 
        lambda: publication_time_distribution(df))
    
    # Plot publication time distribution
    plt.figure(figsize=(12, 6))
    time_dist.plot(kind='bar')
    plt.title('Distribution of Publication Times (Hour of Day)')
    plt.xlabel('Hour')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig('../data/time_distribution.png')
    plt.close()

    # Keyword analysis
    keywords = ["FDA approval", "price target", "earnings", "merger", "acquisition", "downgrade", "upgrade"]
    keyword_counts = time_operation("Keyword Analysis", 
        lambda: simple_keyword_count(df, 'headline', keywords))
    
    # Plot keyword counts
    plt.figure(figsize=(12, 6))
    plt.bar(keyword_counts.keys(), keyword_counts.values())
    plt.title('Frequency of Keywords in Headlines')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../data/keyword_counts.png')
    plt.close()

if __name__ == "__main__":
    main() 