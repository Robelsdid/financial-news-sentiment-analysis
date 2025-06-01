import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Functions for quick overview and printing summaries ---

def display_basic_info(df):
    """
    Prints the basic info about the DataFrame:
    number of rows, columns, and column data types.
    """
    print("=== Basic DataFrame Info ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumn data types:")
    print(df.dtypes)
    print("\n")

def display_summary_stats(df):
    """
    Prints descriptive statistics for numerical columns
    and shows count of missing values.
    """
    print("=== Summary Statistics ===")
    print("Descriptive statistics for numerical columns:")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\n")

# --- Text Analysis Functions ---

def describe_text_length(df, column):
    """
    Returns descriptive statistics for the length of text in a specified column.
    """
    lengths = df[column].astype(str).apply(len)
    return lengths.describe()

def extract_common_words(texts, n=20, min_length=3):
    """
    Extracts the most common words from a series of texts.
    
    Args:
        texts: Series of text strings
        n: Number of top words to return
        min_length: Minimum word length to consider
    """
    words = []
    stop_words = set(stopwords.words('english'))
    
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        words.extend([
            word for word in tokens 
            if word.isalnum() and 
            len(word) >= min_length and 
            word not in stop_words
        ])
    
    return Counter(words).most_common(n)

def simple_keyword_count(df, text_column, keywords):
    """
    Counts occurrences of specified keywords in a text column.
    """
    keyword_counts = Counter()
    for text in df[text_column].dropna().astype(str):
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                keyword_counts[kw] += 1
    return dict(keyword_counts)

# --- Time Series Analysis Functions ---

def analyze_publication_dates(df, date_col='date'):
    """
    Analyze distribution of articles over time.
    Returns sorted count of articles per date.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    return df.groupby(df[date_col].dt.date).size().sort_index()

def publication_time_distribution(df, date_col='date'):
    """
    Analyzes the distribution of publication times throughout the day.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    return df[date_col].dt.hour.value_counts().sort_index()

def analyze_weekly_pattern(df, date_col='date'):
    """
    Analyzes the distribution of articles by day of the week.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    return df.groupby(df[date_col].dt.day_name()).size()

# --- Publisher Analysis Functions ---

def count_articles_per_publisher(df, publisher_col='publisher'):
    """
    Returns a Series with counts of articles per publisher.
    """
    return df[publisher_col].value_counts()

def extract_unique_publisher_domains(df, publisher_col='publisher'):
    """
    Extract unique email domains if publisher names are emails.
    """
    emails = df[publisher_col].dropna().astype(str)
    domains = emails[emails.str.contains('@')].apply(lambda x: x.split('@')[-1].lower())
    return domains.value_counts()

def analyze_publisher_activity_by_hour(df, date_col='date', publisher_col='publisher'):
    """
    Analyzes publisher activity patterns throughout the day.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    hourly_stats = df.groupby(df[date_col].dt.hour).agg({
        publisher_col: ['count', 'nunique']
    })
    hourly_stats.columns = ['article_count', 'unique_publishers']
    return hourly_stats

# --- Helper Functions ---

def check_missing_values(df):
    """Returns the count of missing values per column."""
    return df.isnull().sum()

def get_column_info(df):
    """Returns DataFrame info including dtypes and null counts."""
    return pd.DataFrame({
        "dtype": df.dtypes,
        "non_nulls": df.notnull().sum(),
        "nulls": df.isnull().sum()
    })

def count_unique(df, column):
    """Returns frequency counts of unique values in a column."""
    return df[column].value_counts()
