import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class NewsSentimentAnalyzer:
    """A class for analyzing news sentiment and its correlation with stock movements."""
    
    def __init__(self, news_df: pd.DataFrame, stock_data: Dict[str, pd.DataFrame]):
        """
        Initialize the NewsSentimentAnalyzer with news and stock data.
        
        Args:
            news_df (pd.DataFrame): DataFrame containing news data with columns:
                - headline: news headline text
                - date: publication date
                - stock: stock symbol
            stock_data (Dict[str, pd.DataFrame]): Dictionary of stock data DataFrames with columns:
                - Date: trading date
                - Close: closing price
                - Daily_Return: daily return
        """
        self.news_df = news_df.copy()
        self.stock_data = {symbol: df.copy() for symbol, df in stock_data.items()}
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Convert date columns to datetime
        self.news_df['date'] = pd.to_datetime(self.news_df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        for symbol in self.stock_data:
            self.stock_data[symbol]['Date'] = pd.to_datetime(self.stock_data[symbol]['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    def calculate_headline_sentiment(self) -> None:
        """Calculate sentiment scores for all headlines."""
        # Calculate sentiment scores
        self.news_df['sentiment_scores'] = self.news_df['headline'].apply(
            lambda x: self.sentiment_analyzer.polarity_scores(str(x))
        )
        
        # Extract compound score (overall sentiment)
        self.news_df['sentiment_score'] = self.news_df['sentiment_scores'].apply(
            lambda x: x['compound']
        )
        
        # Add sentiment categories
        self.news_df['sentiment_category'] = pd.cut(
            self.news_df['sentiment_score'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
    
    def get_daily_sentiment(self, symbol: str) -> pd.DataFrame:
        """
        Calculate daily sentiment scores for a specific stock.
        
        Args:
            symbol (str): Stock symbol to analyze
            
        Returns:
            pd.DataFrame: Daily sentiment scores with columns:
                - date: trading date
                - avg_sentiment: average sentiment score
                - sentiment_count: number of articles
                - sentiment_category: dominant sentiment category
        """
        # Filter news for the specific stock
        stock_news = self.news_df[self.news_df['stock'] == symbol].copy()
        
        # Group by date and calculate daily metrics
        daily_sentiment = stock_news.groupby('date').agg({
            'sentiment_score': ['mean', 'count'],
            'sentiment_category': lambda x: x.mode().iloc[0] if not x.empty else 'Neutral'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = ['date', 'avg_sentiment', 'article_count', 'sentiment_category']
        
        return daily_sentiment
    
    def analyze_sentiment_correlation(self, symbol: str, 
                                   lookback_days: int = 1,
                                   forward_days: int = 1) -> Dict:
        """
        Analyze correlation between news sentiment and stock returns.
        
        Args:
            symbol (str): Stock symbol to analyze
            lookback_days (int): Number of days to look back for sentiment impact
            forward_days (int): Number of days to look forward for returns
            
        Returns:
            Dict: Correlation analysis results including:
                - correlation: correlation coefficient
                - sentiment_impact: average returns by sentiment category
                - time_lag_analysis: correlation at different time lags
        """
        # Get daily sentiment
        daily_sentiment = self.get_daily_sentiment(symbol)
        
        # Get stock returns
        stock_df = self.stock_data[symbol].copy()
        stock_df['forward_return'] = stock_df['Close'].pct_change(forward_days).shift(-forward_days)
        
        # Merge sentiment and stock data
        merged_data = pd.merge(
            daily_sentiment,
            stock_df[['Date', 'forward_return']],
            left_on='date',
            right_on='Date',
            how='inner'
        )
        
        # Calculate correlation
        correlation = merged_data['avg_sentiment'].corr(merged_data['forward_return'])
        
        # Analyze sentiment impact on returns
        sentiment_impact = merged_data.groupby('sentiment_category')['forward_return'].agg([
            'mean', 'std', 'count'
        ]).round(4)
        
        # Time lag analysis
        time_lag_corr = {}
        for lag in range(-lookback_days, forward_days + 1):
            if lag != 0:  # Skip same-day correlation
                merged_data[f'return_lag_{lag}'] = merged_data['forward_return'].shift(-lag)
                time_lag_corr[lag] = merged_data['avg_sentiment'].corr(
                    merged_data[f'return_lag_{lag}']
                )
        
        return {
            'correlation': correlation,
            'sentiment_impact': sentiment_impact,
            'time_lag_analysis': time_lag_corr
        }
    
    def plot_sentiment_analysis(self, symbol: str, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> None:
        """
        Plot sentiment analysis results for a stock.
        
        Args:
            symbol (str): Stock symbol to analyze
            start_date (str, optional): Start date for analysis
            end_date (str, optional): End date for analysis
        """
        # Get daily sentiment
        daily_sentiment = self.get_daily_sentiment(symbol)
        
        # Get stock data
        stock_df = self.stock_data[symbol].copy()
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date, format='%Y-%m-%d', errors='coerce')
            daily_sentiment = daily_sentiment[daily_sentiment['date'] >= start_date]
            stock_df = stock_df[stock_df['Date'] >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce')
            daily_sentiment = daily_sentiment[daily_sentiment['date'] <= end_date]
            stock_df = stock_df[stock_df['Date'] <= end_date]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        
        # Plot stock price
        ax1.plot(stock_df['Date'], stock_df['Close'], label='Stock Price', color='blue')
        ax1.set_ylabel('Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plot sentiment on secondary y-axis
        ax1_twin = ax1.twinx()
        ax1_twin.bar(daily_sentiment['date'], daily_sentiment['avg_sentiment'],
                    alpha=0.3, label='Sentiment Score', color='green')
        ax1_twin.set_ylabel('Sentiment Score', color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')
        
        # Plot article count
        ax2.bar(daily_sentiment['date'], daily_sentiment['article_count'],
                label='Article Count', color='gray', alpha=0.7)
        ax2.set_ylabel('Number of Articles')
        ax2.set_xlabel('Date')
        
        # Add legends and title
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.legend(loc='upper left')
        
        plt.suptitle(f'Sentiment Analysis for {symbol}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def get_sentiment_summary(self, symbol: str) -> Dict:
        """
        Generate a summary of sentiment analysis for a stock.
        
        Args:
            symbol (str): Stock symbol to analyze
            
        Returns:
            Dict: Summary statistics including:
                - overall_sentiment: average sentiment score
                - total_articles: total number of articles
                - sentiment_distribution: distribution of sentiment categories
                - top_positive_news: most positive headlines
                - top_negative_news: most negative headlines
        """
        # Filter news for the specific stock
        stock_news = self.news_df[self.news_df['stock'] == symbol].copy()
        
        # Calculate overall sentiment
        overall_sentiment = stock_news['sentiment_score'].mean()
        
        # Get total articles count
        total_articles = len(stock_news)
        
        # Get sentiment distribution
        sentiment_dist = stock_news['sentiment_category'].value_counts(normalize=True)
        
        # Get top positive and negative headlines
        top_positive = stock_news.nlargest(5, 'sentiment_score')[['headline', 'sentiment_score', 'date']]
        top_negative = stock_news.nsmallest(5, 'sentiment_score')[['headline', 'sentiment_score', 'date']]
        
        return {
            'overall_sentiment': overall_sentiment,
            'total_articles': total_articles,
            'sentiment_distribution': sentiment_dist.to_dict(),
            'top_positive_news': top_positive.to_dict('records'),
            'top_negative_news': top_negative.to_dict('records')
        }
    
    def calculate_sentiment(self) -> pd.DataFrame:
        """
        Calculate sentiment scores for all headlines and return the dataframe with sentiment.
        
        Returns:
            pd.DataFrame: News dataframe with sentiment scores and categories added
        """
        self.calculate_headline_sentiment()
        return self.news_df
    
    def analyze_correlation(self, symbol: str, max_lag: int = 5) -> Dict:
        """
        Analyze correlation between sentiment and stock returns with time lag analysis.
        
        Args:
            symbol (str): Stock symbol to analyze
            max_lag (int): Maximum number of days to test for lag correlation
            
        Returns:
            Dict: Correlation analysis results
        """
        # Get daily sentiment
        daily_sentiment = self.get_daily_sentiment(symbol)
        
        # Get stock returns
        stock_df = self.stock_data[symbol].copy()
        stock_df['return'] = stock_df['Close'].pct_change()
        
        # Merge sentiment and stock data
        merged_data = pd.merge(
            daily_sentiment,
            stock_df[['Date', 'return']],
            left_on='date',
            right_on='Date',
            how='inner'
        )
        
        # Calculate overall correlation
        overall_correlation = merged_data['avg_sentiment'].corr(merged_data['return'])
        
        # Time lag analysis
        lag_correlations = {}
        best_lag = 0
        best_correlation = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag != 0:
                merged_data[f'return_lag_{lag}'] = merged_data['return'].shift(-lag)
                corr = merged_data['avg_sentiment'].corr(merged_data[f'return_lag_{lag}'])
                lag_correlations[lag] = corr
                
                if abs(corr) > abs(best_correlation):
                    best_correlation = corr
                    best_lag = lag
        
        return {
            'overall_correlation': overall_correlation,
            'best_lag': best_lag,
            'best_lag_correlation': best_correlation,
            'lag_correlations': lag_correlations
        }
    
    def plot_time_lag_correlation(self, symbol: str, max_lag: int = 5) -> None:
        """
        Plot time lag correlation analysis for a stock.
        
        Args:
            symbol (str): Stock symbol to analyze
            max_lag (int): Maximum number of days to test for lag correlation
        """
        correlation_data = self.analyze_correlation(symbol, max_lag)
        lag_correlations = correlation_data['lag_correlations']
        
        lags = list(lag_correlations.keys())
        correlations = list(lag_correlations.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(lags, correlations, alpha=0.7, color='skyblue')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Same Day')
        
        # Highlight best lag
        best_lag = correlation_data['best_lag']
        best_corr = correlation_data['best_lag_correlation']
        plt.bar(best_lag, best_corr, color='red', alpha=0.8, label=f'Best Lag: {best_lag} days')
        
        plt.xlabel('Time Lag (days)')
        plt.ylabel('Correlation Coefficient')
        plt.title(f'Time Lag Correlation Analysis for {symbol}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Best lag correlation: {best_corr:.3f} at {best_lag} days")
        print(f"Overall correlation: {correlation_data['overall_correlation']:.3f}") 