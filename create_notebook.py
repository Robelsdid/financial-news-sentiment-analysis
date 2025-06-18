import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Financial News Sentiment Analysis\n",
                "\n",
                "This notebook analyzes the correlation between financial news sentiment and stock price movements."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from datetime import datetime\n",
                "import sys\n",
                "import os\n",
                "\n",
                "# Add src directory to path\n",
                "sys.path.append(os.path.abspath('..'))\n",
                "\n",
                "from src.sentiment_analysis import NewsSentimentAnalyzer\n",
                "from src.stock_data_loader import load_stock_data\n",
                "\n",
                "# Set plot style\n",
                "plt.style.use('default')\n",
                "sns.set_palette('husl')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Load Data\n",
                "\n",
                "First, we'll load both the news data and stock data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load news data\n",
                "news_data = pd.read_csv('../data/raw_analyst_ratings.csv')\n",
                "print(f\"News data shape: {news_data.shape}\")\n",
                "news_data.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load stock data for all symbols\n",
                "symbols = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']\n",
                "stock_data = load_stock_data(symbols, data_dir='../data')\n",
                "\n",
                "for symbol in symbols:\n",
                "    if symbol in stock_data:\n",
                "        print(f\"{symbol} data range: {stock_data[symbol].index.min()} to {stock_data[symbol].index.max()}\")\n",
                "    else:\n",
                "        print(f\"No data found for {symbol}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Initialize Sentiment Analyzer and Calculate Sentiment Scores\n",
                "\n",
                "We'll create an instance of NewsSentimentAnalyzer and calculate sentiment scores for all headlines."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize analyzer\n",
                "analyzer = NewsSentimentAnalyzer(news_data, stock_data)\n",
                "\n",
                "# Calculate sentiment scores\n",
                "news_with_sentiment = analyzer.calculate_sentiment()\n",
                "\n",
                "# Display sentiment distribution\n",
                "plt.figure(figsize=(10, 6))\n",
                "news_with_sentiment['sentiment_category'].value_counts().plot(kind='bar')\n",
                "plt.title('Distribution of News Sentiment Categories')\n",
                "plt.xlabel('Sentiment Category')\n",
                "plt.ylabel('Count')\n",
                "plt.xticks(rotation=45)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Analyze Each Stock\n",
                "\n",
                "For each stock, we'll:\n",
                "1. Generate a sentiment summary\n",
                "2. Analyze correlation between sentiment and returns\n",
                "3. Plot sentiment analysis results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for symbol in symbols:\n",
                "    if symbol not in stock_data:\n",
                "        print(f\"Skipping {symbol} - no stock data available\")\n",
                "        continue\n",
                "        \n",
                "    print(f\"\\n{'='*80}\\nAnalyzing {symbol}\\n{'='*80}\")\n",
                "    \n",
                "    try:\n",
                "        # Get sentiment summary\n",
                "        summary = analyzer.get_sentiment_summary(symbol)\n",
                "        print(\"\\nSentiment Summary:\")\n",
                "        print(f\"Overall Sentiment: {summary['overall_sentiment']:.2f}\")\n",
                "        print(f\"Total Articles: {summary['total_articles']}\")\n",
                "        print(\"\\nSentiment Distribution:\")\n",
                "        print(summary['sentiment_distribution'])\n",
                "        \n",
                "        # Analyze correlation\n",
                "        correlation = analyzer.analyze_correlation(symbol)\n",
                "        print(\"\\nCorrelation Analysis:\")\n",
                "        print(f\"Overall Correlation: {correlation['overall_correlation']:.3f}\")\n",
                "        print(f\"Best Lag: {correlation['best_lag']} days\")\n",
                "        print(f\"Best Lag Correlation: {correlation['best_lag_correlation']:.3f}\")\n",
                "        \n",
                "        # Plot sentiment analysis\n",
                "        analyzer.plot_sentiment_analysis(symbol)\n",
                "        \n",
                "        # Plot time lag correlation\n",
                "        analyzer.plot_time_lag_correlation(symbol)\n",
                "        \n",
                "    except Exception as e:\n",
                "        print(f\"Error analyzing {symbol}: {str(e)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cross-Stock Analysis\n",
                "\n",
                "Compare sentiment and correlation metrics across all stocks."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Collect metrics for all stocks\n",
                "metrics = []\n",
                "for symbol in symbols:\n",
                "    if symbol not in stock_data:\n",
                "        continue\n",
                "        \n",
                "    try:\n",
                "        summary = analyzer.get_sentiment_summary(symbol)\n",
                "        correlation = analyzer.analyze_correlation(symbol)\n",
                "        \n",
                "        metrics.append({\n",
                "            'Symbol': symbol,\n",
                "            'Overall Sentiment': summary['overall_sentiment'],\n",
                "            'Total Articles': summary['total_articles'],\n",
                "            'Positive %': summary['sentiment_distribution']['positive'],\n",
                "            'Neutral %': summary['sentiment_distribution']['neutral'],\n",
                "            'Negative %': summary['sentiment_distribution']['negative'],\n",
                "            'Correlation': correlation['overall_correlation'],\n",
                "            'Best Lag': correlation['best_lag'],\n",
                "            'Best Lag Correlation': correlation['best_lag_correlation']\n",
                "        })\n",
                "    except Exception as e:\n",
                "        print(f\"Error collecting metrics for {symbol}: {str(e)}\")\n",
                "\n",
                "if metrics:\n",
                "    metrics_df = pd.DataFrame(metrics)\n",
                "    metrics_df.set_index('Symbol', inplace=True)\n",
                "\n",
                "    # Plot comparison metrics\n",
                "    fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
                "\n",
                "    # Overall sentiment\n",
                "    metrics_df['Overall Sentiment'].plot(kind='bar', ax=axes[0,0])\n",
                "    axes[0,0].set_title('Overall Sentiment by Stock')\n",
                "    axes[0,0].set_ylabel('Sentiment Score')\n",
                "\n",
                "    # News distribution\n",
                "    metrics_df[['Positive %', 'Neutral %', 'Negative %']].plot(kind='bar', ax=axes[0,1])\n",
                "    axes[0,1].set_title('Sentiment Distribution by Stock')\n",
                "    axes[0,1].set_ylabel('Percentage')\n",
                "\n",
                "    # Correlation\n",
                "    metrics_df['Correlation'].plot(kind='bar', ax=axes[1,0])\n",
                "    axes[1,0].set_title('Sentiment-Return Correlation by Stock')\n",
                "    axes[1,0].set_ylabel('Correlation Coefficient')\n",
                "\n",
                "    # Best lag correlation\n",
                "    metrics_df['Best Lag Correlation'].plot(kind='bar', ax=axes[1,1])\n",
                "    axes[1,1].set_title('Best Lag Correlation by Stock')\n",
                "    axes[1,1].set_ylabel('Correlation Coefficient')\n",
                "\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "    # Display metrics table\n",
                "    display(metrics_df.round(3))\n",
                "else:\n",
                "    print(\"No metrics collected - check if sentiment analysis completed successfully\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to file
with open('notebooks/sentiment_correlation_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!") 