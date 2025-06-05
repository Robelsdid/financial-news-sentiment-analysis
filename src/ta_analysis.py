import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf  # For market data

class TechnicalAnalyzer:
    """A class for performing technical analysis on stock data."""
    
    def __init__(self, df: pd.DataFrame, symbol: Optional[str] = None):
        """
        Initialize the TechnicalAnalyzer with stock data.
        
        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data
            symbol (str, optional): Stock symbol for additional market data
        """
        # Validate input data
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        self.df = df.copy()
        self.symbol = symbol
        
        # Convert numeric columns to float
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert Date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Remove any rows with NaN values in required columns
        self.df = self.df.dropna(subset=required_columns)
        
        # Sort by date to ensure proper calculation of indicators
        self.df = self.df.sort_values('Date')
        
        # Verify we have enough data points
        if len(self.df) < 200:  # Need at least 200 days for some indicators
            print(f"Warning: Limited data points ({len(self.df)}). Some indicators may not be accurate.")
        
        self._add_indicators()
        if self.symbol:
            self._add_market_data()
        
    def _add_market_data(self) -> None:
        """Add additional market data using yfinance."""
        try:
            if not self.symbol:
                return
                
            # Get additional market data using yfinance
            ticker = yf.Ticker(self.symbol)
            
            # Add market cap and other fundamental data
            info = ticker.info
            self.market_cap = info.get('marketCap')
            self.pe_ratio = info.get('forwardPE')
            self.dividend_yield = info.get('dividendYield')
            
            # Get institutional holders data
            self.institutional_holders = ticker.institutional_holders
            
            # Get major holders data
            self.major_holders = ticker.major_holders
            
            # Add financial ratios
            self.df['P/E_Ratio'] = info.get('forwardPE')
            self.df['P/B_Ratio'] = info.get('priceToBook')
            self.df['ROE'] = info.get('returnOnEquity')
            
            # Add market sentiment (using institutional ownership as a proxy)
            if self.institutional_holders is not None:
                total_inst_ownership = self.institutional_holders['Value'].sum()
                self.df['Market_Sentiment'] = total_inst_ownership / self.market_cap if self.market_cap else None
            else:
                self.df['Market_Sentiment'] = None
                
        except Exception as e:
            print(f"Warning: Could not fetch additional market data: {str(e)}")
    
    def _add_indicators(self) -> None:
        """Add technical indicators to the dataframe."""
        try:
            # Trend Indicators
            self.df['SMA_20'] = ta.trend.sma_indicator(self.df['Close'], window=20)
            self.df['SMA_50'] = ta.trend.sma_indicator(self.df['Close'], window=50)
            self.df['SMA_200'] = ta.trend.sma_indicator(self.df['Close'], window=200)
            self.df['EMA_20'] = ta.trend.ema_indicator(self.df['Close'], window=20)
            self.df['EMA_50'] = ta.trend.ema_indicator(self.df['Close'], window=50)
            
            # Momentum Indicators
            self.df['RSI'] = ta.momentum.RSIIndicator(self.df['Close']).rsi()
            stoch = ta.momentum.StochasticOscillator(self.df['High'], self.df['Low'], self.df['Close'])
            self.df['Stoch_k'] = stoch.stoch()
            self.df['Stoch_d'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(self.df['Close'])
            self.df['MACD'] = macd.macd()
            self.df['MACD_signal'] = macd.macd_signal()
            self.df['MACD_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            boll = ta.volatility.BollingerBands(self.df['Close'])
            self.df['BB_high'] = boll.bollinger_hband()
            self.df['BB_low'] = boll.bollinger_lband()
            self.df['BB_mid'] = boll.bollinger_mavg()
            self.df['BB_width'] = (self.df['BB_high'] - self.df['BB_low']) / self.df['BB_mid']
            
            # Volume Indicators
            self.df['OBV'] = ta.volume.on_balance_volume(self.df['Close'], self.df['Volume'])
            self.df['MFI'] = ta.volume.money_flow_index(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])
            
            # Volatility Indicators
            self.df['ATR'] = ta.volatility.average_true_range(self.df['High'], self.df['Low'], self.df['Close'])
            
            # Returns and Volatility
            self.df['Daily_Return'] = self.df['Close'].pct_change()
            self.df['Cumulative_Return'] = (1 + self.df['Daily_Return']).cumprod() - 1
            self.df['Volatility_20d'] = self.df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
            
            # Additional Trend Indicators
            self.df['ADX'] = ta.trend.ADXIndicator(self.df['High'], self.df['Low'], self.df['Close']).adx()
            self.df['CCI'] = ta.trend.CCIIndicator(self.df['High'], self.df['Low'], self.df['Close']).cci()
            self.df['DMI_Plus'] = ta.trend.ADXIndicator(self.df['High'], self.df['Low'], self.df['Close']).adx_pos()
            self.df['DMI_Minus'] = ta.trend.ADXIndicator(self.df['High'], self.df['Low'], self.df['Close']).adx_neg()
            
            # Additional Momentum Indicators
            self.df['Williams_R'] = ta.momentum.WilliamsRIndicator(self.df['High'], self.df['Low'], self.df['Close']).williams_r()
            self.df['ROC'] = ta.momentum.ROCIndicator(self.df['Close']).roc()
            self.df['TRIX'] = ta.trend.TRIXIndicator(self.df['Close']).trix()
            
            # Additional Volatility Indicators
            self.df['Keltner_Upper'] = ta.volatility.KeltnerChannel(self.df['High'], self.df['Low'], self.df['Close']).keltner_channel_hband()
            self.df['Keltner_Lower'] = ta.volatility.KeltnerChannel(self.df['High'], self.df['Low'], self.df['Close']).keltner_channel_lband()
            self.df['Keltner_Middle'] = ta.volatility.KeltnerChannel(self.df['High'], self.df['Low'], self.df['Close']).keltner_channel_mband()
            
            # Additional Volume Indicators
            self.df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume']).chaikin_money_flow()
            self.df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume']).volume_weighted_average_price()
            
            # Custom Indicators
            self.df['Price_Channel_High'] = self.df['High'].rolling(window=20).max()
            self.df['Price_Channel_Low'] = self.df['Low'].rolling(window=20).min()
            self.df['Price_Channel_Mid'] = (self.df['Price_Channel_High'] + self.df['Price_Channel_Low']) / 2
            
            # Drop any rows with NaN values that might have been created by the indicators
            self.df = self.df.dropna()
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            raise
    
    def get_technical_summary(self) -> Dict:
        """Generate a summary of current technical indicators and signals."""
        try:
            if len(self.df) == 0:
                raise ValueError("No data available for analysis")
                
            latest = self.df.iloc[-1]
            
            return {
                'Price': {
                    'Current': float(latest['Close']),
                    'SMA_20': float(latest['SMA_20']),
                    'SMA_50': float(latest['SMA_50']),
                    'SMA_200': float(latest['SMA_200'])
                },
                'Trend': {
                    'Above_SMA_20': bool(latest['Close'] > latest['SMA_20']),
                    'Above_SMA_50': bool(latest['Close'] > latest['SMA_50']),
                    'Above_SMA_200': bool(latest['Close'] > latest['SMA_200']),
                    'Golden_Cross': bool(latest['SMA_50'] > latest['SMA_200']),
                    'Death_Cross': bool(latest['SMA_50'] < latest['SMA_200'])
                },
                'Momentum': {
                    'RSI': float(latest['RSI']),
                    'RSI_Signal': 'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral',
                    'MACD_Signal': 'Bullish' if latest['MACD'] > latest['MACD_signal'] else 'Bearish',
                    'Stochastic_Signal': 'Overbought' if latest['Stoch_k'] > 80 else 'Oversold' if latest['Stoch_k'] < 20 else 'Neutral'
                },
                'Volatility': {
                    'BB_Width': float(latest['BB_width']),
                    'ATR': float(latest['ATR']),
                    'Volatility_20d': float(latest['Volatility_20d'])
                },
                'Volume': {
                    'OBV': float(latest['OBV']),
                    'MFI': float(latest['MFI']),
                    'MFI_Signal': 'Overbought' if latest['MFI'] > 80 else 'Oversold' if latest['MFI'] < 20 else 'Neutral'
                }
            }
        except Exception as e:
            print(f"Error generating technical summary: {str(e)}")
            raise
    
    def get_performance_metrics(self) -> Dict:
        """Calculate key performance metrics."""
        returns = self.df['Daily_Return']
        
        return {
            'Total Return (%)': (self.df['Cumulative_Return'].iloc[-1] * 100),
            'Annualized Return (%)': (returns.mean() * 252 * 100),
            'Annualized Volatility (%)': (returns.std() * np.sqrt(252) * 100),
            'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'Max Drawdown (%)': ((self.df['Close'] / self.df['Close'].cummax() - 1).min() * 100),
            'Win Rate (%)': (len(returns[returns > 0]) / len(returns) * 100)
        }
    
    def plot_technical_analysis(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """Create an interactive plotly visualization of technical indicators."""
        try:
            if len(self.df) == 0:
                raise ValueError("No data available for plotting")
                
            df = self.df.copy()
            
            # Convert string dates to datetime if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['Date'] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['Date'] <= end_date]
            
            if len(df) == 0:
                raise ValueError("No data available for the specified date range")
            
            fig = make_subplots(rows=4, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.05,
                               row_heights=[0.4, 0.2, 0.2, 0.2])

            # Candlestick chart
            fig.add_trace(go.Candlestick(x=df['Date'],
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name='OHLC'),
                          row=1, col=1)

            # Moving averages
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name='SMA 200', line=dict(color='red')), row=1, col=1)

            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_high'], name='BB High', line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_low'], name='BB Low', line=dict(color='gray', dash='dash')), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # MACD
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD'), row=4, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'], name='Signal'), row=4, col=1)
            fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_diff'], name='MACD Diff'), row=4, col=1)

            fig.update_layout(
                title='Technical Analysis',
                yaxis_title='Price',
                yaxis2_title='Volume',
                yaxis3_title='RSI',
                yaxis4_title='MACD',
                xaxis_rangeslider_visible=False,
                height=1000
            )

            fig.show()
            
        except Exception as e:
            print(f"Error plotting technical analysis: {str(e)}")
            raise
    
    def analyze_trading_signals(self) -> Dict[str, pd.Series]:
        """Analyze the effectiveness of trading signals based on technical indicators."""
        try:
            if len(self.df) == 0:
                raise ValueError("No data available for signal analysis")
                
            # Define signals
            self.df['RSI_Signal'] = np.where(self.df['RSI'] > 70, -1, np.where(self.df['RSI'] < 30, 1, 0))
            self.df['MACD_Signal'] = np.where(self.df['MACD'] > self.df['MACD_signal'], 1, -1)
            self.df['BB_Signal'] = np.where(self.df['Close'] > self.df['BB_high'], -1, np.where(self.df['Close'] < self.df['BB_low'], 1, 0))
            
            # Calculate forward returns
            self.df['Forward_Return_5d'] = self.df['Close'].shift(-5) / self.df['Close'] - 1
            
            # Analyze signal effectiveness
            signals = ['RSI_Signal', 'MACD_Signal', 'BB_Signal']
            results = {}
            
            for signal in signals:
                if signal in self.df.columns:
                    results[signal] = self.df.groupby(signal)['Forward_Return_5d'].mean()
            
            return results
            
        except Exception as e:
            print(f"Error analyzing trading signals: {str(e)}")
            raise
    
    def plot_advanced_analysis(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """Create an advanced interactive plotly visualization with additional indicators."""
        try:
            if len(self.df) == 0:
                raise ValueError("No data available for plotting")
                
            df = self.df.copy()
            
            # Convert string dates to datetime if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['Date'] >= start_date]
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['Date'] <= end_date]
            
            if len(df) == 0:
                raise ValueError("No data available for the specified date range")
            
            # Create subplots with more rows for additional indicators
            fig = make_subplots(rows=6, cols=1, 
                               shared_xaxes=True,
                               vertical_spacing=0.05,
                               row_heights=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1])

            # Candlestick chart with more indicators
            fig.add_trace(go.Candlestick(x=df['Date'],
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name='OHLC'),
                          row=1, col=1)

            # Add Keltner Channels
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Keltner_Upper'], 
                                   name='Keltner Upper', line=dict(color='purple', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Keltner_Lower'], 
                                   name='Keltner Lower', line=dict(color='purple', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Keltner_Middle'], 
                                   name='Keltner Middle', line=dict(color='purple')), row=1, col=1)

            # Volume with VWAP
            fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['VWAP'], name='VWAP', 
                                   line=dict(color='orange')), row=2, col=1)

            # ADX and DMI
            fig.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], name='ADX'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['DMI_Plus'], name='DMI+'), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['DMI_Minus'], name='DMI-'), row=3, col=1)

            # CCI and Williams %R
            fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI'], name='CCI'), row=4, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Williams_R'], name='Williams %R'), row=4, col=1)
            fig.add_hline(y=100, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=-100, line_dash="dash", line_color="green", row=4, col=1)

            # TRIX and ROC
            fig.add_trace(go.Scatter(x=df['Date'], y=df['TRIX'], name='TRIX'), row=5, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['ROC'], name='ROC'), row=5, col=1)

            # Chaikin Money Flow
            fig.add_trace(go.Scatter(x=df['Date'], y=df['CMF'], name='CMF'), row=6, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=6, col=1)

            fig.update_layout(
                title='Advanced Technical Analysis',
                yaxis_title='Price',
                yaxis2_title='Volume & VWAP',
                yaxis3_title='ADX & DMI',
                yaxis4_title='CCI & Williams %R',
                yaxis5_title='TRIX & ROC',
                yaxis6_title='CMF',
                xaxis_rangeslider_visible=False,
                height=1200
            )

            fig.show()
            
        except Exception as e:
            print(f"Error plotting advanced analysis: {str(e)}")
            raise

    def get_market_summary(self) -> Dict:
        """Generate a comprehensive market summary including fundamental data."""
        try:
            if not self.symbol:
                raise ValueError("Symbol not provided for market summary")
                
            latest = self.df.iloc[-1]
            
            summary = {
                'Technical': self.get_technical_summary(),
                'Performance': self.get_performance_metrics(),
                'Fundamental': {
                    'Market_Cap': self.market_cap,
                    'P/E_Ratio': self.pe_ratio,
                    'Dividend_Yield': self.dividend_yield,
                    'ROE': float(latest['ROE']) if 'ROE' in self.df.columns else None,
                    'P/B_Ratio': float(latest['P/B_Ratio']) if 'P/B_Ratio' in self.df.columns else None
                },
                'Market_Sentiment': {
                    'Current_Sentiment': float(latest['Market_Sentiment']) if 'Market_Sentiment' in self.df.columns else None,
                    'Institutional_Holdings': self.institutional_holders.to_dict() if hasattr(self, 'institutional_holders') else None,
                    'Major_Holders': self.major_holders.to_dict() if hasattr(self, 'major_holders') else None
                }
            }
            
            return summary
            
        except Exception as e:
            print(f"Error generating market summary: {str(e)}")
            raise 