import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
import plotly.graph_objects as go
from ta import momentum, trend, volatility
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks as scipy_find_peaks

class AIStrategyBuilder:
    def __init__(self, market_data_service, trading_strategy):
        self.market_data = market_data_service
        self.strategy = trading_strategy
        self.scaler = MinMaxScaler()
        self.price_data = None
        self.model = self._build_ml_model()
        
    def render(self):
        """Render AI Strategy Builder interface"""
        st.header("AI Strategy Builder")
        
        # Strategy type selection
        strategy_type = st.selectbox(
            "Select Strategy Type",
            [
                "Pattern Recognition",
                "Technical Analysis",
                "Multi-Timeframe Strategy",
                "Hybrid Strategy"
            ]
        )
        
        if strategy_type == "Pattern Recognition":
            self._render_pattern_strategy()
        elif strategy_type == "Technical Analysis":
            self._render_technical_strategy()
        elif strategy_type == "Multi-Timeframe Strategy":
            self._render_multi_timeframe_strategy()
        else:
            self._render_hybrid_strategy()
            
    def _render_pattern_strategy(self):
        """Render pattern recognition strategy interface"""
        st.subheader("Pattern Recognition Strategy")
        
        # Pattern settings
        lookback_period = st.slider("Lookback Period (days)", 5, 100, 20)
        min_pattern_size = st.slider("Minimum Pattern Size", 3, 20, 5)
        
        # Select patterns to detect
        patterns = st.multiselect(
            "Select Patterns to Detect",
            ["Double Top/Bottom", "Head and Shoulders", "Triangle", "Channel", "Flag"]
        )
        
        if st.button("Analyze Patterns"):
            self._analyze_patterns(lookback_period, min_pattern_size, patterns)
    
    def _render_technical_strategy(self):
        """Render technical analysis strategy interface"""
        st.subheader("Technical Analysis Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            indicators = st.multiselect(
                "Select Technical Indicators",
                ["RSI", "MACD", "Bollinger Bands", "Moving Averages"]
            )
            
            timeframe = st.selectbox(
                "Select Timeframe",
                ["1D", "1H", "30min", "15min", "5min"]
            )
            
        with col2:
            rsi_period = st.slider("RSI Period", 7, 21, 14)
            ma_periods = st.multiselect("Moving Average Periods", [9, 20, 50, 200])
            
        if st.button("Generate Technical Analysis"):
            self._generate_technical_analysis(indicators, timeframe, rsi_period, ma_periods)
    
    def _render_multi_timeframe_strategy(self):
        """Render multi-timeframe strategy interface"""
        st.subheader("Multi-Timeframe Strategy")
        
        timeframes = st.multiselect(
            "Select Multiple Timeframes",
            ["1D", "4H", "1H", "30min", "15min", "5min"]
        )
        
        features = st.multiselect(
            "Select Features",
            ["Price Action", "Volume", "RSI", "MACD", "Bollinger Bands"]
        )
        
        if st.button("Analyze Multiple Timeframes"):
            self._analyze_multiple_timeframes(timeframes, features)
    
    def _render_hybrid_strategy(self):
        """Render hybrid strategy interface"""
        st.subheader("Hybrid Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            components = st.multiselect(
                "Strategy Components",
                ["Technical Analysis", "Pattern Recognition", "Price Action", "Volume Analysis"]
            )
            
            risk_level = st.select_slider(
                "Risk Level",
                options=["Low", "Medium", "High"]
            )
            
        with col2:
            position_sizing = st.slider("Position Sizing (%)", 1, 100, 10)
            max_positions = st.number_input("Maximum Open Positions", 1, 10, 3)
            
        if st.button("Generate Hybrid Strategy"):
            self._generate_hybrid_strategy(components, risk_level, position_sizing, max_positions)
    
    def _analyze_patterns(self, lookback: int, min_size: int, patterns: List[str]):
        """Analyze price patterns"""
        if self.price_data is None:
            self.price_data = self._fetch_price_data()
            
        if self.price_data is None:
            st.error("No price data available. Please fetch market data first.")
            return
            
        st.write("Analyzing patterns...")
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in self.price_data.columns for col in required_columns):
            st.error(f"Missing required columns. Available columns: {self.price_data.columns.tolist()}")
            return
        
        try:
            # Plot with detected patterns
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=self.price_data.index,
                open=self.price_data['Open'],
                high=self.price_data['High'],
                low=self.price_data['Low'],
                close=self.price_data['Close'],
                name="Price"
            ))
            
            # Add pattern detection
            for pattern in patterns:
                if pattern == "Double Top/Bottom":
                    # Simple double top/bottom detection
                    peaks = self._find_peaks(self.price_data['Close'])
                    fig.add_trace(go.Scatter(
                        x=self.price_data.index[peaks],
                        y=self.price_data['Close'].iloc[peaks],
                        mode='markers',
                        name=f"{pattern} Points",
                        marker=dict(size=10, symbol='triangle-up')
                    ))
            
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error in pattern analysis: {str(e)}")
    
    def _generate_technical_analysis(self, indicators: List[str], timeframe: str, 
                                   rsi_period: int, ma_periods: List[int]):
        """Generate technical analysis"""
        if self.price_data is None:
            self.price_data = self._fetch_price_data()
            
        if self.price_data is None:
            st.error("No price data available.")
            return
            
        try:
            # Handle complex column names from yfinance
            close_col = [col for col in self.price_data.columns if 'Close' in str(col)][0]
            close_prices = pd.Series(self.price_data[close_col].values, 
                                   index=self.price_data.index)
            
            analysis_results = pd.DataFrame(index=close_prices.index)
            
            if "RSI" in indicators:
                analysis_results['RSI'] = momentum.rsi(close_prices, window=rsi_period, fillna=True)
            
            if "MACD" in indicators:
                analysis_results['MACD'] = trend.macd(pd.Series(close_prices))
                analysis_results['MACD_Signal'] = trend.macd_signal(pd.Series(close_prices))
            
            if "Bollinger Bands" in indicators:
                bb = volatility.BollingerBands(close=pd.Series(close_prices))
                analysis_results['BB_Upper'] = bb.bollinger_hband()
                analysis_results['BB_Middle'] = bb.bollinger_mavg()
                analysis_results['BB_Lower'] = bb.bollinger_lband()
            
            st.dataframe(analysis_results)
            
            # Plot indicators
            self._plot_technical_analysis(analysis_results, indicators)
        except Exception as e:
            st.error(f"Error generating technical analysis: {str(e)}")
    
    def _analyze_multiple_timeframes(self, timeframes: List[str], features: List[str]):
        """Analyze multiple timeframes"""
        if not timeframes:
            st.warning("Please select at least one timeframe")
            return
            
        for timeframe in timeframes:
            st.subheader(f"Analysis for {timeframe} timeframe")
            self.price_data = self._fetch_price_data(timeframe=timeframe)
            self._generate_technical_analysis(features, timeframe, 14, [20, 50])
    
    def _generate_hybrid_strategy(self, components: List[str], risk_level: str,
                                position_size: float, max_positions: int):
        """Generate hybrid trading strategy"""
        if not components:
            st.warning("Please select at least one strategy component")
            return
            
        st.subheader("Hybrid Strategy Analysis")
        
        # Risk management settings
        st.write("Risk Management Settings:")
        st.write(f"- Risk Level: {risk_level}")
        st.write(f"- Position Size: {position_size}%")
        st.write(f"- Maximum Positions: {max_positions}")
        
        # Generate analysis for each component
        for component in components:
            st.subheader(f"{component} Analysis")
            if component == "Technical Analysis":
                self._generate_technical_analysis(
                    ["RSI", "MACD", "Bollinger Bands"],
                    "1D",
                    14,
                    [20, 50]
                )
            elif component == "Pattern Recognition":
                self._analyze_patterns(20, 5, ["Double Top/Bottom", "Triangle"])
    
    def _fetch_price_data(self, timeframe="1D"):
        """Fetch price data for analysis"""
        try:
            # Use the market data service to fetch data
            if not hasattr(self.market_data, 'get_history'):
                st.error("Market data service does not support historical data fetching")
                return None
                
            return self.market_data.get_history(
                symbol="NIFTY",
                start=date.today() - timedelta(days=100),
                end=date.today()
            )
        except Exception as e:
            st.error(f"Error fetching price data: {str(e)}")
            return None
    
    def _plot_technical_analysis(self, analysis_results: pd.DataFrame, indicators: List[str]):
        """Plot technical analysis indicators"""
        if self.price_data is None:
            return
            
        fig = go.Figure()
        
        # Add price
        fig.add_trace(go.Candlestick(
            x=self.price_data.index,
            open=self.price_data['Open'],
            high=self.price_data['High'],
            low=self.price_data['Low'],
            close=self.price_data['Close'],
            name="Price"
        ))
        
        # Add indicators
        for indicator in indicators:
            if indicator == "RSI":
                fig.add_trace(go.Scatter(
                    x=analysis_results.index,
                    y=analysis_results['RSI'],
                    name="RSI"
                ))
            elif indicator == "MACD":
                fig.add_trace(go.Scatter(
                    x=analysis_results.index,
                    y=analysis_results['MACD'],
                    name="MACD"
                ))
                
        st.plotly_chart(fig)
    
    def _build_ml_model(self):
        """Build machine learning model"""
        return RandomForestRegressor(n_estimators=100, random_state=42) 
    
    def _find_peaks(self, prices):
        """Find peaks in price data for pattern detection"""
        try:
            # Convert to 1D numpy array
            price_array = np.array(prices.values).flatten()
            # Remove any NaN values
            price_array = price_array[~np.isnan(price_array)]
            peaks, _ = scipy_find_peaks(price_array, distance=20, prominence=1)
            return peaks
        except Exception as e:
            st.error(f"Error finding peaks: {str(e)}")
            return [] 