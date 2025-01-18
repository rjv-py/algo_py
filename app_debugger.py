import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import traceback
from options_chain import OptionsChainAnalyzer
from quantum_trading_app import QuantumTrading
from ai_strategy_builder import AIStrategyBuilder

class AppDebugger:
    def __init__(self):
        st.set_page_config(page_title="Quantum AI Trading Debugger", layout="wide")
        self.options_analyzer = OptionsChainAnalyzer()
        self.quantum_trader = QuantumTrading()
        self.strategy_builder = AIStrategyBuilder(
            market_data_service=self.options_analyzer,
            trading_strategy=self.quantum_trader
        )

    def run_diagnostics(self):
        st.title("🔍 Trading System Diagnostics")
        
        tests = {
            "Data Fetching": self.test_data_fetching,
            "Options Chain": self.test_options_chain,
            "Technical Analysis": self.test_technical_analysis,
            "Pattern Recognition": self.test_pattern_recognition,
            "Quantum Features": self.test_quantum_features
        }
        
        # Test selector
        test_to_run = st.selectbox("Select Test to Run", list(tests.keys()))
        
        if st.button("Run Test"):
            with st.spinner(f"Running {test_to_run} test..."):
                tests[test_to_run]()

    def test_data_fetching(self):
        st.subheader("📊 Testing Data Fetching")
        try:
            # Test different symbols
            symbols = ["NIFTY", "RELIANCE", "INFY"]
            
            for symbol in symbols:
                st.write(f"Testing {symbol}...")
                data = self.options_analyzer.get_history(
                    symbol=symbol,
                    start=date.today() - timedelta(days=30),
                    end=date.today()
                )
                
                if data is not None and not data.empty:
                    st.success(f"✅ Successfully fetched data for {symbol}")
                    st.write("Data Sample:")
                    st.dataframe(data.head())
                    st.write("Data Info:")
                    st.write({
                        "Shape": data.shape,
                        "Columns": data.columns.tolist(),
                        "Missing Values": data.isnull().sum().to_dict(),
                        "Data Types": data.dtypes.to_dict()
                    })
                else:
                    st.error(f"❌ Failed to fetch data for {symbol}")
        
        except Exception as e:
            st.error("❌ Data Fetching Test Failed")
            st.code(traceback.format_exc())

    def test_options_chain(self):
        st.subheader("🔗 Testing Options Chain")
        try:
            symbol = "NIFTY"
            expiry = date.today() + timedelta(days=30)
            
            st.write("Fetching options chain...")
            options_data = self.options_analyzer.fetch_options_chain(symbol, expiry)
            
            if options_data is not None:
                st.success("✅ Options chain generated successfully")
                st.write("Options Data Sample:")
                st.dataframe(options_data.head())
                
                # Validate data
                st.write("Data Validation:")
                st.write({
                    "Shape": options_data.shape,
                    "Columns": options_data.columns.tolist(),
                    "Contains NaN": options_data.isna().any().any(),
                    "Contains Inf": np.isinf(options_data.values).any()
                })
            else:
                st.error("❌ Failed to generate options chain")
        
        except Exception as e:
            st.error("❌ Options Chain Test Failed")
            st.code(traceback.format_exc())

    def test_technical_analysis(self):
        st.subheader("📈 Testing Technical Analysis")
        try:
            st.write("Fetching price data...")
            self.strategy_builder.price_data = self.strategy_builder._fetch_price_data()
            
            if self.strategy_builder.price_data is None:
                st.error("Failed to fetch price data")
                return
            
            # Clean up column names for display
            st.write("Raw Columns:", [str(col) for col in self.strategy_builder.price_data.columns])
            
            # Show sample data
            st.write("Sample Data:")
            st.dataframe(self.strategy_builder.price_data.head())
            
            st.write("Price Data Info:")
            st.write({
                "Shape": self.strategy_builder.price_data.shape,
                "Columns": [str(col) for col in self.strategy_builder.price_data.columns],
                "Data Types": {str(col): str(dtype) for col, dtype in 
                             self.strategy_builder.price_data.dtypes.items()}
            })
            
            # Test all indicators
            indicators = ["RSI", "MACD", "Bollinger Bands"]
            timeframe = "1D"
            rsi_period = 14
            ma_periods = [20, 50]
            
            st.write("Generating technical analysis...")
            self.strategy_builder._generate_technical_analysis(
                indicators, timeframe, rsi_period, ma_periods
            )
            
            st.success("✅ Technical analysis completed")
            
        except Exception as e:
            st.error("❌ Technical Analysis Test Failed")
            st.error(f"Error: {str(e)}")
            st.code(traceback.format_exc())

    def test_pattern_recognition(self):
        st.subheader("🎯 Testing Pattern Recognition")
        try:
            patterns = ["Double Top/Bottom", "Head and Shoulders", "Triangle"]
            
            st.write("Analyzing patterns...")
            self.strategy_builder._analyze_patterns(20, 5, patterns)
            
            st.success("✅ Pattern recognition completed")
            
        except Exception as e:
            st.error("❌ Pattern Recognition Test Failed")
            st.code(traceback.format_exc())

    def test_quantum_features(self):
        st.subheader("⚛️ Testing Quantum Features")
        try:
            # Generate sample price data
            price_data = pd.Series(np.random.rand(100))
            
            st.write("Original Data Sample:", price_data[:5])
            st.write("Data Shape:", price_data.shape)
            st.write("Data Type:", type(price_data))
            
            # Test feature extraction
            features = self.quantum_trader.quantum_feature_extraction(price_data)
            
            if features is not None:
                st.success("✅ Quantum features generated successfully")
                st.write("Features Sample:", features[:5])
                st.write("Features Shape:", features.shape)
                st.write("Features Type:", type(features))
            else:
                st.error("❌ Failed to generate quantum features")
                
        except Exception as e:
            st.error("❌ Quantum Features Test Failed")
            st.code(traceback.format_exc())

def main():
    debugger = AppDebugger()
    debugger.run_diagnostics()

if __name__ == "__main__":
    main() 