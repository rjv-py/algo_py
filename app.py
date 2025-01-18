import streamlit as st
from quantum_trading_app import QuantumTrading
from options_chain import OptionsChainAnalyzer
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta, datetime
from ai_strategy_builder import AIStrategyBuilder

st.set_page_config(page_title="Quantum Options Trading AI", layout="wide")

def main():
    st.title("Quantum AI Options Trading Platform")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Add system status indicators
    system_status = st.sidebar.empty()
    
    # Add tabs for different components
    tab1, tab2 = st.tabs(["Market Analysis", "AI Strategy Builder"])
    
    with tab1:
        try:
            # Initialize our trading systems
            quantum_trader = QuantumTrading()
            options_analyzer = OptionsChainAnalyzer()
            system_status.success("Systems initialized successfully")
            
            # Sidebar for user inputs
            st.sidebar.header("Trading Parameters")
            symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., NIFTY)", "NIFTY")
            
            # Add market status check
            market_status = "Open" if is_market_open() else "Closed"
            st.sidebar.info(f"Market Status: {market_status}")
            
            expiry_date = st.sidebar.date_input("Select Expiry Date", 
                                               min_value=date.today(),
                                               value=date.today() + timedelta(days=30))
            
            # Fetch and display options chain
            if st.button("Analyze Market"):
                with st.spinner("Fetching market data..."):
                    try:
                        options_data = options_analyzer.fetch_options_chain(symbol, expiry_date)
                        
                        if debug_mode:
                            st.write("Raw Options Data:")
                            st.write(options_data)
                        
                        if isinstance(options_data, pd.DataFrame) and not options_data.empty:
                            st.subheader("Options Chain Analysis")
                            st.dataframe(options_data)
                            
                            if 'Close' not in options_data.columns:
                                st.error("Missing 'Close' price data")
                                return
                            
                            # Convert Close prices to proper format
                            close_prices = options_data['Close'].values.astype(float)
                            
                            # Quantum Analysis
                            quantum_features = quantum_trader.quantum_feature_extraction(
                                close_prices)
                            
                            if debug_mode:
                                st.write("Quantum Features:")
                                st.write(quantum_features)
                            
                            # Plot predictions
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=options_data['Close'], 
                                                   name="Actual Price"))
                            fig.add_trace(go.Scatter(y=quantum_features, 
                                                   name="Quantum Prediction"))
                            st.plotly_chart(fig)
                            
                            # Trading signals
                            st.subheader("Trading Signals")
                            generate_trading_signals(quantum_features, options_data)
                        else:
                            st.error("Failed to fetch options data")
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        if debug_mode:
                            st.exception(e)
        except Exception as e:
            system_status.error(f"System initialization failed: {str(e)}")

    with tab2:
        # Initialize and render AI Strategy Builder
        strategy_builder = AIStrategyBuilder(
            market_data_service=options_analyzer,
            trading_strategy=quantum_trader
        )
        strategy_builder.render()

def generate_trading_signals(quantum_features, options_data):
    signals = pd.DataFrame()
    signals['Price'] = options_data['Close']
    signals['Quantum_Signal'] = np.where(quantum_features > np.mean(quantum_features), 
                                       'Buy', 'Sell')
    
    st.dataframe(signals)
    
    # Display current positions
    st.subheader("Recommended Positions")
    current_signal = signals['Quantum_Signal'].iloc[-1]
    st.write(f"Current Signal: {current_signal}")
    
    if current_signal == 'Buy':
        st.success("🔥 Strong Buy Signal Detected!")
    else:
        st.error("📉 Sell Signal Detected!")

def is_market_open():
    """Check if Indian market is open"""
    now = datetime.now()
    if now.weekday() in [5, 6]:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    return market_open <= now <= market_close

if __name__ == "__main__":
    main() 