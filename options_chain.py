import pandas as pd
import numpy as np
from nsepy import get_history
from datetime import date, timedelta
from scipy.stats import norm
from math import log, sqrt, exp
import streamlit as st

class GreeksCalculator:
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
    
    def calculate_iv(self, option_data):
        """Calculate Implied Volatility using Newton-Raphson method"""
        try:
            S = option_data['Underlying']  # Current stock price
            K = option_data['Strike Price']
            T = (option_data['Expiry'] - date.today()).days / 365
            r = self.risk_free_rate
            C = option_data['Last Price']
            
            # Initial volatility guess
            sigma = 0.3
            
            for i in range(100):
                d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
                d2 = d1 - sigma*sqrt(T)
                
                # Calculate option price
                if option_data['Option Type'] == 'CE':
                    price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
                else:
                    price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                
                diff = C - price
                
                if abs(diff) < 0.0001:
                    return sigma
                    
                # Vega calculation
                vega = S*sqrt(T)*norm.pdf(d1)
                
                # Update volatility
                sigma = sigma + diff/vega
                
            return sigma
        except:
            return np.nan
    
    def calculate_delta(self, option_data):
        """Calculate Delta"""
        try:
            S = option_data['Underlying']
            K = option_data['Strike Price']
            T = (option_data['Expiry'] - date.today()).days / 365
            r = self.risk_free_rate
            sigma = option_data['IV']
            
            d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
            
            if option_data['Option Type'] == 'CE':
                return norm.cdf(d1)
            else:
                return norm.cdf(d1) - 1
        except:
            return np.nan
    
    def calculate_gamma(self, option_data):
        """Calculate Gamma"""
        try:
            S = option_data['Underlying']
            K = option_data['Strike Price']
            T = (option_data['Expiry'] - date.today()).days / 365
            r = self.risk_free_rate
            sigma = option_data['IV']
            
            d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
            
            return norm.pdf(d1)/(S*sigma*sqrt(T))
        except:
            return np.nan
    
    def calculate_theta(self, option_data):
        """Calculate Theta"""
        try:
            S = option_data['Underlying']
            K = option_data['Strike Price']
            T = (option_data['Expiry'] - date.today()).days / 365
            r = self.risk_free_rate
            sigma = option_data['IV']
            
            d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
            d2 = d1 - sigma*sqrt(T)
            
            if option_data['Option Type'] == 'CE':
                theta = (-S*norm.pdf(d1)*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)
            else:
                theta = (-S*norm.pdf(d1)*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2)
                
            return theta/365  # Convert to daily theta
        except:
            return np.nan
    
    def calculate_vega(self, option_data):
        """Calculate Vega"""
        try:
            S = option_data['Underlying']
            K = option_data['Strike Price']
            T = (option_data['Expiry'] - date.today()).days / 365
            r = self.risk_free_rate
            sigma = option_data['IV']
            
            d1 = (log(S/K) + (r + sigma**2/2)*T) / (sigma*sqrt(T))
            
            return S*sqrt(T)*norm.pdf(d1)/100  # Divide by 100 for percentage move
        except:
            return np.nan

class OptionsChainAnalyzer:
    def __init__(self):
        self.greeks_calculator = GreeksCalculator()
    
    def get_history(self, symbol: str, start: date, end: date):
        """Get historical price data"""
        try:
            import yfinance as yf
            
            # Add debug info
            st.write(f"Fetching data for {symbol} from {start} to {end}")
            
            if symbol == "NIFTY":
                symbol = "^NSEI"  # NIFTY 50 index
                # Fallback symbols if primary fails
                fallback_symbols = ["^NSEI", "NIFTY50.NS", "NIFTY.NS"]
                
                for sym in fallback_symbols:
                    try:
                        data = yf.download(sym, start=start, end=end, interval="1d")
                        if not data.empty:
                            st.success(f"Successfully fetched data using {sym}")
                            return data
                    except:
                        continue
                
                st.error("Failed to fetch NIFTY data using all fallback symbols")
                return None
            else:
                symbol = f"{symbol}.NS"
                
            data = yf.download(symbol, start=start, end=end, interval="1d")
            
            # Validate data
            if data.empty:
                st.error("No data received from yfinance")
                return None
                
            st.success(f"Successfully fetched {len(data)} data points")
            return data
            
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def fetch_options_chain(self, symbol, expiry_date):
        try:
            underlying_data = self.get_history(symbol, 
                                             start=date.today() - timedelta(days=30),
                                             end=date.today())
            
            if underlying_data is None or underlying_data.empty:
                st.error("No underlying data available")
                return None
            
            # Handle complex column names from yfinance
            close_col = [col for col in underlying_data.columns if 'Close' in str(col)][0]
            close_prices = underlying_data[close_col].values  # Already numpy array
            
            # Create a sample options chain for demonstration
            current_price = float(close_prices[-1])
            strike_prices = np.arange(current_price * 0.9, current_price * 1.1, current_price * 0.01)
            
            # Ensure all arrays are 1D
            strike_prices = strike_prices.flatten()
            n_strikes = len(strike_prices)
            
            # Get last n prices and ensure they're 1D
            last_n_prices = close_prices[-n_strikes:] if len(close_prices) >= n_strikes else \
                           np.full(n_strikes, current_price)
            
            options_data = pd.DataFrame({
                'Strike Price': strike_prices,
                'Option Type': ['CE'] * len(strike_prices),
                'Expiry': [expiry_date] * len(strike_prices),
                'Last Price': np.random.uniform(10, 100, len(strike_prices)),
                'Underlying': [current_price] * len(strike_prices),
                'Close': last_n_prices.flatten()
            })
            
            # Add debug information
            if st.checkbox("Show data validation"):
                st.write("Data shape:", options_data.shape)
                st.write("Contains NaN:", options_data.isna().any().any())
                st.write("Contains Inf:", np.isinf(options_data.values).any())
            
            return self.process_options_chain(options_data)
        except Exception as e:
            import traceback
            st.error(f"Error fetching options chain: {str(e)}")
            if st.checkbox("Show detailed error"):
                st.code(traceback.format_exc())
            return None
    
    def process_options_chain(self, options_data):
        if isinstance(options_data, pd.DataFrame):
            # Calculate implied volatility and greeks
            options_data['IV'] = options_data.apply(
                lambda row: self.greeks_calculator.calculate_iv(row), axis=1)
            
            # Calculate option greeks
            options_data['Delta'] = options_data.apply(
                lambda row: self.greeks_calculator.calculate_delta(row), axis=1)
            options_data['Gamma'] = options_data.apply(
                lambda row: self.greeks_calculator.calculate_gamma(row), axis=1)
            options_data['Theta'] = options_data.apply(
                lambda row: self.greeks_calculator.calculate_theta(row), axis=1)
            options_data['Vega'] = options_data.apply(
                lambda row: self.greeks_calculator.calculate_vega(row), axis=1)
            
            return options_data
        return None 