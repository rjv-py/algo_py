import streamlit as st
import numpy as np
import pandas as pd
from nsepy import get_history
from datetime import date, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

class QuantumTrading:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = self._build_ml_model()
    
    def _build_ml_model(self):
        """Build a RandomForest model instead of quantum circuit"""
        return RandomForestRegressor(n_estimators=100, random_state=42)

    def create_advanced_features(self, data_points):
        """Create advanced features instead of quantum features"""
        features = []
        # Clean and normalize data
        data_points = np.array(data_points, dtype=np.float64)
        data_points = np.nan_to_num(data_points, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate various technical features
        for i in range(len(data_points) - 3):
            window = data_points[i:i+3]
            # Avoid division by zero
            returns = (window[-1] - window[0]) / (window[0] + 1e-8)
            features.append([
                np.mean(window),
                np.std(window),
                returns,  # Returns with safety
                np.max(window) - np.min(window)  # Range
            ])
        
        return np.array(features, dtype=np.float64)

    def quantum_feature_extraction(self, price_data):
        if not isinstance(price_data, np.ndarray):
            price_data = np.array(price_data).flatten()
        
        if len(price_data) < 4:
            st.warning("Not enough price data for analysis")
            return price_data
        
        normalized_data = self.scaler.fit_transform(price_data.reshape(-1, 1)).flatten()
        features = self.create_advanced_features(normalized_data)
        
        if len(features) == 0:
            st.warning("Could not generate features")
            return price_data
        
        # Use RandomForest to generate predictions
        X = features[:-1]  # Features except last point
        y = normalized_data[4:].reshape(-1, 1)
        
        if len(X) == 0 or len(y) == 0:
            st.warning("Not enough data for prediction")
            return price_data
        
        # Train model
        self.model.fit(X, y.ravel())
        
        # Generate predictions
        predictions = self.model.predict(features)
        
        return self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten() 