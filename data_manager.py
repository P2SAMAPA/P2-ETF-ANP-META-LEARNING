"""
data_manager.py - Data loading, preprocessing, and feature engineering for ANP meta-training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Tuple, Optional, List
import logging
from us_calendar import US_TradingCalendar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config):
        self.config = config
        self.calendar = US_TradingCalendar()
        self.etf_universe = config.ETFS_MASTER
        self.macro_tickers = config.MACRO_TICKERS
        self.start_date = config.START_DATE
        self.end_date = config.END_DATE
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ETF price data and macro indicators.
        Handles ETFs with different trading histories by aligning to common dates.
        """
        logger.info(f"Loading data for {len(self.etf_universe)} ETFs...")
        
        # Load ETF data
        etf_data = {}
        for ticker in self.etf_universe:
            try:
                df = yf.download(
                    ticker, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True
                )
                if not df.empty:
                    etf_data[ticker] = df['Close']
                    logger.debug(f"Loaded {ticker}: {len(df)} days")
                else:
                    logger.warning(f"No data for {ticker}")
            except Exception as e:
                logger.error(f"Error loading {ticker}: {e}")
        
        # Combine ETF data
        etf_df = pd.DataFrame(etf_data)
        
        # Load macro data
        macro_data = {}
        for ticker in self.macro_tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True
                )
                if not df.empty:
                    macro_data[ticker] = df['Close']
            except Exception as e:
                logger.error(f"Error loading macro {ticker}: {e}")
        
        macro_df = pd.DataFrame(macro_data)
        
        # CRITICAL FIX: Align all data to common dates where ALL ETFs and macros have data
        # This handles ETFs with shorter trading histories
        logger.info("Aligning data to common dates...")
        
        # Combine all data
        combined = etf_df.join(macro_df, how='inner')
        
        # Drop rows where ANY column has missing data
        # This ensures every date has complete data for ALL ETFs and macros
        combined = combined.dropna()
        
        # Split back into ETF and macro dataframes
        aligned_etf_df = combined[self.etf_universe]
        aligned_macro_df = combined[self.macro_tickers]
        
        logger.info(f"Aligned data: {len(aligned_etf_df)} common dates")
        logger.info(f"Date range: {aligned_etf_df.index[0]} to {aligned_etf_df.index[-1]}")
        
        # Calculate returns
        log_returns = np.log(aligned_etf_df / aligned_etf_df.shift(1)).dropna()
        
        return log_returns, aligned_macro_df
    
    def build_features(
        self, 
        log_returns: pd.DataFrame, 
        macro_df: pd.DataFrame,
        avail: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build feature matrix X and target matrix Y for ANP training.
        
        Args:
            log_returns: DataFrame of log returns for all ETFs
            macro_df: DataFrame of macro indicators
            avail: Optional availability mask (not used in current version)
            
        Returns:
            X: Feature matrix [n_samples, n_features]
            Y: Target matrix [n_samples, n_etfs]
            dates: Array of dates
        """
        logger.info("Building features...")
        
        # Ensure data is aligned
        common_idx = log_returns.index.intersection(macro_df.index)
        log_returns = log_returns.loc[common_idx]
        macro_df = macro_df.loc[common_idx]
        
        # Create feature rows
        feature_rows = []
        target_rows = []
        valid_dates = []
        
        context_size = self.config.CONTEXT_SIZE
        # We need context_size + query_size samples for each episode
        # For feature building, we create sliding windows
        
        for i in range(context_size, len(log_returns)):
            # Get date for this sample
            current_date = log_returns.index[i]
            
            # Get context window (last 21 days of returns and macros)
            context_returns = log_returns.iloc[i-context_size:i].values  # [21, n_etfs]
            context_macro = macro_df.iloc[i-context_size:i].values  # [21, n_macros]
            
            # Flatten context: ETF returns (21*42) + macro indicators (21*4)
            # This creates a consistent feature vector regardless of ETF history
            flat_features = np.concatenate([
                context_returns.flatten(),  # 21 * n_etfs
                context_macro.flatten()     # 21 * n_macros
            ])
            
            # Target: next day returns for all ETFs (should all be available)
            target = log_returns.iloc[i].values  # [n_etfs]
            
            # Check for any NaN values (shouldn't happen after alignment)
            if np.isnan(flat_features).any() or np.isnan(target).any():
                logger.warning(f"NaN found at date {current_date}, skipping")
                continue
            
            feature_rows.append(flat_features)
            target_rows.append(target)
            valid_dates.append(current_date)
        
        # Convert to numpy arrays
        try:
            X = np.array(feature_rows, dtype=np.float32)
            Y = np.array(target_rows, dtype=np.float32)
            dates = np.array(valid_dates)
            
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Target matrix shape: {Y.shape}")
            
            return X, Y, dates
            
        except ValueError as e:
            logger.error(f"Error converting to arrays: {e}")
            logger.error(f"Feature rows lengths: {[len(row) for row in feature_rows[:5]]}")
            logger.error(f"Expected features per row: {len(feature_rows[0]) if feature_rows else 0}")
            raise
    
    def get_episode_data(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        dates: np.ndarray,
        context_size: int,
        query_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random episode for meta-training.
        
        Args:
            X: Feature matrix
            Y: Target matrix
            dates: Array of dates
            context_size: Number of context points
            query_size: Number of query points
            
        Returns:
            context_x, context_y, query_x, query_y
        """
        # Randomly select a start index
        max_start = len(X) - context_size - query_size
        if max_start <= 0:
            raise ValueError("Not enough data for episode sampling")
        
        start_idx = np.random.randint(0, max_start)
        
        # Split into context and query
        context_end = start_idx + context_size
        query_end = context_end + query_size
        
        context_x = X[start_idx:context_end]
        context_y = Y[start_idx:context_end]
        query_x = X[context_end:query_end]
        query_y = Y[context_end:query_end]
        
        return context_x, context_y, query_x, query_y

# Factory function for backward compatibility
def build_features(log_returns, macro_df, avail=None):
    """Legacy function wrapper for meta_trainer.py compatibility"""
    from config import config
    dm = DataManager(config)
    return dm.build_features(log_returns, macro_df, avail)

def get_episode_data(X, Y, dates, context_size, query_size):
    """Legacy function wrapper for meta_trainer.py compatibility"""
    from config import config
    dm = DataManager(config)
    return dm.get_episode_data(X, Y, dates, context_size, query_size)
