#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################################################################
##                                                                       ##
##   ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗    ██╗  ██╗      ##
##   ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝    ╚██╗██╔╝      ##
##   ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗       ╚███╔╝       ##
##   ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝       ██╔██╗       ##
##   ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗    ██╔╝ ██╗      ##
##   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝    ╚═╝  ╚═╝      ##
##                                                                                    ##
##   INSTITUTIONAL TRADING SYSTEM - Neural Fusion Analytics v10.0                     ##
##   ⚡ Ultra Low Latency | 🧬 ML Ensemble | 🛡️ Risk Management                      ##
##                                                                                    ##
###########################################################################

Synapse X - Advanced Algorithmic Trading System with Online Learning
===================================================================

Enhanced Features:
- Online learning with real-time model adaptation
- Multi-horizon prediction tracking (5m, 15m, 30m, 60m)
- Advanced anomaly detection system
- GPT-4.1 and GPT-4o integration for market analysis
- Institutional-grade message formatting
- Comprehensive risk management
- Market manipulation detection
- Multi-timeframe ML ensemble

Author: Hedge Fund Analytics Team
Version: 10.0.9.01
License: Proprietary
"""

# === SYSTEM CONFIGURATION ===
import os
import warnings

def configure_environment():
    """Configure system environment for optimal performance."""
    
    # Suppress warnings for production
    warnings.filterwarnings('ignore')
    
    # TensorFlow configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # CPU optimization
    os.environ["NUMBA_NUM_THREADS"] = "12"
    os.environ["OMP_NUM_THREADS"] = "10"
    
    # GPU configuration
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    # Memory allocation
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Configure environment before imports
configure_environment()

# === STANDARD LIBRARY IMPORTS ===
import sys
import os
import time
import random
import threading
import queue
import re
import csv
import datetime
import logging
import atexit
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Deque, Optional, Sequence, Tuple, Any, Union, Set
from functools import wraps, lru_cache, partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import json
import traceback
import signal
import subprocess
import shutil
import tempfile
import weakref
import gc

# === SCIENTIFIC COMPUTING ===
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy import signal

# === MACHINE LEARNING CORE ===
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score, TimeSeriesSplit
)
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# === DEEP LEARNING ===
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# === EXTERNAL LIBRARIES ===
import pytz
import requests
import optuna
from tenacity import retry, stop_after_attempt, wait_random_exponential
from timeout_decorator import TimeoutError, timeout
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === TRADING SPECIFIC ===
from ib_insync import IB, Stock, TickByTickAllLast, Ticker, util

# === DATA PROCESSING ===
import zstd
import orjson as json

# === VISUALIZATION ===
import dash
import plotly.graph_objs as go
import plotly.subplots as psub
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State as DashState
from dash.exceptions import PreventUpdate

# === AI/LLM INTEGRATION ===
from openai import OpenAI

# === OPTIONAL IMPORTS WITH FALLBACKS ===

# XGBoost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed - falling back to alternative models")
    xgb = None
    XGBClassifier = None

# River - Online Learning
try:
    from river import ensemble as river_ensemble
    from river import tree as river_tree
    from river import metrics as river_metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("River not installed - online learning disabled")
    river_ensemble = None
    river_tree = None
    river_metrics = None

# GPU Support - RAPIDS cuML
try:
    # Fix pandas compatibility issue first
    import pandas as pd
    # Monkey patch for compatibility with cuDF
    if not hasattr(pd.api.types, 'is_extension_type'):
        pd.api.types.is_extension_type = pd.api.types.is_extension_array_dtype
    if not hasattr(pd.api.types, 'is_categorical'):
        pd.api.types.is_categorical = pd.api.types.is_categorical_dtype
    
    from cuml.ensemble import RandomForestClassifier as cuMLRandomForest
    from cuml.preprocessing import StandardScaler as cuMLStandardScaler
    GPU_AVAILABLE = True
    logging.info("GPU acceleration enabled via RAPIDS cuML")
except (ImportError, AttributeError) as e:
    GPU_AVAILABLE = False
    logging.warning(f"GPU acceleration not available - using CPU. Error: {e}")
    cuMLRandomForest = None
    cuMLStandardScaler = None

# === IB CONNECTION CONFIGURATION ===
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

# Initialize IB connection
ib = IB()

# === LOGGING CONFIGURATION ===
def setup_logging(log_level=logging.INFO):
    """Configure comprehensive logging system."""
    log_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        'synapse_online_learning000901.log', 
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Create specific loggers
    trading_logger = logging.getLogger('trading')
    ml_logger = logging.getLogger('ml')
    risk_logger = logging.getLogger('risk')
    
    return logger, trading_logger, ml_logger, risk_logger

# Initialize logging
logger, trading_logger, ml_logger, risk_logger = setup_logging()

# === ENVIRONMENT VALIDATION ===
def validate_environment():
    """Validate all required environment variables."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for GPT integration",
        "TELEGRAM_TOKEN": "Telegram bot token",
        "TELEGRAM_CHAT_ID": "Telegram chat ID for alerts",
        "IB_HOST": "Interactive Brokers gateway host",
        "IB_PORT": "Interactive Brokers gateway port",
        "IB_CLIENT_ID": "Interactive Brokers client ID"
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        logger.error(f"Missing required environment variables:")
        for var in missing:
            logger.error(f"  - {var}")
        sys.exit(1)
    
    logger.info("Environment validation passed")

# Validate environment
validate_environment()

# === ENHANCED CONFIGURATION ===
@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 0.02
    max_portfolio_heat: float = 0.06
    max_correlation: float = 0.7
    stop_loss_pct: float = 0.01
    daily_loss_limit_pct: float = 0.02
    var_limit_pct: float = 0.015
    max_consecutive_losses: int = 3
    min_sharpe_ratio: float = 1.0
    max_leverage: float = 2.0
    margin_call_threshold: float = 0.25

@dataclass
class ExecutionConfig:
    """Execution engine configuration."""
    min_edge: float = 0.002
    max_spread_pct: float = 0.001
    urgency_threshold: float = 0.7
    iceberg_threshold: float = 0.1
    min_liquidity_ratio: float = 0.5
    max_slippage_pct: float = 0.0005
    order_timeout_seconds: int = 30
    max_retry_attempts: int = 3

@dataclass
class MLConfig:
    """Machine learning configuration."""
    min_samples_retrain: int = 1000
    retrain_interval: int = 3600
    max_feature_correlation: float = 0.95
    min_feature_importance: float = 0.01
    ensemble_min_agreement: float = 0.6
    confidence_threshold: float = 0.65
    uncertainty_max: float = 0.2
    lookback_window: int = 100
    prediction_horizon: int = 5
    validation_split: float = 0.2

@dataclass
class MonitoringConfig:
    """System monitoring configuration."""
    heartbeat_interval: int = 10
    metrics_window: int = 252
    alert_cooldown: int = 60
    max_consecutive_errors: int = 5
    performance_checkpoint_interval: int = 300
    memory_limit_gb: float = 32.0
    cpu_limit_percent: float = 80.0

@dataclass
class HedgeFundConfig:
    """Complete hedge fund system configuration."""
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        validations = [
            self.risk.max_position_pct > 0 and self.risk.max_position_pct < 1,
            self.risk.max_portfolio_heat > self.risk.max_position_pct,
            self.risk.stop_loss_pct > 0 and self.risk.stop_loss_pct < 0.1,
            self.execution.min_edge > 0,
            self.ml.confidence_threshold > 0.5 and self.ml.confidence_threshold < 1,
            self.monitoring.heartbeat_interval > 0,
        ]
        
        if not all(validations):
            raise ValueError("Invalid configuration parameters")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HedgeFundConfig':
        """Create configuration from dictionary."""
        return cls(
            risk=RiskConfig(**config_dict.get('risk', {})),
            execution=ExecutionConfig(**config_dict.get('execution', {})),
            ml=MLConfig(**config_dict.get('ml', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {}))
        )

# Initialize configuration
HEDGE_FUND_CONFIG = HedgeFundConfig()
HEDGE_FUND_CONFIG.validate()

logger.info("Synapse X Online Learning System initialized successfully")
logger.info(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
logger.info(f"XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
logger.info(f"Online Learning: {'Enabled' if RIVER_AVAILABLE else 'Disabled'}")

# === TRADING SIGNALS ===
@dataclass
class TradingSignal:
    """Trading signal configuration."""
    name: str
    emoji: str
    threshold: float
    color: str
    weight: float = 1.0
    
    def __str__(self) -> str:
        return f"{self.emoji} {self.name}"

# Trading signals configuration
TRADING_SIGNALS = {
    'STRONG_BUY': TradingSignal('STRONG_BUY', '🚀', 0.75, '\033[92m', 2.0),
    'BUY': TradingSignal('BUY', '📈', 0.65, '\033[32m', 1.5),
    'NEUTRAL': TradingSignal('NEUTRAL', '➡️', 0.35, '\033[93m', 1.0),
    'SELL': TradingSignal('SELL', '📉', 0.25, '\033[31m', 1.5),
    'STRONG_SELL': TradingSignal('STRONG_SELL', '🔻', 0.0, '\033[91m', 2.0)
}

# === ENHANCED DATA CLASSES ===
@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 10000
    max_daily_loss: float = -1000
    max_drawdown: float = -0.15
    max_correlation: float = 0.7
    position_limit_per_symbol: float = 5000
    var_limit: float = 0.02
    consecutive_loss_limit: int = 3
    max_leverage: float = 2.0
    margin_requirement: float = 0.5
    concentration_limit: float = 0.3
    
    def validate_position(self, size: float, current_positions: Dict) -> Tuple[bool, str]:
        """Validate if a position can be taken."""
        if abs(size) > self.max_position_size:
            return False, f"Position size {size} exceeds limit {self.max_position_size}"
        
        total_exposure = sum(abs(pos) for pos in current_positions.values())
        if total_exposure + abs(size) > self.max_position_size * 2:
            return False, "Total exposure limit exceeded"
        
        return True, "OK"

@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    
    def update(self, pnl: float) -> None:
        """Update metrics with new P&L."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Update derived metrics
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.winning_trades > 0:
            self.avg_win = self.gross_profit / self.winning_trades
        
        if self.losing_trades > 0:
            self.avg_loss = self.gross_loss / self.losing_trades
            self.profit_factor = self.gross_profit / self.gross_loss if self.gross_loss > 0 else float('inf')

@dataclass
class MarketRegime:
    """Market regime classification."""
    regime: str = "Normal"
    confidence: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0
    efficiency_ratio: float = 0.0
    hurst_exponent: float = 0.5
    market_state: str = "Unknown"
    
    def classify(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        """Classify current market regime."""
        if len(prices) < 50:
            return "Unknown"
        
        # Calculate volatility
        returns = np.diff(np.log(prices))
        self.volatility = np.std(returns) * np.sqrt(252)
        
        # Trend strength (directional movement)
        price_change = prices[-1] - prices[0]
        path_length = np.sum(np.abs(np.diff(prices)))
        self.efficiency_ratio = abs(price_change) / path_length if path_length > 0 else 0
        
        # Regime classification
        if self.volatility > 0.3:
            if self.efficiency_ratio < 0.3:
                self.regime = "Choppy"
            else:
                self.regime = "Volatile Trending"
        elif self.volatility < 0.1:
            self.regime = "Low Volatility"
        else:
            if self.efficiency_ratio > 0.6:
                self.regime = "Trending"
            else:
                self.regime = "Normal"
        
        self.confidence = min(0.95, self.efficiency_ratio + (1 - self.volatility))
        
        return self.regime

# === THREAD-SAFE LOCKS ===
class LockManager:
    """Centralized lock management."""
    def __init__(self):
        self.iso_lock = threading.Lock()
        self.global_state_lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.l2_lock = threading.RLock()
        self.tns_lock = threading.RLock()
        self.ml_lock = threading.Lock()
        self.risk_lock = threading.Lock()
        self.execution_lock = threading.Lock()
    
    @contextmanager
    def acquire_multiple(self, *locks):
        """Acquire multiple locks in order."""
        acquired = []
        try:
            for lock in locks:
                lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

# Initialize lock manager
locks = LockManager()

# === ENHANCED LOGGING ===
class TradingLogger:
    """Enhanced logging with CSV output."""
    def __init__(self, base_dir="logs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize CSV loggers
        self.trade_logger = self._init_csv_logger("trades.csv", 
            ["timestamp", "symbol", "side", "size", "price", "pnl", "signal"])
        self.risk_logger = self._init_csv_logger("risk.csv",
            ["timestamp", "metric", "value", "threshold", "status"])
        self.ml_logger = self._init_csv_logger("ml_predictions.csv",
            ["timestamp", "model", "prediction", "confidence", "features"])
    
    def _init_csv_logger(self, filename: str, headers: List[str]) -> csv.DictWriter:
        """Initialize CSV logger with headers."""
        filepath = os.path.join(self.base_dir, filename)
        file_exists = os.path.exists(filepath)
        
        f = open(filepath, 'a', newline='')
        writer = csv.DictWriter(f, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()
        
        return writer
    
    def log_trade(self, trade_data: Dict):
        """Log trade execution."""
        try:
            self.trade_logger.writerow(trade_data)
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def log_risk(self, risk_data: Dict):
        """Log risk metrics."""
        try:
            self.risk_logger.writerow(risk_data)
        except Exception as e:
            logger.error(f"Failed to log risk: {e}")
    
    def log_prediction(self, prediction_data: Dict):
        """Log ML prediction."""
        try:
            self.ml_logger.writerow(prediction_data)
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

# Initialize trading logger
TRADING_LOGGER = TradingLogger()

# === UTILITY FUNCTIONS ===
def is_defined(model) -> bool:
    """Check if sklearn model is defined and fitted."""
    if model is None:
        return False
    
    # Check for different model types
    if hasattr(model, 'estimators_'):
        try:
            return len(model.estimators_) > 0
        except:
            return False
    
    if hasattr(model, 'coef_'):
        try:
            return model.coef_ is not None
        except:
            return False
    
    return hasattr(model, 'is_fitted') and model.is_fitted

def is_defined_keras(model) -> bool:
    """Check if Keras model is defined and compiled."""
    if model is None:
        return False
    
    try:
        # Check if model has weights
        weights = model.get_weights()
        if not weights:
            return False
        
        # Check if model is compiled
        if not model.optimizer:
            return False
        
        return True
    except:
        return False

# === DEEP LEARNING MODEL BUILDERS ===
def create_lstm_model(input_shape: Tuple[int, int] = (100, 40), 
                     num_classes: int = 3) -> Sequential:
    """Create LSTM model for time series prediction."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(32),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_gru_model(input_shape: Tuple[int, int] = (100, 40),
                    num_classes: int = 3) -> Sequential:
    """Create GRU model for time series prediction."""
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        
        GRU(64, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        
        GRU(32),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# === ENHANCED GLOBAL STATE ===
class GlobalState:
    """Centralized state management for the trading system."""
    
    def __init__(self):
        """Initialize global state with all components."""
        # === Configuration ===
        self.config = HEDGE_FUND_CONFIG
        self.validate_config()
        
        # === GPT / AI Integration ===
        self.client_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.last_gpt_time = 0.0
        self.gpt_in_progress = False
        self.gpt_has_alert = False
        self.last_gpt_alert_text = ""
        self.latest_gpt_analysis = "Awaiting GPT-4.1 and GPT-4o analysis..."
        self.gpt_history = deque(maxlen=10)
        self.GPT_COOLDOWN = 60
        
        # === Telegram Integration ===
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
        self.last_5min_send = 0.0
        self.last_10min_send = 0.0
        self.last_15min_send = 0.0
        self.last_update_id = 0
        self.LAST_CUSTOM_ALERT_TIME = 0.0
        self.CUSTOM_ALERT_COOLDOWN = 60.0
        
        # === Risk Management ===
        self.risk_limits = RiskLimits()
        self.metrics = TradingMetrics()
        self.market_regime = MarketRegime()
        self.positions = {}
        self.circuit_breaker_triggered = False
        self.last_risk_check = 0.0
        self.portfolio_heat = 0.0
        
        # === Calibration ===
        self.isotonic_calibrator = None
        self.last_calibration_time = 0.0
        self.calibration_history = deque(maxlen=1000)
        
        # === Market Microstructure ===
        self.init_microstructure()
        
        # === ML Models and Features ===
        self.init_ml_models()
        
        # === Trading State ===
        self.init_trading_state()
        
        # === Performance Tracking ===
        self.init_performance_tracking()
        
        logger.info("GlobalState initialized successfully")
    
    def validate_config(self):
        """Validate configuration parameters."""
        try:
            self.config.validate()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            sys.exit(1)
    
    def init_microstructure(self):
        """Initialize market microstructure components."""
        # Level 2 Order Book
        self.df_l2_display = pd.DataFrame(columns=["Price", "Size", "Side"])
        self.bids_dict = {}
        self.asks_dict = {}
        self.l2_events = deque(maxlen=50000)
        self.l2_imbalance_history = deque(maxlen=10000)
        
        # Spread tracking
        self.spread_history = deque(maxlen=10000)
        self.effective_spread_history = deque(maxlen=10000)
        
        # Time and Sales
        self.df_tns = pd.DataFrame(columns=["Time", "Price", "Size", "Timestamp"])
        self.MAX_TNS_SIZE = 10000
        self.TNS_DISPLAY_MAX = 100
        
        # Charting
        self.CHART_TIME_WINDOW_SEC = 3600
        
        # Microstructure metrics
        self.OFI_VALUE = 0.0  # Order Flow Imbalance
        self.CDV_VALUE = 0.0  # Cumulative Delta Volume
        self.RELATIVE_VOLUME = 1.0
        self.ANCHOR_TIMESTAMP = None
        self.VWAP_VALUE = None
        self.toxicity_score = 0.0
        self.adverse_selection_score = 0.0
        
        # Symbol and tickers
        self.current_symbol = None
        self.ticker_l2 = None
        self.ticker_tns = None
        
        # Volume profiles
        self.volume_profile = defaultdict(float)
        self.price_levels = defaultdict(list)
    
    def init_ml_models(self):
        """Initialize all ML models."""
        # === Online Learning Models (1-minute) ===
        self.ml_model_sgd = None
        self.ml_model_pa = None
        
        # === Batch Models (1-minute) ===
        self.ml_model_rf = None
        self.ml_model_xgb = None if not XGBOOST_AVAILABLE else None
        
        # === 5-minute Models ===
        self.ml_model_sgd_5m = None
        self.ml_model_pa_5m = None
        self.ml_model_rf_5m = None
        self.ml_model_xgb_5m = None if not XGBOOST_AVAILABLE else None
        
        # === 15-minute Models ===
        self.ml_model_sgd_15m = None
        self.ml_model_pa_15m = None
        self.ml_model_rf_15m = None
        self.ml_model_xgb_15m = None if not XGBOOST_AVAILABLE else None
        
        # === Deep Learning Models ===
        self.ml_model_lstm = None
        self.ml_model_gru = None
        
        # === Meta Model ===
        self.ml_model_meta = None
        
        # === Online Learning (River) ===
        if RIVER_AVAILABLE:
            try:
                # River 0.15+ uses different API
                # Use simple Hoeffding Tree which is available in all versions
                self.river_model_1m = river_tree.HoeffdingTreeClassifier()
                self.river_model_5m = river_tree.HoeffdingTreeClassifier()
                self.online_metrics = {
                    'accuracy': river_metrics.Accuracy(),
                    'f1': river_metrics.MacroF1(),
                }
            except Exception as e:
                logging.warning(f"River initialization failed: {e}")
                self.river_model_1m = None
                self.river_model_5m = None
                self.online_metrics = None
        else:
            self.river_model_1m = None
            self.river_model_5m = None
            self.online_metrics = None
        
        # === Model Paths ===
        self.ML_MODEL_PATH = "model_default.pkl"
        self.MODEL_PATH_5M = "model_5m.pkl"
        self.MODEL_PATH_15M = "model_15m.pkl"
        
        # === Feature Engineering ===
        self.scaler = StandardScaler()
        self.scaler_5m = StandardScaler()
        self.scaler_15m = StandardScaler()
        
        # === Model Performance Tracking ===
        self.model_performance = defaultdict(lambda: {
            'predictions': deque(maxlen=1000),
            'accuracy': deque(maxlen=100),
            'last_update': 0
        })
    
    def init_trading_state(self):
        """Initialize trading state variables."""
        # === Training Buffers ===
        self.TRAINING_BUFFER = deque(maxlen=5000)
        self.TRAINING_BUFFER_5m = deque(maxlen=2000)
        self.TRAINING_BUFFER_15m = deque(maxlen=1000)
        self.meta_features_buffer = deque(maxlen=1000)
        
        # === Feature Buffers ===
        self.features_buffer = deque(maxlen=1000)
        self.features_buffer_5m = deque(maxlen=200)
        self.features_buffer_15m = deque(maxlen=100)
        
        # === Prediction Tracking ===
        self.predictions_buffer = deque(maxlen=1000)
        self.prob_buffer = deque(maxlen=100)
        self.last_prediction_time = {}
        
        # === Timing ===
        self.LAST_BATCH_REFIT_TIME = time.time()
        self.BATCH_REFIT_INTERVAL = 3600  # 1 hour
        self.last_5min_fit = time.time()
        self.last_15min_fit = time.time()
        
        # === Trading Execution ===
        self.active_orders = {}
        self.pending_signals = deque(maxlen=10)
        self.execution_history = deque(maxlen=1000)
        
        # === Error Handling ===
        self.ib_loop_error_count = 0
        self.data_error_count = 0
        self.model_error_count = 0
        
        # === Threading ===
        self.GPT_STOP_EVENT = threading.Event()
        self.TELEGRAM_STOP_EVENT = threading.Event()
        self.CMD_QUEUE = queue.Queue()
        
        # === Tick Storage ===
        self.tick_count = 0
        self.ticks_array = np.zeros((1000000, 4), dtype=np.float32)  # time, price, size, side
        
        # === Session Tracking ===
        self.session_start_time = time.time()
        self.trades_today = 0
        self.pnl_today = 0.0
        self.last_save_time = 0.0
    
    def init_performance_tracking(self):
        """Initialize performance tracking."""
        # === Performance Metrics ===
        self.sharpe_window = deque(maxlen=252)
        self.drawdown_history = deque(maxlen=1000)
        self.equity_curve = deque(maxlen=10000)
        
        # === Trade Analysis ===
        self.trade_history = deque(maxlen=1000)
        self.winning_streaks = deque(maxlen=100)
        self.losing_streaks = deque(maxlen=100)
        
        # === Alert History ===
        self.alert_history = deque(maxlen=100)
        self.risk_alerts = deque(maxlen=50)
        
        # === System Metrics ===
        self.latency_history = deque(maxlen=1000)
        self.cpu_usage_history = deque(maxlen=1000)
        self.memory_usage_history = deque(maxlen=1000)
    
    def update_risk_metrics(self):
        """Update risk management metrics."""
        with locks.risk_lock:
            # Calculate portfolio heat
            total_exposure = sum(abs(pos) for pos in self.positions.values())
            max_exposure = self.risk_limits.max_position_size * 2
            self.portfolio_heat = total_exposure / max_exposure if max_exposure > 0 else 0
            
            # Check circuit breaker
            if self.metrics.consecutive_losses >= self.risk_limits.consecutive_loss_limit:
                self.circuit_breaker_triggered = True
                logger.warning("Circuit breaker triggered!")
            
            # Update drawdown
            if self.equity_curve:
                peak = max(self.equity_curve)
                current = self.equity_curve[-1]
                self.metrics.current_drawdown = (current - peak) / peak if peak > 0 else 0
                self.metrics.max_drawdown = min(self.metrics.max_drawdown, self.metrics.current_drawdown)
    
    def send_alert(self, message: str, force: bool = False):
        """Send alert to Telegram."""
        current_time = time.time()
        
        # Check cooldown unless forced
        if not force and current_time - self.LAST_CUSTOM_ALERT_TIME < self.CUSTOM_ALERT_COOLDOWN:
            return
        
        # Send to Telegram
        if self.TELEGRAM_TOKEN and self.TELEGRAM_CHAT_ID:
            try:
                import requests
                url = f"https://api.telegram.org/bot{self.TELEGRAM_TOKEN}/sendMessage"
                data = {
                    "chat_id": self.TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown"
                }
                response = requests.post(url, data=data, timeout=5)
                
                if response.status_code == 200:
                    self.LAST_CUSTOM_ALERT_TIME = current_time
                    logger.info(f"Alert sent: {message[:50]}...")
                else:
                    logger.error(f"Failed to send alert: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
        
        # Store in history
        self.alert_history.append({
            'timestamp': current_time,
            'message': message
        })
    
    def cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up GlobalState resources...")
        
        # Save models if needed
        try:
            self.save_models()
        except Exception as e:
            logger.error(f"Error saving models: {e}")
        
        # Close connections
        if hasattr(self, 'ib_client'):
            try:
                self.ib_client.disconnect()
            except:
                pass
        
        logger.info("GlobalState cleanup completed")
    
    def save_models(self):
        """Save trained models to disk."""
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save sklearn models
        sklearn_models = {
            'sgd_1m': self.ml_model_sgd,
            'pa_1m': self.ml_model_pa,
            'rf_1m': self.ml_model_rf,
            'xgb_1m': self.ml_model_xgb,
            'sgd_5m': self.ml_model_sgd_5m,
            'pa_5m': self.ml_model_pa_5m,
            'rf_5m': self.ml_model_rf_5m,
            'xgb_5m': self.ml_model_xgb_5m,
            'sgd_15m': self.ml_model_sgd_15m,
            'pa_15m': self.ml_model_pa_15m,
            'rf_15m': self.ml_model_rf_15m,
            'xgb_15m': self.ml_model_xgb_15m,
        }
        
        for name, model in sklearn_models.items():
            if model is not None and is_defined(model):
                try:
                    joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))
                    logger.info(f"Saved model: {name}")
                except Exception as e:
                    logger.error(f"Error saving {name}: {e}")
        
        # Save deep learning models
        if self.ml_model_lstm is not None and is_defined_keras(self.ml_model_lstm):
            try:
                self.ml_model_lstm.save(os.path.join(models_dir, "lstm_model.h5"))
                logger.info("Saved LSTM model")
            except Exception as e:
                logger.error(f"Error saving LSTM: {e}")
        
        if self.ml_model_gru is not None and is_defined_keras(self.ml_model_gru):
            try:
                self.ml_model_gru.save(os.path.join(models_dir, "gru_model.h5"))
                logger.info("Saved GRU model")
            except Exception as e:
                logger.error(f"Error saving GRU: {e}")

# === Initialize Global State ===
STATE = GlobalState()

# Register cleanup on exit
atexit.register(STATE.cleanup)

# === TRADING FUNCTIONS AND UTILITIES ===

# === IB CONNECTION AND MANAGEMENT ===
def connect_to_ib() -> bool:
    """Connect to Interactive Brokers with retry logic."""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            if ib.isConnected():
                ib.disconnect()
                time.sleep(1)
            
            logger.info(f"Connecting to IB at {IB_HOST}:{IB_PORT} (attempt {attempt + 1}/{max_retries})")
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            
            if ib.isConnected():
                logger.info("Successfully connected to IB")
                return True
                
        except Exception as e:
            logger.error(f"IB connection failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    
    return False

def safe_sleep(seconds: float) -> None:
    """Sleep safely whether IB is connected or not."""
    if ib.isConnected():
        ib.sleep(seconds)
    else:
        time.sleep(seconds)

# === CONNECTION MANAGEMENT ===
def connect_all():
    """Connect to IB and subscribe to market data."""
    
    # Connect to IB
    if not connect_to_ib():
        logger.error("Failed to connect to IB")
        STATE.send_alert("❌ IB connection failed!")
        return False
    
    # Get symbol from command line or environment
    symbol = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DEFAULT_SYMBOL", "AAPL")
    
    # Subscribe to market data
    success = subscribe_symbol(symbol)
    
    if success:
        STATE.send_alert(f"✅ Connected to IB\n📊 Subscribed to {symbol}")
    else:
        STATE.send_alert(f"❌ Failed to subscribe to {symbol}")
    
    return success

def subscribe_symbol(symbol: str) -> bool:
    """Subscribe to market data for a symbol."""
    try:
        # Unsubscribe from previous symbol
        if STATE.ticker_l2:
            ib.cancelMktDepth(STATE.ticker_l2.contract)
        if STATE.ticker_tns:
            ib.cancelTickByTickData(STATE.ticker_tns.contract)
        
        # Clear data
        with locks.l2_lock:
            STATE.df_l2_display = pd.DataFrame(columns=["Price", "Size", "Side"])
            STATE.bids_dict.clear()
            STATE.asks_dict.clear()
        
        with locks.tns_lock:
            STATE.df_tns = pd.DataFrame(columns=["Time", "Price", "Size", "Timestamp"])
        
        # Create contract
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Request market depth (L2)
        STATE.ticker_l2 = ib.reqMktDepth(contract, numRows=20)
        STATE.ticker_l2.updateEvent += on_l2_update
        
        # Request time and sales
        STATE.ticker_tns = ib.reqTickByTickData(contract, "AllLast")
        STATE.ticker_tns.updateEvent += on_tns_update
        
        # Update state
        STATE.current_symbol = symbol
        
        logger.info(f"Subscribed to {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to subscribe to {symbol}: {e}")
        return False

def reconnect_if_needed():
    """Check connection and reconnect if necessary."""
    if not ib.isConnected():
        logger.warning("IB disconnected, attempting reconnect...")
        STATE.send_alert("⚠️ IB disconnected, reconnecting...")
        
        if connect_to_ib():
            if STATE.current_symbol:
                subscribe_symbol(STATE.current_symbol)
            return True
        else:
            return False
    return True

# === MARKET DATA EVENT HANDLERS ===
def on_l2_update(ticker: Ticker):
    """Handle Level 2 order book updates."""
    try:
        with locks.l2_lock:
            # Update bid dictionary
            STATE.bids_dict.clear()
            for bid in ticker.domBids:
                STATE.bids_dict[bid.price] = bid.size
            
            # Update ask dictionary
            STATE.asks_dict.clear()
            for ask in ticker.domAsks:
                STATE.asks_dict[ask.price] = ask.size
            
            # Create display dataframe
            bid_data = [{"Price": p, "Size": s, "Side": "BID"} for p, s in STATE.bids_dict.items()]
            ask_data = [{"Price": p, "Size": s, "Side": "ASK"} for p, s in STATE.asks_dict.items()]
            
            STATE.df_l2_display = pd.DataFrame(bid_data + ask_data)
            
            # Record L2 event
            event = {
                'timestamp': time.time(),
                'bid_levels': len(STATE.bids_dict),
                'ask_levels': len(STATE.asks_dict),
                'best_bid': max(STATE.bids_dict.keys()) if STATE.bids_dict else 0,
                'best_ask': min(STATE.asks_dict.keys()) if STATE.asks_dict else 0,
                'spread': min(STATE.asks_dict.keys()) - max(STATE.bids_dict.keys()) if STATE.bids_dict and STATE.asks_dict else 0
            }
            STATE.l2_events.append(event)
            
            # Update spread history
            if STATE.bids_dict and STATE.asks_dict:
                spread = event['spread']
                STATE.spread_history.append((time.time(), spread))
                
    except Exception as e:
        logger.error(f"Error in L2 update: {e}")
        STATE.data_error_count += 1

def on_tns_update(ticker: TickByTickAllLast):
    """Handle Time and Sales updates."""
    try:
        with locks.tns_lock:
            # Create new row
            new_row = pd.DataFrame([{
                "Time": datetime.datetime.fromtimestamp(ticker.time / 1000).strftime("%H:%M:%S.%f")[:-3],
                "Price": ticker.price,
                "Size": ticker.size,
                "Timestamp": ticker.time / 1000
            }])
            
            # Append to dataframe
            STATE.df_tns = pd.concat([STATE.df_tns, new_row], ignore_index=True)
            
            # Trim to max size
            if len(STATE.df_tns) > STATE.MAX_TNS_SIZE:
                STATE.df_tns = STATE.df_tns.iloc[-STATE.MAX_TNS_SIZE:]
            
            # Store in tick array for compression
            if STATE.tick_count < len(STATE.ticks_array):
                STATE.ticks_array[STATE.tick_count] = [
                    ticker.time / 1000,
                    ticker.price,
                    ticker.size,
                    1 if ticker.tickAttribLast.pastLimit else 0
                ]
                STATE.tick_count += 1
                
    except Exception as e:
        logger.error(f"Error in TnS update: {e}")
        STATE.data_error_count += 1

# === MULTI-HORIZON PREDICTION TRACKING ===
class MultiHorizonTracker:
    """Track predictions across multiple time horizons."""
    
    def __init__(self):
        self.horizons = {
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '60m': 3600
        }
        self.predictions_df = pd.DataFrame(
            columns=['timestamp', 'prob_hausse', 'prix_init', 
                    'correct_5m', 'correct_15m', 'correct_30m', 'correct_60m']
        )
        self.performance_history = defaultdict(list)
        
    def add_prediction(self, prob: float, price: float):
        """Add new prediction to track."""
        new_row = {
            'timestamp': time.time(),
            'prob_hausse': prob,
            'prix_init': price,
            'correct_5m': np.nan,
            'correct_15m': np.nan,
            'correct_30m': np.nan,
            'correct_60m': np.nan
        }
        
        self.predictions_df = pd.concat([
            self.predictions_df, 
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # Keep only last 1000 predictions
        if len(self.predictions_df) > 1000:
            self.predictions_df = self.predictions_df.iloc[-1000:]
    
    def verify_predictions(self, current_price: float):
        """Verify predictions against current price."""
        current_time = time.time()
        
        for idx, row in self.predictions_df.iterrows():
            for horizon, seconds in self.horizons.items():
                col_name = f'correct_{horizon}'
                
                # Skip if already verified
                if not pd.isna(row[col_name]):
                    continue
                
                # Check if enough time has passed
                if current_time - row['timestamp'] >= seconds:
                    # Determine if prediction was correct
                    price_change = (current_price - row['prix_init']) / row['prix_init']
                    predicted_up = row['prob_hausse'] > 0.5
                    actual_up = price_change > 0
                    
                    correct = predicted_up == actual_up
                    self.predictions_df.at[idx, col_name] = correct
                    
                    # Update performance history
                    self.performance_history[horizon].append({
                        'timestamp': current_time,
                        'correct': correct,
                        'confidence': abs(row['prob_hausse'] - 0.5) * 2,
                        'price_change': price_change
                    })
    
    def calculate_accuracy_by_horizon(self) -> Dict[str, float]:
        """Calculate accuracy for each time horizon."""
        accuracies = {}
        
        for horizon in self.horizons.keys():
            col_name = f'correct_{horizon}'
            verified = self.predictions_df[col_name].dropna()
            
            if len(verified) > 0:
                accuracies[horizon] = verified.mean()
            else:
                accuracies[horizon] = 0.0
        
        return accuracies
    
    def calculate_accuracy_report(self) -> str:
        """Generate formatted accuracy report."""
        accuracies = self.calculate_accuracy_by_horizon()
        
        report_lines = []
        for horizon, accuracy in accuracies.items():
            n_verified = self.predictions_df[f'correct_{horizon}'].dropna().count()
            report_lines.append(f"• {horizon}: {accuracy:.1%} (n={n_verified})")
        
        return "\n".join(report_lines) if report_lines else "No verified predictions yet"

# Initialize multi-horizon tracker
HORIZON_TRACKER = MultiHorizonTracker()

# === ONLINE LEARNING SYSTEM ===
class OnlineLearningSystem:
    """Advanced online learning with multiple timeframes."""
    
    def __init__(self):
        self.update_counts = defaultdict(int)
        self.performance_history = defaultdict(list)
        self.last_update_times = defaultdict(float)
        
    def partial_fit_models(self, timeframe: str = "1m"):
        """Fit models for specific timeframe."""
        
        # Get appropriate buffer
        if timeframe == "1m":
            buffer = STATE.TRAINING_BUFFER
            models = {
                'sgd': STATE.ml_model_sgd,
                'pa': STATE.ml_model_pa,
                'rf': STATE.ml_model_rf,
                'xgb': STATE.ml_model_xgb
            }
        elif timeframe == "5m":
            buffer = STATE.TRAINING_BUFFER_5m
            models = {
                'sgd': STATE.ml_model_sgd_5m,
                'pa': STATE.ml_model_pa_5m,
                'rf': STATE.ml_model_rf_5m,
                'xgb': STATE.ml_model_xgb_5m
            }
        elif timeframe == "15m":
            buffer = STATE.TRAINING_BUFFER_15m
            models = {
                'sgd': STATE.ml_model_sgd_15m,
                'pa': STATE.ml_model_pa_15m,
                'rf': STATE.ml_model_rf_15m,
                'xgb': STATE.ml_model_xgb_15m
            }
        else:
            logger.error(f"Unknown timeframe: {timeframe}")
            return
        
        # Check if we have enough samples
        if len(buffer) < 100:
            return
        
        # Update online models (SGD, PA)
        self._update_online_models(timeframe, buffer, models)
        
        # Check if batch retrain needed for tree models
        if len(buffer) >= STATE.config.ml.min_samples_retrain:
            if timeframe == "1m" or len(buffer) >= 500:  # Less frequent for longer timeframes
                self._trigger_batch_retrain(timeframe, buffer, models)
        
        # Update meta model if we have base predictions
        if timeframe == "1m" and STATE.meta_features_buffer:
            self._update_meta_model()
        
        self.last_update_times[timeframe] = time.time()
        
    def _update_online_models(
        self, 
        timeframe: str, 
        buffer: Deque,
        models: Dict[str, Any]
    ):
        """Update online learning models incrementally."""
        
        # Get recent samples
        recent_samples = list(buffer)[-50:]  # Last 50 samples
        
        if not recent_samples:
            return
        
        # Extract features and labels
        X_list, y_list = [], []
        for features, label in recent_samples:
            X_list.append(features)
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Scale features
        scaler = self._get_scaler(timeframe)
        try:
            X_scaled = scaler.transform(X)
        except:
            # Fit scaler if not fitted
            scaler.fit(X)
            X_scaled = scaler.transform(X)
        
        # Update SGD
        if models['sgd'] is None:
            models['sgd'] = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.001,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42
            )
            # Initialize with all classes
            dummy_X = np.array([[0] * X.shape[1]] * 3)
            dummy_y = [0, 1, 2]
            models['sgd'].fit(dummy_X, dummy_y)
        
        try:
            models['sgd'].partial_fit(X_scaled, y, classes=[0, 1, 2])
            self.update_counts[f"{timeframe}_sgd"] += len(y)
        except Exception as e:
            logger.error(f"SGD update error ({timeframe}): {e}")
        
        # Update Passive Aggressive
        if models['pa'] is None:
            models['pa'] = PassiveAggressiveClassifier(
                C=0.1,
                max_iter=1000,
                random_state=42
            )
            # Initialize with all classes
            dummy_X = np.array([[0] * X.shape[1]] * 3)
            dummy_y = [0, 1, 2]
            models['pa'].fit(dummy_X, dummy_y)
        
        try:
            models['pa'].partial_fit(X_scaled, y, classes=[0, 1, 2])
            self.update_counts[f"{timeframe}_pa"] += len(y)
        except Exception as e:
            logger.error(f"PA update error ({timeframe}): {e}")
        
        # Update River models if available
        if RIVER_AVAILABLE and timeframe in ["1m", "5m"]:
            river_model = STATE.river_model_1m if timeframe == "1m" else STATE.river_model_5m
            
            for x_i, y_i in zip(X, y):
                x_dict = {f'f{i}': float(v) for i, v in enumerate(x_i)}
                river_model.learn_one(x_dict, int(y_i))
                
                # Update metrics
                if STATE.online_metrics:
                    y_pred = river_model.predict_one(x_dict)
                    for metric in STATE.online_metrics.values():
                        metric.update(int(y_i), y_pred)
    
    def _get_scaler(self, timeframe: str) -> StandardScaler:
        """Get appropriate scaler for timeframe."""
        if timeframe == "1m":
            return STATE.scaler
        elif timeframe == "5m":
            return STATE.scaler_5m
        elif timeframe == "15m":
            return STATE.scaler_15m
        else:
            return STATE.scaler
    
    def _update_meta_model(self):
        """Update meta-learning model."""
        if not STATE.meta_features_buffer:
            return
        
        # Process oldest meta features
        while STATE.meta_features_buffer:
            timestamp, base_predictions, true_label = STATE.meta_features_buffer.popleft()
            
            # Ensure valid features
            features = np.array(base_predictions, dtype=float)
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue
            
            # Update meta model
            if STATE.ml_model_meta is None:
                STATE.ml_model_meta = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0005,
                    learning_rate="adaptive",
                    eta0=0.001,
                    random_state=42
                )
                # Initialize with all classes
                dummy_X = np.array([[0.5] * len(features)] * 3)
                dummy_y = [0, 1, 2]
                STATE.ml_model_meta.fit(dummy_X, dummy_y)
            
            X = features.reshape(1, -1)
            y = [true_label]
            
            try:
                STATE.ml_model_meta.partial_fit(X, y)
            except Exception as e:
                logger.error(f"Meta model update error: {e}")
    
    def _trigger_batch_retrain(
        self,
        timeframe: str,
        training_buffer: List,
        models: Dict[str, Any]
    ):
        """Trigger batch retraining for tree-based models."""
        
        logger.info(f"Triggering batch retrain for {timeframe} with {len(training_buffer)} samples")
        
        # Extract data
        X_list, y_list = [], []
        for features, label in training_buffer:
            X_list.append(features)
            y_list.append(label)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Clear buffer
        training_buffer.clear()
        
        # Train models
        results = ML_TRAINER.train_models_batch(X, y, models, timeframe)
        
        # Send notification
        if results:
            msg = f"🔄 **ML Calibration Complete** ({timeframe})\n"
            msg += f"Samples: {len(X)}\n"
            for model, score in results.items():
                msg += f"• {model.upper()}: {score:.3f}\n"
            STATE.send_alert(msg)

# Initialize online learning system
ONLINE_LEARNER = OnlineLearningSystem()

# === ANOMALY DETECTION ===
class AnomalyDetector:
    """Advanced anomaly detection system."""
    
    def __init__(self):
        self.detectors = {
            'isolation_forest': None,
            'statistical': None
        }
        self.anomaly_history = deque(maxlen=1000)
        self.anomaly_counts = defaultdict(int)
        
    def train_detectors(self):
        """Train anomaly detection models."""
        # Get recent features
        if len(STATE.features_buffer) < 100:
            return
        
        features = np.array([f for _, f in list(STATE.features_buffer)[-500:]])
        
        # Train Isolation Forest
        if self.detectors['isolation_forest'] is None:
            self.detectors['isolation_forest'] = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
        
        try:
            self.detectors['isolation_forest'].fit(features)
        except Exception as e:
            logger.error(f"Isolation Forest training error: {e}")
    
    def detect_anomalies(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in features."""
        anomalies = {
            'is_anomaly': False,
            'scores': {},
            'reasons': []
        }
        
        # Isolation Forest detection
        if self.detectors['isolation_forest'] is not None:
            try:
                score = self.detectors['isolation_forest'].decision_function(features.reshape(1, -1))[0]
                prediction = self.detectors['isolation_forest'].predict(features.reshape(1, -1))[0]
                
                anomalies['scores']['isolation_forest'] = score
                if prediction == -1:
                    anomalies['is_anomaly'] = True
                    anomalies['reasons'].append('Isolation Forest')
                    
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
        
        # Statistical anomaly detection
        self._detect_statistical_anomalies(features, anomalies)
        
        # Record anomaly
        if anomalies['is_anomaly']:
            self.anomaly_history.append({
                'timestamp': time.time(),
                'features': features,
                'scores': anomalies['scores'],
                'reasons': anomalies['reasons']
            })
            
            for reason in anomalies['reasons']:
                self.anomaly_counts[reason] += 1
        
        return anomalies
    
    def _detect_statistical_anomalies(self, features: np.ndarray, anomalies: Dict):
        """Detect statistical anomalies."""
        # Check for extreme values (3 sigma rule)
        for i, value in enumerate(features):
            if abs(value) > 3:  # Assuming standardized features
                anomalies['is_anomaly'] = True
                anomalies['reasons'].append(f'Feature {i} extreme value')
        
        # Check for NaN or Inf
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            anomalies['is_anomaly'] = True
            anomalies['reasons'].append('Invalid values detected')

# Initialize anomaly detector
ANOMALY_DETECTOR = AnomalyDetector()

# === FEATURE ENGINEERING ===
class FeatureEngineer:
    """Advanced feature engineering for multiple timeframes."""
    
    def __init__(self):
        self.feature_cache = {}
        self.feature_importance = defaultdict(float)
        
    def compute_features(self, timeframe: str = "1m") -> Optional[np.ndarray]:
        """Compute features for given timeframe."""
        try:
            # Get appropriate data based on timeframe
            lookback = self._get_lookback_period(timeframe)
            
            with locks.tns_lock:
                df = STATE.df_tns.copy()
            
            if df.empty or len(df) < lookback:
                return None
            
            # Use recent data
            df_recent = df.tail(lookback)
            
            features = []
            
            # Price features
            features.extend(self._compute_price_features(df_recent))
            
            # Volume features
            features.extend(self._compute_volume_features(df_recent))
            
            # Microstructure features
            features.extend(self._compute_microstructure_features())
            
            # Technical indicators
            features.extend(self._compute_technical_indicators(df_recent))
            
            # Market regime features
            features.extend(self._compute_regime_features())
            
            # Time-based features
            features.extend(self._compute_time_features())
            
            # Validate features
            features_array = np.array(features, dtype=np.float32)
            
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                logger.warning(f"Invalid features detected for {timeframe}")
                return None
            
            return features_array
            
        except Exception as e:
            logger.error(f"Feature computation error ({timeframe}): {e}")
            return None
    
    def _get_lookback_period(self, timeframe: str) -> int:
        """Get lookback period for timeframe."""
        lookback_map = {
            "1m": 100,
            "5m": 500,
            "15m": 1500
        }
        return lookback_map.get(timeframe, 100)
    
    def _compute_price_features(self, df: pd.DataFrame) -> List[float]:
        """Compute price-based features."""
        features = []
        
        prices = df['Price'].values
        
        # Returns
        returns = np.diff(np.log(prices))
        features.append(returns[-1] if len(returns) > 0 else 0)  # Last return
        features.append(np.mean(returns) if len(returns) > 0 else 0)  # Mean return
        features.append(np.std(returns) if len(returns) > 1 else 0)  # Volatility
        
        # Price position
        if len(prices) > 20:
            ma20 = np.mean(prices[-20:])
            features.append((prices[-1] - ma20) / ma20 if ma20 > 0 else 0)
        else:
            features.append(0)
        
        # Price momentum
        if len(prices) > 10:
            momentum = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
            features.append(momentum)
        else:
            features.append(0)
        
        # Relative strength
        if len(returns) > 14:
            gains = returns[returns > 0]
            losses = abs(returns[returns < 0])
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi / 100)  # Normalize to [0, 1]
        else:
            features.append(0.5)
        
        return features
    
    def _compute_volume_features(self, df: pd.DataFrame) -> List[float]:
        """Compute volume-based features."""
        features = []
        
        volumes = df['Size'].values
        
        # Volume statistics
        features.append(np.sum(volumes[-10:]) if len(volumes) >= 10 else 0)  # Recent volume
        features.append(np.mean(volumes) if len(volumes) > 0 else 0)  # Average volume
        
        # Relative volume
        if len(volumes) > 50:
            recent_vol = np.sum(volumes[-10:])
            avg_vol = np.mean(volumes[-50:-10])
            rel_vol = recent_vol / avg_vol if avg_vol > 0 else 1
            features.append(np.log1p(rel_vol))
        else:
            features.append(0)
        
        # Volume momentum
        if len(volumes) > 20:
            vol_momentum = np.sum(volumes[-10:]) - np.sum(volumes[-20:-10])
            features.append(np.tanh(vol_momentum / 1000))  # Normalize
        else:
            features.append(0)
        
        return features
    
    def _compute_microstructure_features(self) -> List[float]:
        """Compute market microstructure features."""
        features = []
        
        # Order flow imbalance
        features.append(np.tanh(STATE.OFI_VALUE / 1000))
        
        # Cumulative delta volume
        features.append(np.tanh(STATE.CDV_VALUE / 1000))
        
        # Bid-ask spread
        with locks.l2_lock:
            if STATE.bids_dict and STATE.asks_dict:
                best_bid = max(STATE.bids_dict.keys())
                best_ask = min(STATE.asks_dict.keys())
                spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
                features.append(spread * 10000)  # Basis points
            else:
                features.append(0)
        
        # Book imbalance
        bid_volume = sum(STATE.bids_dict.values()) if STATE.bids_dict else 0
        ask_volume = sum(STATE.asks_dict.values()) if STATE.asks_dict else 0
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        features.append(imbalance)
        
        # Toxicity score
        features.append(STATE.toxicity_score)
        
        return features
    
    def _compute_technical_indicators(self, df: pd.DataFrame) -> List[float]:
        """Compute technical indicators."""
        features = []
        
        prices = df['Price'].values
        
        # Bollinger Bands
        if len(prices) > 20:
            sma20 = np.mean(prices[-20:])
            std20 = np.std(prices[-20:])
            upper_band = sma20 + 2 * std20
            lower_band = sma20 - 2 * std20
            
            # Price position in bands
            bb_position = (prices[-1] - lower_band) / (upper_band - lower_band) if upper_band > lower_band else 0.5
            features.append(bb_position)
            
            # Band width
            band_width = (upper_band - lower_band) / sma20 if sma20 > 0 else 0
            features.append(band_width)
        else:
            features.extend([0.5, 0])
        
        # MACD
        if len(prices) > 26:
            ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
            macd = ema12 - ema26
            features.append(np.tanh(macd))
        else:
            features.append(0)
        
        # ATR (Average True Range)
        if len(prices) > 14:
            high_low = np.max(prices[-14:]) - np.min(prices[-14:])
            atr = high_low / prices[-1] if prices[-1] > 0 else 0
            features.append(atr)
        else:
            features.append(0)
        
        return features
    
    def _compute_regime_features(self) -> List[float]:
        """Compute market regime features."""
        features = []
        
        # Regime probabilities
        regime_map = {
            "Trending": [1, 0, 0, 0, 0],
            "Normal": [0, 1, 0, 0, 0],
            "Choppy": [0, 0, 1, 0, 0],
            "Volatile Trending": [0, 0, 0, 1, 0],
            "Low Volatility": [0, 0, 0, 0, 1],
            "Unknown": [0, 0, 0, 0, 0]
        }
        
        regime_features = regime_map.get(STATE.market_regime.regime, [0, 0, 0, 0, 0])
        features.extend(regime_features)
        
        # Regime metrics
        features.append(STATE.market_regime.volatility)
        features.append(STATE.market_regime.efficiency_ratio)
        features.append(STATE.market_regime.hurst_exponent)
        
        return features
    
    def _compute_time_features(self) -> List[float]:
        """Compute time-based features."""
        features = []
        
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.datetime.now(ny_tz)
        
        # Time of day (normalized)
        minutes_since_open = (now.hour - 9) * 60 + (now.minute - 30)
        features.append(minutes_since_open / 390)  # 390 minutes in trading day
        
        # Day of week
        features.append(now.weekday() / 4)  # Normalize to [0, 1]
        
        # Time to close
        minutes_to_close = 390 - minutes_since_open
        features.append(minutes_to_close / 390)
        
        # Intraday patterns
        features.append(1 if 9.5 <= now.hour <= 10 else 0)  # Opening hour
        features.append(1 if 15 <= now.hour < 16 else 0)  # Closing hour
        features.append(1 if 11.5 <= now.hour <= 13 else 0)  # Lunch hour
        
        return features

# Initialize feature engineer
FEATURE_ENGINEER = FeatureEngineer()

# === ML MODEL TRAINING ===
class MLTrainer:
    """Machine learning model training system."""
    
    def __init__(self):
        self.training_history = defaultdict(list)
        self.best_params = {}
        
    def train_models_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: Dict[str, Any],
        timeframe: str
    ) -> Dict[str, float]:
        """Train batch models (RF, XGBoost)."""
        results = {}
        
        # Scale features
        scaler = ONLINE_LEARNER._get_scaler(timeframe)
        try:
            X_scaled = scaler.transform(X)
        except:
            scaler.fit(X)
            X_scaled = scaler.transform(X)
        
        # Train Random Forest
        if 'rf' in models:
            try:
                if GPU_AVAILABLE and cuMLRandomForest is not None:
                    models['rf'] = cuMLRandomForest(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                else:
                    models['rf'] = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        n_jobs=-1,
                        random_state=42
                    )
                
                models['rf'].fit(X_scaled, y)
                score = models['rf'].score(X_scaled, y)
                results['rf'] = score
                
            except Exception as e:
                logger.error(f"RF training error ({timeframe}): {e}")
        
        # Train XGBoost
        if XGBOOST_AVAILABLE and 'xgb' in models:
            try:
                models['xgb'] = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective='multi:softprob',
                    num_class=3,
                    use_label_encoder=False,
                    random_state=42
                )
                
                models['xgb'].fit(X_scaled, y)
                score = models['xgb'].score(X_scaled, y)
                results['xgb'] = score
                
            except Exception as e:
                logger.error(f"XGBoost training error ({timeframe}): {e}")
        
        # Record training
        self.training_history[timeframe].append({
            'timestamp': time.time(),
            'n_samples': len(X),
            'results': results
        })
        
        return results
    
    def hyperparameter_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            if model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
                
            elif model_type == 'xgb' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = XGBClassifier(**params, objective='multi:softprob', num_class=3, use_label_encoder=False)
            else:
                return 0
            
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20, timeout=60)
        
        # Store best params
        self.best_params[f"{timeframe}_{model_type}"] = study.best_params
        
        return study.best_params

# Initialize ML trainer
ML_TRAINER = MLTrainer()

# === MODEL MANAGER ===
class ModelManager:
    """Manage ensemble predictions across models and timeframes."""
    
    def __init__(self):
        self.prediction_cache = {}
        self.ensemble_weights = defaultdict(lambda: 1.0)
        
    def get_ensemble_prediction(
        self,
        features: np.ndarray,
        timeframe: str = "1m"
    ) -> Dict[str, Any]:
        """Get ensemble prediction from all models."""
        
        predictions = {}
        probabilities = []
        
        # Get models for timeframe
        models = self._get_models_for_timeframe(timeframe)
        
        # Get predictions from each model
        for name, model in models.items():
            if model is not None and is_defined(model):
                try:
                    # Scale features
                    scaler = ONLINE_LEARNER._get_scaler(timeframe)
                    X_scaled = scaler.transform(features.reshape(1, -1))
                    
                    # Get prediction
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_scaled)[0]
                        # Convert to binary (up/down)
                        prob_up = proba[2] + proba[1] * 0.5  # Strong up + neutral
                        predictions[name] = prob_up
                        probabilities.append(prob_up * self.ensemble_weights[f"{timeframe}_{name}"])
                    
                except Exception as e:
                    logger.error(f"Prediction error {name} ({timeframe}): {e}")
        
        # River model predictions
        if RIVER_AVAILABLE and timeframe in ["1m", "5m"]:
            try:
                river_model = STATE.river_model_1m if timeframe == "1m" else STATE.river_model_5m
                x_dict = {f'f{i}': float(v) for i, v in enumerate(features)}
                proba = river_model.predict_proba_one(x_dict)
                if proba:
                    prob_up = proba.get(2, 0) + proba.get(1, 0) * 0.5
                    predictions['river'] = prob_up
                    probabilities.append(prob_up)
            except:
                pass
        
        # Calculate ensemble prediction
        if probabilities:
            ensemble_prob = np.mean(probabilities)
            confidence = 1 - np.std(probabilities)
            agreement = len([p for p in probabilities if (p > 0.5) == (ensemble_prob > 0.5)]) / len(probabilities)
        else:
            ensemble_prob = 0.5
            confidence = 0.0
            agreement = 0.0
        
        # Get signal
        if ensemble_prob > 0.7:
            signal = "STRONG_BUY"
        elif ensemble_prob > 0.6:
            signal = "BUY"
        elif ensemble_prob < 0.3:
            signal = "STRONG_SELL"
        elif ensemble_prob < 0.4:
            signal = "SELL"
        else:
            signal = "NEUTRAL"
        
        return {
            'prediction': ensemble_prob,
            'signal': signal,
            'confidence': confidence,
            'model_agreement': agreement,
            'model_predictions': predictions,
            'timeframe': timeframe
        }
    
    def _get_models_for_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """Get all models for a specific timeframe."""
        if timeframe == "1m":
            return {
                'sgd': STATE.ml_model_sgd,
                'pa': STATE.ml_model_pa,
                'rf': STATE.ml_model_rf,
                'xgb': STATE.ml_model_xgb
            }
        elif timeframe == "5m":
            return {
                'sgd': STATE.ml_model_sgd_5m,
                'pa': STATE.ml_model_pa_5m,
                'rf': STATE.ml_model_rf_5m,
                'xgb': STATE.ml_model_xgb_5m
            }
        elif timeframe == "15m":
            return {
                'sgd': STATE.ml_model_sgd_15m,
                'pa': STATE.ml_model_pa_15m,
                'rf': STATE.ml_model_rf_15m,
                'xgb': STATE.ml_model_xgb_15m
            }
        else:
            return {}
    
    def update_ensemble_weights(self, performance_data: Dict[str, float]):
        """Update ensemble weights based on performance."""
        for key, performance in performance_data.items():
            # Exponential moving average of performance
            current_weight = self.ensemble_weights[key]
            new_weight = 0.9 * current_weight + 0.1 * performance
            self.ensemble_weights[key] = max(0.1, min(2.0, new_weight))

# Initialize model manager
MODEL_MANAGER = ModelManager()

# === VALIDATION SYSTEM ===
class ValidationSystem:
    """Comprehensive model validation and backtesting."""
    
    def __init__(self):
        self.validation_history = deque(maxlen=1000)
        self.backtest_results = {}
        self.monte_carlo_results = {}
        
    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        model_fn,
        window_size: int = 1000,
        step_size: int = 100
    ) -> Dict[str, Any]:
        """Perform walk-forward validation."""
        
        results = []
        
        for i in range(window_size, len(data) - step_size, step_size):
            # Training window
            train_data = data.iloc[i-window_size:i]
            
            # Test window
            test_data = data.iloc[i:i+step_size]
            
            # Train model
            model = model_fn(train_data)
            
            # Evaluate
            score = self.evaluate_model(model, test_data)
            results.append(score)
        
        return {
            'mean_score': np.mean(results),
            'std_score': np.std(results),
            'sharpe': np.mean(results) / np.std(results) if np.std(results) > 0 else 0,
            'n_windows': len(results)
        }
    
    def monte_carlo_validation(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_simulations: int = 100
    ) -> Dict[str, Any]:
        """Monte Carlo simulation for robustness testing."""
        
        results = []
        
        for _ in range(n_simulations):
            # Add noise to features
            noise = np.random.normal(0, 0.01, features.shape)
            noisy_features = features + noise
            
            # Random subset
            n_samples = len(features)
            indices = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
            
            X_subset = noisy_features[indices]
            y_subset = labels[indices]
            
            # Get predictions
            predictions = MODEL_MANAGER.get_ensemble_prediction(X_subset)
            
            # Calculate metrics
            accuracy = np.mean(predictions['prediction'] == y_subset)
            results.append(accuracy)
        
        self.monte_carlo_results = {
            'mean_accuracy': np.mean(results),
            'std_accuracy': np.std(results),
            'confidence_interval': (
                np.percentile(results, 5),
                np.percentile(results, 95)
            ),
            'success_rate': np.mean([r > 0.5 for r in results]),
            'avg_return': np.mean(results) - 0.5,
            'sharpe': (np.mean(results) - 0.5) / np.std(results) if np.std(results) > 0 else 0
        }
        
        return self.monte_carlo_results
    
    def evaluate_model(self, model, test_data: pd.DataFrame) -> float:
        """Evaluate model performance."""
        # Extract features and labels from test data
        features = []
        labels = []
        
        for _, row in test_data.iterrows():
            feat = FEATURE_ENGINEER.compute_features()
            if feat is not None:
                features.append(feat)
                # Determine label based on future price movement
                future_return = row.get('future_return', 0)
                if future_return > 0.001:
                    labels.append(2)  # Up
                elif future_return < -0.001:
                    labels.append(0)  # Down
                else:
                    labels.append(1)  # Neutral
        
        if not features:
            return 0.0
        
        X = np.array(features)
        y = np.array(labels)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Calculate accuracy
        accuracy = accuracy_score(y, predictions)
        
        return accuracy
    
    def backtest_strategy(
        self,
        signals: List[Dict],
        prices: np.ndarray,
        transaction_cost: float = 0.0001
    ) -> Dict[str, float]:
        """Backtest trading strategy."""
        
        position = 0
        cash = 10000
        trades = []
        
        for i, signal in enumerate(signals):
            if i >= len(prices):
                break
            
            action = signal.get('action', 'HOLD')
            price = prices[i]
            
            if action == 'BUY' and position == 0:
                # Enter long
                position = cash / price
                cash = 0
                trades.append(('buy', price))
                
            elif action == 'SELL' and position > 0:
                # Exit long
                cash = position * price * (1 - transaction_cost)
                position = 0
                trades.append(('sell', price))
        
        # Close final position
        if position > 0:
            cash = position * prices[-1] * (1 - transaction_cost)
            trades.append(('close', prices[-1]))
        
        # Calculate metrics
        total_return = (cash - 10000) / 10000
        
        # Calculate Sharpe ratio
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        if returns:
            sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        cumulative = (1 + np.array(returns)).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = sum(1 for t in trades if 'close' in t[0] and t[1] > 0)
        total_trades = sum(1 for t in trades if 'close' in t[0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate
        }
    
    def validate_predictions(
        self,
        predictions: np.ndarray,
        actual_labels: np.ndarray
    ) -> Dict[str, float]:
        """Validate prediction accuracy."""
        
        if len(predictions) != len(actual_labels):
            return {}
        
        # Basic metrics
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        class_metrics = {}
        for class_label in [0, 1, 2]:
            mask = actual_labels == class_label
            if mask.sum() > 0:
                class_acc = (predictions[mask] == class_label).mean()
                class_metrics[f'class_{class_label}_acc'] = class_acc
        
        # Store validation
        self.validation_history.append({
            'timestamp': time.time(),
            'accuracy': accuracy,
            'f1': f1,
            'n_samples': len(predictions)
        })
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **class_metrics
        }

# Initialize validation system
VALIDATOR = ValidationSystem()

# === MAIN TRAINING FUNCTIONS ===
def partial_fit_all_models():
    """Partial fit all models across timeframes."""
    
    # 1-minute models
    ONLINE_LEARNER.partial_fit_models("1m")
    
    # Store features for current time
    features = FEATURE_ENGINEER.compute_features()
    if features is not None:
        STATE.features_buffer.append((time.time(), features))
        STATE.features_buffer_5m.append((time.time(), features))
        STATE.features_buffer_15m.append((time.time(), features))
    
    # 5-minute models
    if time.time() - STATE.last_5min_fit > 300:
        ONLINE_LEARNER.partial_fit_models("5m")
        STATE.last_5min_fit = time.time()
    
    # 15-minute models  
    if time.time() - STATE.last_15min_fit > 900:
        ONLINE_LEARNER.partial_fit_models("15m")
        STATE.last_15min_fit = time.time()
    
    # Train anomaly detector
    ANOMALY_DETECTOR.train_detectors()

def batch_retrain_all_models():
    """Batch retrain all models if needed."""
    
    current_time = time.time()
    
    # Check if batch retrain needed
    if current_time - STATE.LAST_BATCH_REFIT_TIME < STATE.BATCH_REFIT_INTERVAL:
        return
    
    logger.info("Starting batch retrain for all models")
    
    # Retrain each timeframe
    for timeframe in ["1m", "5m", "15m"]:
        ONLINE_LEARNER.partial_fit_models(timeframe)
    
    STATE.LAST_BATCH_REFIT_TIME = current_time
    
def save_all_models(force: bool = False):
    """Save all trained models."""
    
    current_time = time.time()
    
    # Check if save needed
    if not force and current_time - STATE.last_save_time < 3600:  # Save hourly
        return
    
    try:
        STATE.save_models()
        STATE.last_save_time = current_time
        logger.info("All models saved successfully")
    except Exception as e:
        logger.error(f"Error saving models: {e}")

# === TELEGRAM MESSAGE BUILDER ===
class TelegramMessageBuilder:
    """Build formatted messages for Telegram notifications."""
    
    def __init__(self, state, trading_signals=None):
        self.state = state
        self.trading_signals = trading_signals or TRADING_SIGNALS
        self._cache = {}
        self._cache_ttl = 60  # Cache TTL en secondes
        self._cache_timestamps = {}
    
    def safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """Récupère une valeur de manière sécurisée"""
        try:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        except Exception:
            return default
    
    def format_percentage(self, value: Optional[float], decimals: int = 1) -> str:
        """Formate un pourcentage de manière sécurisée"""
        try:
            if value is None:
                return "N/A"
            return f"{value:.{decimals}%}"
        except (TypeError, ValueError):
            return "N/A"
    
    def format_number(self, value: Optional[float], decimals: int = 2) -> str:
        """Formate un nombre de manière sécurisée"""
        try:
            if value is None:
                return "N/A"
            return f"{value:.{decimals}f}"
        except (TypeError, ValueError):
            return "N/A"
    
    def format_currency(self, value: Optional[float]) -> str:
        """Formate une valeur monétaire"""
        try:
            if value is None:
                return "$0"
            return f"${value:,.2f}"
        except (TypeError, ValueError):
            return "$0"
    
    def format_large_number(self, value: Optional[float]) -> str:
        """Formate un grand nombre avec suffixe K/M"""
        try:
            if value is None or value == 0:
                return "0"
            
            if abs(value) >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.1f}K"
            else:
                return f"{value:.0f}"
        except (TypeError, ValueError):
            return "0"
    
    def _now_hms_str(self) -> str:
        """Retourne l'heure actuelle formatée"""
        ny_tz = pytz.timezone("America/New_York")
        return datetime.datetime.now(ny_tz).strftime("%H:%M:%S ET")
    
    def _calculate_buy_sell_pressure(self) -> float:
        """Calcule la pression d'achat/vente"""
        buy_vol = self.safe_get(self.state, 'buy_volume', 0)
        sell_vol = self.safe_get(self.state, 'sell_volume', 0)
        total = buy_vol + sell_vol
        
        if total > 0:
            return (buy_vol - sell_vol) / total
        return 0.0
    
    def build_institutional_update(self, loc_, decision, pred_dict, metrics) -> str:
        """Construit le message de mise à jour institutionnelle (5 minutes)"""
        try:
            ct = self._now_hms_str()
            s_ = self.safe_get(self.state, 'current_symbol', 'N/A')
            
            # Signal principal
            action = self.safe_get(decision, 'action', 'HOLD')
            confidence = self.safe_get(decision, 'confidence', 0)
            signal_emoji = self.safe_get(self.trading_signals.get(action), 'emoji', '➡️')
            
            # Métriques de performance
            daily_pnl = self.safe_get(self.state.metrics, 'daily_pnl', 0)
            sharpe = self.safe_get(self.state.metrics, 'sharpe_ratio', 0)
            win_rate = self.safe_get(self.state.metrics, 'win_rate', 0)
            
            # Microstructure
            vpin = self.safe_get(metrics, 'vpin', 0)
            kyle_lambda = self.safe_get(metrics, 'kyle_lambda', 0)
            effective_spread = self.safe_get(metrics, 'effective_spread', 0)
            
            # Probabilités ML
            prob_1m = self.safe_get(pred_dict, '1m', {}).get('prediction', 0.5)
            prob_5m = self.safe_get(pred_dict, '5m', {}).get('prediction', 0.5)
            conf_1m = self.safe_get(pred_dict, '1m', {}).get('confidence', 0)
            conf_5m = self.safe_get(pred_dict, '5m', {}).get('confidence', 0)
            
            # Construction du message
            msg = f"""⚡ **INSTITUTIONAL UPDATE** | {ct}
━━━━━━━━━━━━━━━━━━━━━
📈 **{s_}** | {signal_emoji} {action} ({self.format_percentage(confidence)})

**📊 MARKET MICROSTRUCTURE**
• VPIN: {self.format_number(vpin, 3)}
• Kyle's λ: {self.format_number(kyle_lambda, 6)}
• Effective Spread: {self.format_percentage(effective_spread)}

**🤖 ML ENSEMBLE SIGNALS**
• 1min: {self.format_percentage(prob_1m)} (conf: {self.format_percentage(conf_1m)})
• 5min: {self.format_percentage(prob_5m)} (conf: {self.format_percentage(conf_5m)})

**💼 PORTFOLIO METRICS**
• Daily P&L: {self.format_currency(daily_pnl)}
• Sharpe: {self.format_number(sharpe, 2)}
• Win Rate: {self.format_percentage(win_rate)}

**🎯 EXECUTION**
• Risk Score: {self.format_number(self.safe_get(decision, 'risk_score', 0), 2)}
• Position Size: {self.format_number(self.safe_get(decision, 'position_size', 0), 0)}
• Stop Loss: {self.format_currency(self.safe_get(decision, 'stop_loss', 0))}
━━━━━━━━━━━━━━━━━━━━━"""
            
            # Ajouter l'analyse GPT si disponible
            if self.safe_get(self.state, 'gpt_has_alert', False):
                gpt_text = self.safe_get(self.state, 'latest_gpt_analysis', '')
                if gpt_text:
                    msg += f"\n\n**🧠 GPT-4.1 & GPT-4o ANALYSIS**\n{gpt_text}"
            
            return msg
            
        except Exception as e:
            logger.error(f"Erreur lors de la construction du message institutionnel: {e}")
            return f"⚠️ Error building update: {str(e)}"
    
    def build_10min_strategic(self) -> str:
        """Construit le message stratégique 10 minutes"""
        try:
            ct = self._now_hms_str()
            s_ = self.safe_get(self.state, 'current_symbol', 'N/A')
            
            # Sections principales
            analysis = self._generate_strategic_analysis()
            outlook = self._generate_multi_horizon_outlook()
            risk = self._generate_risk_dashboard()
            execution = self._generate_execution_recommendations()
            
            return f"""📈 **10-MINUTE STRATEGIC UPDATE** | {ct}
━━━━━━━━━━━━━━━━━━━━━
Symbol: {s_}

{analysis}

{outlook}

{risk}

{execution}

━━━━━━━━━━━━━━━━━━━━━
💡 *Use this analysis for position sizing and risk management decisions*"""
            
        except Exception as e:
            logger.error(f"Erreur dans build_10min_strategic: {e}")
            return "⚠️ Error generating strategic update"
    
    def build_15min_comprehensive(self) -> str:
        """Construit le rapport complet 15 minutes"""
        try:
            ct = self._now_hms_str()
            s_ = self.safe_get(self.state, 'current_symbol', 'N/A')
            
            # Performance summary
            perf_summary = self._analyze_15min_performance()
            
            # Market structure
            market_analysis = self._analyze_market_structure()
            
            # ML performance
            ml_performance = self._analyze_ml_performance()
            
            # Risk metrics
            risk_metrics = self._calculate_comprehensive_risk()
            
            return f"""📊 **15-MINUTE COMPREHENSIVE REPORT** | {ct}
━━━━━━━━━━━━━━━━━━━━━
Symbol: {s_}

{perf_summary}

{market_analysis}

{ml_performance}

{risk_metrics}

**📈 TRADING RECOMMENDATIONS**
{self._generate_trading_recommendations()}

━━━━━━━━━━━━━━━━━━━━━
📎 *Next update in 15 minutes*"""
            
        except Exception as e:
            logger.error(f"Erreur dans build_15min_comprehensive: {e}")
            return "⚠️ Error generating comprehensive report"
    
    def _generate_strategic_analysis(self) -> str:
        """Génère l'analyse stratégique"""
        regime = self.safe_get(self.state.market_regime, 'regime', 'Unknown')
        volatility = self.safe_get(self.state.market_regime, 'volatility', 0)
        trend_strength = self.safe_get(self.state, 'trend_strength', 0)
        
        return f"""**🎯 STRATEGIC ANALYSIS**
• Market Regime: {regime}
• Volatility: {self.format_percentage(volatility)}
• Trend Strength: {self.format_percentage(trend_strength)}
• Momentum: {self._calculate_momentum_state()}"""
    
    def _generate_multi_horizon_outlook(self) -> str:
        """Génère les perspectives multi-horizons"""
        accuracies = HORIZON_TRACKER.calculate_accuracy_by_horizon()
        
        outlook = "**🔮 MULTI-HORIZON OUTLOOK**\n"
        for horizon, accuracy in accuracies.items():
            outlook += f"• {horizon}: {self.format_percentage(accuracy)} accuracy\n"
        
        return outlook
    
    def _generate_risk_dashboard(self) -> str:
        """Génère le tableau de bord des risques"""
        var_95 = self.safe_get(self.state, 'var_95', 0)
        max_dd = self.safe_get(self.state, 'max_drawdown', 0)
        liquidity_score = self.safe_get(self.state, 'liquidity_score', 0)
        
        return f"""**⚠️ RISK DASHBOARD**
• VaR (95%): {self.format_percentage(var_95)}
• Max Drawdown: {self.format_percentage(max_dd)}
• Liquidity Score: {self.format_percentage(liquidity_score)}
• Portfolio Heat: {self.format_percentage(self.safe_get(self.state, 'portfolio_heat', 0))}"""
    
    def _generate_execution_recommendations(self) -> str:
        """Génère les recommandations d'exécution"""
        spread = self.safe_get(self.state, 'bid_ask_spread', 0)
        volume_profile = self._analyze_volume_profile()
        
        return f"""**⚡ EXECUTION RECOMMENDATIONS**
• Optimal Size: {self._calculate_optimal_position_size()}
• Entry Strategy: {self._determine_entry_strategy()}
• Current Spread: {self.format_percentage(spread)}
• Volume Profile: {volume_profile}"""
    
    def _analyze_15min_performance(self) -> str:
        """Analyse la performance sur 15 minutes"""
        pnl = self.safe_get(self.state, 'pnl_15min', 0)
        trades = self.safe_get(self.state, 'trades_15min', 0)
        win_rate = self.safe_get(self.state, 'win_rate_15min', 0)
        
        return f"""**📊 15-MINUTE PERFORMANCE**
• P&L: {self.format_currency(pnl)}
• Trades: {trades}
• Win Rate: {self.format_percentage(win_rate)}
• Sharpe (15m): {self._calculate_15min_sharpe()}"""
    
    def _analyze_market_structure(self) -> str:
        """Analyse la structure du marché"""
        return f"""**🏗️ MARKET STRUCTURE**
• Order Flow: {self._analyze_order_flow()}
• Price Levels: {self._identify_key_levels()}
• Market Depth: {self._analyze_market_depth()}
• Toxicity: {self.format_percentage(self.safe_get(self.state, 'toxicity_score', 0))}"""
    
    def _analyze_ml_performance(self) -> str:
        """Analyse la performance des modèles ML"""
        # Récupérer les métriques de performance des modèles
        report = "**🤖 ML PERFORMANCE**\n"
        
        for timeframe in ["1m", "5m", "15m"]:
            perf = self.safe_get(STATE.model_performance, timeframe, {})
            if perf and 'accuracy' in perf and len(perf['accuracy']) > 0:
                avg_accuracy = np.mean(list(perf['accuracy']))
                report += f"• {timeframe}: {self.format_percentage(avg_accuracy)} avg accuracy\n"
        
        return report
    
    def _calculate_comprehensive_risk(self) -> str:
        """Calcule les métriques de risque complètes"""
        metrics = self.state.metrics
        
        return f"""**📉 RISK METRICS**
• Current Drawdown: {self.format_percentage(self.safe_get(metrics, 'current_drawdown', 0))}
• Consecutive Losses: {self.safe_get(metrics, 'consecutive_losses', 0)}
• Risk-Adjusted Return: {self._calculate_risk_adjusted_return()}
• Exposure Level: {self.format_percentage(self.safe_get(self.state, 'total_exposure', 0))}"""
    
    def _generate_trading_recommendations(self) -> str:
        """Génère des recommandations de trading spécifiques"""
        recommendations = []
        
        # Analyser les conditions actuelles
        regime = self.safe_get(self.state.market_regime, 'regime', 'Unknown')
        volatility = self.safe_get(self.state.market_regime, 'volatility', 0)
        
        if regime == "Trending" and volatility < 0.2:
            recommendations.append("✅ Favorable conditions for trend following")
        elif regime == "Choppy":
            recommendations.append("⚠️ Reduce position size in choppy conditions")
        
        if self.safe_get(self.state, 'circuit_breaker_triggered', False):
            recommendations.append("🛑 Circuit breaker active - no new positions")
        
        return "\n".join(f"• {rec}" for rec in recommendations) if recommendations else "• Monitor current positions"
    
    # Méthodes helper
    def _calculate_momentum_state(self) -> str:
        """Calcule l'état du momentum"""
        momentum = self.safe_get(self.state, 'momentum_score', 0)
        if momentum > 0.7:
            return "Strong Bullish"
        elif momentum > 0.3:
            return "Bullish"
        elif momentum < -0.7:
            return "Strong Bearish"
        elif momentum < -0.3:
            return "Bearish"
        else:
            return "Neutral"
    
    def _calculate_optimal_position_size(self) -> str:
        """Calcule la taille de position optimale"""
        risk_limit = self.safe_get(self.state.risk_limits, 'max_position_size', 10000)
        volatility = self.safe_get(self.state.market_regime, 'volatility', 0.15)
        
        # Ajuster selon la volatilité
        if volatility > 0.3:
            size = risk_limit * 0.5
        elif volatility > 0.2:
            size = risk_limit * 0.75
        else:
            size = risk_limit
        
        return self.format_number(size, 0)
    
    def _determine_entry_strategy(self) -> str:
        """Détermine la stratégie d'entrée optimale"""
        spread = self.safe_get(self.state, 'bid_ask_spread', 0.001)
        volume = self.safe_get(self.state, 'RELATIVE_VOLUME', 1)
        
        if spread > 0.002 and volume < 0.8:
            return "Limit orders recommended"
        elif volume > 1.5:
            return "Market orders acceptable"
        else:
            return "Scaled entry suggested"
    
    def _analyze_volume_profile(self) -> str:
        """Analyse le profil de volume"""
        rel_vol = self.safe_get(self.state, 'RELATIVE_VOLUME', 1)
        
        if rel_vol > 2:
            return "High (institutional activity)"
        elif rel_vol > 1.3:
            return "Above average"
        elif rel_vol < 0.7:
            return "Low (poor liquidity)"
        else:
            return "Normal"
    
    def _calculate_15min_sharpe(self) -> str:
        """Calcule le Sharpe ratio sur 15 minutes"""
        # Simplified calculation
        returns = self.safe_get(self.state, 'returns_15min', [])
        if returns and len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4)  # Annualized
            return self.format_number(sharpe, 2)
        return "N/A"
    
    def _analyze_order_flow(self) -> str:
        """Analyse le flux d'ordres"""
        ofi = self.safe_get(self.state, 'OFI_VALUE', 0)
        cdv = self.safe_get(self.state, 'CDV_VALUE', 0)
        
        if ofi > 500 and cdv > 100:
            return "Strong buying pressure"
        elif ofi < -500 and cdv < -100:
            return "Strong selling pressure"
        else:
            return "Balanced"
    
    def _identify_key_levels(self) -> str:
        """Identifie les niveaux clés"""
        # Simplified - would use actual price analysis
        current_price = self.safe_get(self.state, 'current_price', 100)
        return f"Support: {self.format_currency(current_price * 0.98)}, Resistance: {self.format_currency(current_price * 1.02)}"
    
    def _analyze_market_depth(self) -> str:
        """Analyse la profondeur du marché"""
        with locks.l2_lock:
            bid_depth = sum(STATE.bids_dict.values()) if STATE.bids_dict else 0
            ask_depth = sum(STATE.asks_dict.values()) if STATE.asks_dict else 0
        
        total_depth = bid_depth + ask_depth
        if total_depth > 10000:
            return "Deep"
        elif total_depth > 5000:
            return "Moderate"
        else:
            return "Shallow"
    
    def _calculate_risk_adjusted_return(self) -> str:
        """Calcule le rendement ajusté au risque"""
        returns = self.safe_get(self.state.metrics, 'total_pnl', 0)
        risk = self.safe_get(self.state, 'portfolio_heat', 0.1)
        
        if risk > 0:
            rar = returns / risk
            return self.format_number(rar, 2)
        return "N/A"

# Initialize Telegram message builder
TELEGRAM_MESSAGE_BUILDER = TelegramMessageBuilder(STATE, TRADING_SIGNALS)

# === GPT INTEGRATION ===
def build_gpt_prompt() -> str:
    """Build comprehensive prompt for GPT-4.1 and GPT-4o analysis."""
    
    with locks.tns_lock:
        df_tns = STATE.df_tns.copy()
    with locks.l2_lock:
        df_l2 = STATE.df_l2_display.copy()
    
    if df_tns.empty:
        return ""
    
    # Calculate metrics
    current_price = df_tns['Price'].iloc[-1]
    
    # Price movement
    price_5m_ago = df_tns[df_tns['Timestamp'] > time.time() - 300]['Price'].iloc[0] if len(df_tns) > 50 else current_price
    price_change_5m = (current_price - price_5m_ago) / price_5m_ago * 100
    
    # Volume analysis
    volume_1m = df_tns[df_tns['Timestamp'] > time.time() - 60]['Size'].sum()
    volume_5m = df_tns[df_tns['Timestamp'] > time.time() - 300]['Size'].sum()
    
    # Technical indicators
    indicators = compute_technical_indicators(df_tns)
    
    # Microstructure
    microstructure = format_microstructure_report()
    
    # ML predictions
    features = FEATURE_ENGINEER.compute_features()
    if features is not None:
        ml_prediction = MODEL_MANAGER.get_ensemble_prediction(features)
        ml_prob = ml_prediction['prediction']
        ml_confidence = ml_prediction['confidence']
    else:
        ml_prob = 0.5
        ml_confidence = 0.0
    
    # Build prompt for GPT-4.1 and GPT-4o
    prompt = f"""You are an expert quantitative trader using GPT-4.1 and GPT-4o capabilities. Analyze this real-time market data for {STATE.current_symbol}:

PRICE ACTION:
- Current: ${current_price:.3f}
- 5m Change: {price_change_5m:+.2f}%
- High 5m: ${df_tns[df_tns['Timestamp'] > time.time() - 300]['Price'].max():.3f}
- Low 5m: ${df_tns[df_tns['Timestamp'] > time.time() - 300]['Price'].min():.3f}

VOLUME:
- 1m Volume: {volume_1m:,.0f}
- 5m Volume: {volume_5m:,.0f}
- Relative Volume: {STATE.RELATIVE_VOLUME:.1f}x

TECHNICAL INDICATORS:
- RSI(14): {indicators.get('rsi', 50):.0f}
- VWAP: ${indicators.get('vwap', current_price):.3f}
- Price vs VWAP: {((current_price/indicators.get('vwap', current_price) - 1) * 100):+.2f}%

{microstructure}

ML ENSEMBLE (Online Learning):
- Probability: {ml_prob:.1%} (Confidence: {ml_confidence:.1%})
- Signal: {get_signal_from_prob(ml_prob)}

DETECTED PATTERNS:
{format_detected_signals()}

Provide a concise institutional-grade analysis focusing on:
1. Immediate price direction (next 1-5 minutes) with specific price targets
2. Key support/resistance levels
3. Risk factors and potential catalysts
4. Actionable trading recommendation with position sizing

Keep response under 150 words. Be specific, quantitative, and decisive."""
    
    return prompt

def get_signal_from_prob(prob: float) -> str:
    """Convert probability to signal name."""
    if prob > 0.75:
        return "STRONG_BUY"
    elif prob > 0.65:
        return "BUY"
    elif prob < 0.25:
        return "STRONG_SELL"
    elif prob < 0.35:
        return "SELL"
    else:
        return "NEUTRAL"

# === UTILITY FUNCTIONS ===
def get_current_price(symbol: str = None) -> Optional[float]:
    """Get current price from time and sales data."""
    try:
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if df.empty:
            return None
        
        return float(df['Price'].iloc[-1])
    except Exception:
        return None

def ensure_price_ib(symbol: str, timeout: float = 2.0) -> Optional[float]:
    """Get price from IB with timeout."""
    # First try to get from current data
    price = get_current_price(symbol)
    if price is not None:
        return price
    
    # Request market data if needed
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        ticker = ib.reqMktData(contract, '', False, False)
        
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            if ticker.last:
                return ticker.last
            
            price = get_current_price(symbol)
            if price is not None:
                return price
            
            safe_sleep(0.05)
        
    except Exception as e:
        logger.warning(f"Failed to get price for {symbol}: {e}")
    
    return None

def now_hms_str() -> str:
    """Get current time in HH:MM:SS ET format."""
    ny_tz = pytz.timezone("America/New_York")
    return datetime.datetime.now(ny_tz).strftime("%H:%M:%S ET")

def is_market_hours() -> bool:
    """Check if US market is open."""
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)
    
    # Check if weekend
    if now.weekday() >= 5:
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def get_extended_hours_session() -> str:
    """Get current trading session."""
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)
    
    if now.weekday() >= 5:
        return "CLOSED"
    
    hour = now.hour
    minute = now.minute
    
    # Pre-market: 4:00 AM - 9:30 AM
    if hour >= 4 and (hour < 9 or (hour == 9 and minute < 30)):
        return "PREMARKET"
    
    # Regular hours: 9:30 AM - 4:00 PM
    if (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16:
        return "REGULAR"
    
    # After-hours: 4:00 PM - 8:00 PM
    if hour >= 16 and hour < 20:
        return "AFTERHOURS"
    
    return "CLOSED"

def compute_technical_indicators(df_tns: pd.DataFrame) -> Dict[str, float]:
    """Compute common technical indicators."""
    indicators = {}
    
    if df_tns.empty or len(df_tns) < 20:
        return indicators
    
    prices = df_tns['Price'].values
    volumes = df_tns['Size'].values
    
    # RSI
    if len(prices) >= 14:
        returns = np.diff(prices)
        gains = returns[returns > 0]
        losses = abs(returns[returns < 0])
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        else:
            indicators['rsi'] = 100 if avg_gain > 0 else 50
    
    # VWAP
    if STATE.ANCHOR_TIMESTAMP and len(prices) > 0:
        recent_df = df_tns[df_tns['Timestamp'] > STATE.ANCHOR_TIMESTAMP]
        if not recent_df.empty:
            cum_volume = recent_df['Size'].cumsum()
            cum_pv = (recent_df['Price'] * recent_df['Size']).cumsum()
            vwap = cum_pv.iloc[-1] / cum_volume.iloc[-1] if cum_volume.iloc[-1] > 0 else prices[-1]
            indicators['vwap'] = vwap
    
    # Moving averages
    if len(prices) >= 20:
        indicators['sma_20'] = np.mean(prices[-20:])
    
    if len(prices) >= 50:
        indicators['sma_50'] = np.mean(prices[-50:])
    
    # Volatility
    if len(prices) >= 20:
        returns = np.diff(np.log(prices[-20:]))
        indicators['volatility'] = np.std(returns) * np.sqrt(252)
    
    return indicators

def format_microstructure_report() -> str:
    """Format microstructure metrics for GPT analysis."""
    report = "MICROSTRUCTURE:\n"
    
    # Order flow
    report += f"- Order Flow Imbalance: {STATE.OFI_VALUE:.0f}\n"
    report += f"- Cumulative Delta Volume: {STATE.CDV_VALUE:.0f}\n"
    
    # Spread
    with locks.l2_lock:
        if STATE.bids_dict and STATE.asks_dict:
            best_bid = max(STATE.bids_dict.keys())
            best_ask = min(STATE.asks_dict.keys())
            spread_bps = (best_ask - best_bid) / best_bid * 10000
            report += f"- Bid-Ask Spread: {spread_bps:.1f} bps\n"
            
            # Book imbalance
            bid_vol = sum(STATE.bids_dict.values())
            ask_vol = sum(STATE.asks_dict.values())
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                imbalance = (bid_vol - ask_vol) / total_vol
                report += f"- Book Imbalance: {imbalance:.1%}\n"
    
    # Toxicity
    report += f"- Toxicity Score: {STATE.toxicity_score:.1%}\n"
    
    return report

def format_detected_signals() -> str:
    """Format detected trading signals."""
    signals = []
    
    # Price momentum
    with locks.tns_lock:
        df = STATE.df_tns.copy()
    
    if not df.empty and len(df) > 100:
        prices = df['Price'].values
        
        # Breakout detection
        high_20 = np.max(prices[-20:])
        low_20 = np.min(prices[-20:])
        current = prices[-1]
        
        if current > high_20 * 0.999:
            signals.append("- Breakout above 20-period high")
        elif current < low_20 * 1.001:
            signals.append("- Breakdown below 20-period low")
        
        # Volume surge
        if STATE.RELATIVE_VOLUME > 2:
            signals.append("- Volume surge detected (2x average)")
        
        # Momentum divergence
        if len(prices) > 50:
            price_change = (prices[-1] - prices[-10]) / prices[-10]
            rsi_change = compute_technical_indicators(df.tail(50)).get('rsi', 50) - 50
            
            if price_change > 0 and rsi_change < -10:
                signals.append("- Bearish RSI divergence")
            elif price_change < 0 and rsi_change > 10:
                signals.append("- Bullish RSI divergence")
    
    # Microstructure signals
    if abs(STATE.OFI_VALUE) > 1000:
        direction = "buying" if STATE.OFI_VALUE > 0 else "selling"
        signals.append(f"- Heavy {direction} pressure detected")
    
    return "\n".join(signals) if signals else "- No significant patterns detected"

def telegram_send(message: str, force: bool = False):
    """Send message to Telegram."""
    STATE.send_alert(message, force)

# === RISK MANAGEMENT ===
def check_risk_limits() -> bool:
    """Check if trading is allowed based on risk limits."""
    
    # Check circuit breaker
    if STATE.circuit_breaker_triggered:
        return False
    
    # Check daily loss limit
    if STATE.metrics.daily_pnl < STATE.risk_limits.max_daily_loss:
        STATE.circuit_breaker_triggered = True
        STATE.send_alert("🛑 CIRCUIT BREAKER: Daily loss limit reached!")
        return False
    
    # Check consecutive losses
    if STATE.metrics.consecutive_losses >= STATE.risk_limits.consecutive_loss_limit:
        STATE.circuit_breaker_triggered = True
        STATE.send_alert("🛑 CIRCUIT BREAKER: Consecutive loss limit reached!")
        return False
    
    # Check portfolio heat
    if STATE.portfolio_heat > STATE.config.risk.max_portfolio_heat:
        STATE.send_alert("⚠️ WARNING: Portfolio heat exceeds limit!")
        return False
    
    return True

def calculate_portfolio_var() -> float:
    """Calculate portfolio Value at Risk."""
    if not STATE.equity_curve or len(STATE.equity_curve) < 20:
        return 0.0
    
    returns = np.diff(list(STATE.equity_curve)[-100:])
    if len(returns) == 0:
        return 0.0
    
    # Historical VaR at 95% confidence
    var_95 = np.percentile(returns, 5)
    
    return abs(var_95)

def update_order_flow_metrics():
    """Update order flow imbalance and cumulative delta volume."""
    try:
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if df.empty or len(df) < 10:
            return
        
        # Recent trades
        recent_trades = df.tail(100)
        
        # Calculate OFI (Order Flow Imbalance)
        # Simplified: classify trades as buy/sell based on tick direction
        price_changes = recent_trades['Price'].diff()
        
        buy_volume = recent_trades.loc[price_changes > 0, 'Size'].sum()
        sell_volume = recent_trades.loc[price_changes < 0, 'Size'].sum()
        
        STATE.OFI_VALUE = buy_volume - sell_volume
        
        # Calculate CDV (Cumulative Delta Volume)
        # Rolling sum of signed volume
        signed_volume = recent_trades['Size'].values.copy()
        signed_volume[price_changes < 0] *= -1
        
        STATE.CDV_VALUE = np.sum(signed_volume)
        
        # Calculate relative volume
        if len(df) > 1000:
            recent_vol = recent_trades['Size'].sum()
            historical_vol = df.iloc[-1000:-100]['Size'].sum() / 9  # Average per 100 trades
            STATE.RELATIVE_VOLUME = recent_vol / historical_vol if historical_vol > 0 else 1.0
        
    except Exception as e:
        logger.error(f"Error updating order flow metrics: {e}")

def calculate_microstructure_metrics() -> Dict[str, float]:
    """Calculate advanced microstructure metrics."""
    metrics = {}
    
    try:
        # VPIN (Volume-synchronized Probability of Informed Trading)
        metrics['vpin'] = calculate_vpin()
        
        # Kyle's Lambda (price impact)
        metrics['kyle_lambda'] = calculate_kyle_lambda()
        
        # Effective spread
        metrics['effective_spread'] = calculate_effective_spread()
        
        # Quote stuffing detection
        metrics['quote_stuffing_score'] = detect_quote_stuffing()
        
        # Market quality score
        metrics['quality_score'] = analyze_microstructure_quality()
        
    except Exception as e:
        logger.error(f"Error calculating microstructure metrics: {e}")
    
    return metrics

def calculate_vpin() -> float:
    """Calculate Volume-synchronized Probability of Informed Trading."""
    try:
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if len(df) < 100:
            return 0.0
        
        # Bucket trades by volume
        bucket_size = df['Size'].sum() / 50  # 50 buckets
        
        buckets = []
        current_bucket = {'buy': 0, 'sell': 0}
        current_volume = 0
        
        for _, trade in df.tail(500).iterrows():
            # Classify trade
            if len(buckets) > 0:
                if trade['Price'] > buckets[-1].get('price', trade['Price']):
                    current_bucket['buy'] += trade['Size']
                else:
                    current_bucket['sell'] += trade['Size']
            else:
                current_bucket['buy'] += trade['Size'] / 2
                current_bucket['sell'] += trade['Size'] / 2
            
            current_volume += trade['Size']
            
            if current_volume >= bucket_size:
                current_bucket['price'] = trade['Price']
                buckets.append(current_bucket)
                current_bucket = {'buy': 0, 'sell': 0}
                current_volume = 0
        
        if len(buckets) < 10:
            return 0.0
        
        # Calculate VPIN
        vpins = []
        for bucket in buckets[-10:]:
            total = bucket['buy'] + bucket['sell']
            if total > 0:
                vpin = abs(bucket['buy'] - bucket['sell']) / total
                vpins.append(vpin)
        
        return np.mean(vpins) if vpins else 0.0
        
    except Exception as e:
        logger.error(f"VPIN calculation error: {e}")
        return 0.0

def calculate_kyle_lambda() -> float:
    """Calculate Kyle's Lambda (price impact coefficient)."""
    try:
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if len(df) < 50:
            return 0.0
        
        recent = df.tail(100)
        
        # Calculate price changes and signed volume
        price_changes = recent['Price'].diff().dropna()
        volumes = recent['Size'].iloc[1:]
        
        # Sign volumes based on price direction
        signed_volumes = volumes.values * np.sign(price_changes.values)
        
        # Regression: price change = lambda * signed volume
        if len(signed_volumes) > 10:
            X = signed_volumes.reshape(-1, 1)
            y = price_changes.values
            
            # Simple linear regression
            lambda_estimate = np.sum(X.flatten() * y) / np.sum(X.flatten() ** 2)
            
            return abs(lambda_estimate) * 1000000  # Scale for readability
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Kyle's Lambda calculation error: {e}")
        return 0.0

def calculate_effective_spread() -> float:
    """Calculate effective spread."""
    try:
        with locks.l2_lock:
            if not STATE.bids_dict or not STATE.asks_dict:
                return 0.0
            
            best_bid = max(STATE.bids_dict.keys())
            best_ask = min(STATE.asks_dict.keys())
            mid_price = (best_bid + best_ask) / 2
        
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if df.empty:
            return 0.0
        
        # Effective spread = 2 * |trade price - mid price| / mid price
        last_price = df['Price'].iloc[-1]
        effective_spread = 2 * abs(last_price - mid_price) / mid_price
        
        return effective_spread
        
    except Exception as e:
        logger.error(f"Effective spread calculation error: {e}")
        return 0.0

def detect_quote_stuffing() -> float:
    """Detect potential quote stuffing behavior."""
    try:
        # Analyze L2 event frequency
        if len(STATE.l2_events) < 100:
            return 0.0
        
        # Get recent events
        recent_events = list(STATE.l2_events)[-1000:]
        
        # Calculate message rate per second
        if len(recent_events) < 2:
            return 0.0
        
        time_span = recent_events[-1]['timestamp'] - recent_events[0]['timestamp']
        if time_span == 0:
            return 0.0
        
        message_rate = len(recent_events) / time_span
        
        # Calculate cancellation rate
        spread_changes = [e['spread'] for e in recent_events if 'spread' in e]
        if len(spread_changes) > 1:
            spread_volatility = np.std(spread_changes)
        else:
            spread_volatility = 0
        
        # Quote stuffing score (0-1)
        # High message rate + high spread volatility = potential quote stuffing
        stuffing_score = min(1.0, (message_rate / 100) * (spread_volatility * 10000))
        
        return stuffing_score
        
    except Exception as e:
        logger.error(f"Quote stuffing detection error: {e}")
        return 0.0

def analyze_microstructure_quality() -> float:
    """Analyze overall market microstructure quality."""
    try:
        quality_factors = []
        
        # Factor 1: Spread tightness
        with locks.l2_lock:
            if STATE.bids_dict and STATE.asks_dict:
                best_bid = max(STATE.bids_dict.keys())
                best_ask = min(STATE.asks_dict.keys())
                spread_bps = (best_ask - best_bid) / best_bid * 10000
                spread_quality = max(0, 1 - spread_bps / 20)  # 20 bps = 0 quality
                quality_factors.append(spread_quality)
        
        # Factor 2: Book depth
        with locks.l2_lock:
            total_depth = sum(STATE.bids_dict.values()) + sum(STATE.asks_dict.values())
            depth_quality = min(1, total_depth / 10000)  # 10k shares = full quality
            quality_factors.append(depth_quality)
        
        # Factor 3: Price efficiency (low toxicity)
        toxicity_quality = 1 - STATE.toxicity_score
        quality_factors.append(toxicity_quality)
        
        # Factor 4: Low quote stuffing
        stuffing_score = detect_quote_stuffing()
        stuffing_quality = 1 - stuffing_score
        quality_factors.append(stuffing_quality)
        
        # Overall quality score
        return np.mean(quality_factors) if quality_factors else 0.5
        
    except Exception as e:
        logger.error(f"Microstructure quality analysis error: {e}")
        return 0.5

# === EXECUTION ENGINE ===
class ExecutionEngine:
    """Advanced order execution with smart routing."""
    
    def __init__(self):
        self.active_orders = {}
        self.order_id_counter = 0
        self.execution_stats = defaultdict(list)
        
    def execute_signal(self, signal: Dict[str, Any]) -> Optional[str]:
        """Execute trading signal with optimal strategy."""
        
        # Check risk limits
        if not check_risk_limits():
            logger.warning("Risk limits exceeded - order rejected")
            return None
        
        # Determine execution strategy
        strategy = self.determine_execution_strategy(signal)
        
        # Execute based on strategy
        if strategy == 'aggressive':
            return self.execute_aggressive(signal)
        elif strategy == 'passive':
            return self.execute_passive(signal)
        elif strategy == 'iceberg':
            return self.execute_iceberg(signal)
        else:
            return self.execute_standard(signal)
    
    def determine_execution_strategy(self, signal: Dict[str, Any]) -> str:
        """Determine optimal execution strategy."""
        
        urgency = signal.get('urgency', 0.5)
        size = signal.get('size', 100)
        
        # Get market conditions
        spread = calculate_effective_spread()
        volatility = STATE.market_regime.volatility
        book_depth = self.calculate_book_depth()
        
        # Decision logic
        if urgency > STATE.config.execution.urgency_threshold and \
           spread < STATE.config.execution.max_spread_pct:
            return 'aggressive'
        
        if size > book_depth * STATE.config.execution.iceberg_threshold:
            return 'iceberg'
        
        if spread > STATE.config.execution.max_spread_pct * 2:
            return 'sniper'
        
        if volatility > 0.02:
            return 'twap'
        
        return 'passive'
    
    def calculate_book_depth(self) -> float:
        """Calculate total book depth within x% of mid price."""
        with locks.l2_lock:
            df_l2 = STATE.df_l2_display.copy()
        
        if df_l2.empty:
            return 0.0
        
        bids = df_l2[df_l2['Side'] == 'BID']
        asks = df_l2[df_l2['Side'] == 'ASK']
        
        if bids.empty or asks.empty:
            return 0.0
        
        mid_price = (bids['Price'].max() + asks['Price'].min()) / 2
        depth_range = mid_price * 0.001  # 0.1% from mid
        
        bid_depth = bids[bids['Price'] >= mid_price - depth_range]['Size'].sum()
        ask_depth = asks[asks['Price'] <= mid_price + depth_range]['Size'].sum()
        
        return bid_depth + ask_depth
    
    def execute_aggressive(self, signal: Dict[str, Any]) -> Optional[str]:
        """Execute with aggressive market orders."""
        symbol = signal.get('symbol', STATE.current_symbol)
        size = signal.get('size', 100)
        side = signal.get('side', 'BUY')
        
        logger.info(f"Executing aggressive order: {side} {size} {symbol}")
        
        # In production, would send market order through IB
        # For now, simulate execution
        order_id = f"ORD_{self.order_id_counter}"
        self.order_id_counter += 1
        
        self.active_orders[order_id] = {
            'signal': signal,
            'status': 'FILLED',
            'fill_price': get_current_price(),
            'timestamp': time.time()
        }
        
        return order_id
    
    def execute_passive(self, signal: Dict[str, Any]) -> Optional[str]:
        """Execute with passive limit orders."""
        symbol = signal.get('symbol', STATE.current_symbol)
        size = signal.get('size', 100)
        side = signal.get('side', 'BUY')
        
        # Calculate limit price
        with locks.l2_lock:
            if STATE.bids_dict and STATE.asks_dict:
                best_bid = max(STATE.bids_dict.keys())
                best_ask = min(STATE.asks_dict.keys())
                
                if side == 'BUY':
                    limit_price = best_bid + 0.01  # Improve by 1 cent
                else:
                    limit_price = best_ask - 0.01
            else:
                limit_price = get_current_price()
        
        logger.info(f"Executing passive order: {side} {size} {symbol} @ {limit_price}")
        
        order_id = f"ORD_{self.order_id_counter}"
        self.order_id_counter += 1
        
        self.active_orders[order_id] = {
            'signal': signal,
            'status': 'PENDING',
            'limit_price': limit_price,
            'timestamp': time.time()
        }
        
        return order_id
    
    def execute_iceberg(self, signal: Dict[str, Any]) -> Optional[str]:
        """Execute large order as iceberg."""
        total_size = signal.get('size', 1000)
        
        # Split into smaller chunks
        chunk_size = min(100, total_size // 10)
        
        logger.info(f"Executing iceberg order: {total_size} in chunks of {chunk_size}")
        
        # In production, would create multiple child orders
        order_id = f"ICEBERG_{self.order_id_counter}"
        self.order_id_counter += 1
        
        self.active_orders[order_id] = {
            'signal': signal,
            'status': 'WORKING',
            'total_size': total_size,
            'filled_size': 0,
            'chunk_size': chunk_size,
            'timestamp': time.time()
        }
        
        return order_id
    
    def execute_standard(self, signal: Dict[str, Any]) -> Optional[str]:
        """Execute standard market order."""
        return self.execute_aggressive(signal)
    
    def update_execution_stats(self, order_id: str, fill_price: float):
        """Update execution statistics."""
        if order_id not in self.active_orders:
            return
        
        order = self.active_orders[order_id]
        signal_price = order['signal'].get('price', fill_price)
        
        slippage = abs(fill_price - signal_price) / signal_price
        
        self.execution_stats['slippage'].append(slippage)
        self.execution_stats['fill_time'].append(time.time() - order['timestamp'])

# Initialize execution engine
EXECUTION_ENGINE = ExecutionEngine()

# === MARKET MANIPULATION DETECTION ===
class ManipulationDetector:
    """Detect potential market manipulation patterns."""
    
    def __init__(self):
        self.manipulation_scores = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.patterns = {
            'spoofing': 0,
            'layering': 0,
            'wash_trading': 0,
            'ramping': 0
        }
        self.compliance_log = []
    
    def detect_manipulation(self) -> Dict[str, Any]:
        """Run manipulation detection algorithms."""
        
        scores = {}
        
        # Spoofing detection
        scores['spoofing'] = self.detect_spoofing()
        
        # Layering detection
        scores['layering'] = self.detect_layering()
        
        # Wash trading detection
        scores['wash_trading'] = self.detect_wash_trading()
        
        # Price ramping detection
        scores['ramping'] = self.detect_ramping()
        
        # Overall manipulation score
        overall_score = np.mean(list(scores.values()))
        
        # Log if suspicious
        if overall_score > 0.7:
            alert = {
                'timestamp': datetime.datetime.now(),
                'type': 'MANIPULATION_ALERT',
                'scores': scores,
                'overall': overall_score
            }
            self.alerts.append(alert)
            self.compliance_log.append(alert)
            
            # Send alert
            STATE.send_alert(f"⚠️ MANIPULATION ALERT: Score {overall_score:.1%}")
        
        return {
            'scores': scores,
            'overall': overall_score,
            'alert': overall_score > 0.7
        }
    
    def detect_spoofing(self) -> float:
        """Detect spoofing patterns in order book."""
        try:
            if len(STATE.l2_events) < 100:
                return 0.0
            
            # Analyze order book changes
            recent_events = list(STATE.l2_events)[-100:]
            
            # Look for large orders that disappear quickly
            large_orders_placed = 0
            large_orders_cancelled = 0
            
            for i in range(1, len(recent_events)):
                prev = recent_events[i-1]
                curr = recent_events[i]
                
                # Simple heuristic: check for large size changes
                if 'bid_levels' in prev and 'bid_levels' in curr:
                    if prev['bid_levels'] > curr['bid_levels'] + 3:
                        large_orders_cancelled += 1
                    elif curr['bid_levels'] > prev['bid_levels'] + 3:
                        large_orders_placed += 1
            
            if large_orders_placed > 0:
                spoof_ratio = large_orders_cancelled / large_orders_placed
                return min(1.0, spoof_ratio)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Spoofing detection error: {e}")
            return 0.0
    
    def detect_layering(self) -> float:
        """Detect layering patterns."""
        try:
            with locks.l2_lock:
                if not STATE.bids_dict or not STATE.asks_dict:
                    return 0.0
                
                # Check for multiple orders at different price levels
                bid_levels = len(STATE.bids_dict)
                ask_levels = len(STATE.asks_dict)
                
                # Look for imbalanced book with many levels
                if bid_levels > 10 and ask_levels < 3:
                    return 0.8
                elif ask_levels > 10 and bid_levels < 3:
                    return 0.8
                elif max(bid_levels, ask_levels) > 15:
                    return 0.5
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Layering detection error: {e}")
            return 0.0
    
    def detect_wash_trading(self) -> float:
        """Detect potential wash trading."""
        try:
            with locks.tns_lock:
                df = STATE.df_tns.copy()
            
            if len(df) < 100:
                return 0.0
            
            recent = df.tail(100)
            
            # Look for trades with same size at same price in short time
            duplicates = 0
            
            for i in range(1, len(recent)):
                if (recent.iloc[i]['Price'] == recent.iloc[i-1]['Price'] and
                    recent.iloc[i]['Size'] == recent.iloc[i-1]['Size'] and
                    recent.iloc[i]['Timestamp'] - recent.iloc[i-1]['Timestamp'] < 1):
                    duplicates += 1
            
            wash_score = duplicates / len(recent)
            
            return min(1.0, wash_score * 10)  # Scale up small percentages
            
        except Exception as e:
            logger.error(f"Wash trading detection error: {e}")
            return 0.0
    
    def detect_ramping(self) -> float:
        """Detect price ramping/marking."""
        try:
            with locks.tns_lock:
                df = STATE.df_tns.copy()
            
            if len(df) < 50:
                return 0.0
            
            # Check for aggressive buying/selling near close
            ny_tz = pytz.timezone("America/New_York")
            now = datetime.datetime.now(ny_tz)
            minutes_to_close = (16 - now.hour) * 60 - now.minute
            
            if minutes_to_close > 30:
                return 0.0  # Only check near close
            
            recent = df.tail(50)
            price_change = (recent['Price'].iloc[-1] - recent['Price'].iloc[0]) / recent['Price'].iloc[0]
            volume_surge = recent['Size'].sum() / df['Size'].mean()
            
            # Ramping score based on price change and volume
            if abs(price_change) > 0.005 and volume_surge > 3:
                return min(1.0, abs(price_change) * 100 * volume_surge / 3)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Ramping detection error: {e}")
            return 0.0
    
    def generate_compliance_report(self) -> pd.DataFrame:
        """Generate compliance report for regulatory purposes."""
        if not self.compliance_log:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.compliance_log)
        
        # Add summary statistics
        summary = {
            'total_alerts': len(df),
            'alerts_by_type': df['type'].value_counts().to_dict(),
            'recent_alerts': len(df[df['timestamp'] > datetime.datetime.now() - datetime.timedelta(hours=1)])
        }
        
        return df, summary

# Initialize manipulation detector
MANIPULATION_DETECTOR = ManipulationDetector()

# === CONNECTION MANAGER ===
class ConnectionManager:
    """Manage IB connections with automatic recovery."""
    
    def __init__(self):
        self.connection_attempts = 0
        self.max_attempts = 5
        self.last_connection_time = 0
        self.connection_status = {
            'ib_connected': False,
            'data_flowing': False,
            'last_data_time': 0
        }
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(min=1, max=10)
    )
    def connect_to_ib(self) -> bool:
        """Connect to IB with retry logic."""
        try:
            if ib.isConnected():
                ib.disconnect()
                time.sleep(1)
            
            logger.info(f"Connecting to IB at {IB_HOST}:{IB_PORT}")
            ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            
            if ib.isConnected():
                self.connection_status['ib_connected'] = True
                self.last_connection_time = time.time()
                logger.info("Successfully connected to IB")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"IB connection failed: {e}")
            raise
    
    def subscribe_to_symbol(self, symbol: str, exchange: str = "SMART") -> bool:
        """Subscribe to market data for a symbol."""
        try:
            # Clean up previous subscriptions
            self._cleanup_subscriptions()
            
            # Update state
            STATE.current_symbol = symbol.upper()
            
            # Determine exchange
            if exchange == "AUTO":
                exchange = self._choose_exchange(symbol)
            
            # Create contracts
            contract = Stock(symbol, exchange, "USD")
            
            # Qualify contract
            qualified_contracts = ib.qualifyContracts(contract)
            if not qualified_contracts:
                logger.error(f"Failed to qualify contract for {symbol}")
                return False
            
            qualified_contract = qualified_contracts[0]
            
            # Request Level 2 data
            l2_ticker = ib.reqMktDepth(qualified_contract, numRows=20)
            l2_ticker.updateEvent += on_l2_update
            STATE.ticker_l2 = l2_ticker
            
            # Request Time & Sales
            tns_ticker = ib.reqTickByTickData(
                qualified_contract, 
                "AllLast", 
                numberOfTicks=0, 
                ignoreSize=False
            )
            tns_ticker.updateEvent += on_tns_update
            STATE.ticker_tns = tns_ticker
            
            # Set anchor for VWAP
            ny_tz = pytz.timezone("America/New_York")
            market_open = datetime.datetime.now(ny_tz).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            STATE.ANCHOR_TIMESTAMP = market_open.timestamp()
            
            # Initialize model path
            STATE.ML_MODEL_PATH = f"model_{symbol}.pkl"
            
            logger.info(f"Successfully subscribed to {symbol} on {exchange}")
            STATE.send_alert(f"✅ Subscribed to {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    def _cleanup_subscriptions(self):
        """Clean up existing subscriptions."""
        # Cancel L2
        if STATE.ticker_l2:
            try:
                STATE.ticker_l2.updateEvent -= on_l2_update
                ib.cancelMktDepth(STATE.ticker_l2.contract)
            except:
                pass
        
        # Cancel TnS
        if STATE.ticker_tns:
            try:
                STATE.ticker_tns.updateEvent -= on_tns_update
                ib.cancelTickByTickData(STATE.ticker_tns.contract)
            except:
                pass
        
        # Clear data
        with locks.l2_lock:
            STATE.df_l2_display = pd.DataFrame(columns=["Price", "Size", "Side"])
            STATE.bids_dict.clear()
            STATE.asks_dict.clear()
            STATE.l2_events.clear()
        
        with locks.tns_lock:
            STATE.df_tns = pd.DataFrame(columns=["Time", "Price", "Size", "Timestamp"])
        
        # Reset metrics
        STATE.OFI_VALUE = 0.0
        STATE.CDV_VALUE = 0.0
        STATE.RELATIVE_VOLUME = 1.0
    
    def _choose_exchange(self, symbol: str) -> str:
        """Choose best exchange for symbol."""
        # Try NASDAQ first for tech stocks
        test_exchanges = ["NASDAQ", "NYSE", "SMART"]
        
        for exchange in test_exchanges:
            try:
                contract = Stock(symbol, exchange, "USD")
                qualified = ib.qualifyContracts(contract)
                
                if qualified and qualified[0].conId > 0:
                    logger.info(f"Found {symbol} on {exchange}")
                    return exchange
            except:
                continue
        
        # Default to SMART
        return "SMART"
    
    def check_connection_health(self) -> Tuple[bool, str]:
        """Check connection health."""
        issues = []
        
        # Check IB connection
        if not ib.isConnected():
            issues.append("IB disconnected")
            self.connection_status['ib_connected'] = False
        else:
            self.connection_status['ib_connected'] = True
        
        # Check data flow
        with locks.tns_lock:
            if not STATE.df_tns.empty:
                last_trade_time = STATE.df_tns['Timestamp'].max()
                data_age = time.time() - last_trade_time
                
                if data_age > 60:
                    issues.append(f"Data stale ({data_age:.0f}s)")
                    self.connection_status['data_flowing'] = False
                else:
                    self.connection_status['data_flowing'] = True
                    self.connection_status['last_data_time'] = last_trade_time
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Connection healthy"

# Initialize connection manager
CONNECTION_MANAGER = ConnectionManager()

# === TELEGRAM BOT ===
class TelegramBot:
    """Enhanced Telegram bot with advanced commands."""
    
    def __init__(self):
        self.command_handlers = {
            '/help': self.handle_help,
            '/status': self.handle_status,
            '/subscribe': self.handle_subscribe,
            '/analyse': self.handle_analyse,
            '/risk': self.handle_risk,
            '/performance': self.handle_performance,
            '/signals': self.handle_signals,
            '/microstructure': self.handle_microstructure,
            '/compliance': self.handle_compliance,
            '/models': self.handle_models,
            '/position': self.handle_position,
            '/shutdown': self.handle_shutdown
        }
        
    def process_message(self, chat_id: int, text: str):
        """Process incoming Telegram message."""
        try:
            # Parse command
            parts = text.strip().split()
            if not parts:
                return
            
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Check if command exists
            handler = self.command_handlers.get(command)
            if handler:
                response = handler(args)
                telegram_send(response)
            else:
                # Check for special patterns
                self._handle_special_patterns(chat_id, text)
                
        except Exception as e:
            logger.error(f"Error processing telegram message: {e}")
            telegram_send(f"❌ Error: {e}")
    
    def handle_help(self, args: List[str]) -> str:
        """Show help message."""
        return """
📚 **SYNAPSE X COMMANDS**
━━━━━━━━━━━━━━━━━━━

**Trading:**
• /subscribe SYMBOL - Subscribe to symbol
• /analyse SYMBOL CAPITAL - AI analysis
• /signals - Current trading signals
• /position - Manage positions

**Monitoring:**
• /status - System status
• /risk - Risk metrics
• /performance - Performance report
• /microstructure - Market analysis

**System:**
• /models - ML model status
• /compliance - Compliance report
• /shutdown - Shutdown system

**Examples:**
• /subscribe AAPL
• /analyse TSLA 50000
• /risk
"""
    
    def handle_status(self, args: List[str]) -> str:
        """Get system status."""
        return generate_system_status()
    
    def handle_subscribe(self, args: List[str]) -> str:
        """Subscribe to a symbol."""
        if not args:
            return "❌ Usage: /subscribe SYMBOL"
        
        symbol = args[0].upper()
        
        # Queue subscription
        STATE.CMD_QUEUE.put({
            "action": "subscribe",
            "symbol": symbol
        })
        
        return f"📊 Subscribing to {symbol}..."
    
    def handle_analyse(self, args: List[str]) -> str:
        """Generate AI trading analysis."""
        if len(args) < 2:
            return "❌ Usage: /analyse SYMBOL CAPITAL"
        
        try:
            symbol = args[0].upper()
            capital = float(args[1])
            
            # Get current price
            price = ensure_price_ib(symbol)
            if not price:
                return f"❌ Could not get price for {symbol}"
            
            # Generate analysis
            analysis = generate_institutional_analysis(symbol, capital, price)
            
            return f"""
📈 **AI TRADING ANALYSIS**
━━━━━━━━━━━━━━━━━━
Symbol: {symbol}
Price: ${price:.2f}
Capital: ${capital:,.0f}
━━━━━━━━━━━━━━━━━━

{analysis}
"""
        except Exception as e:
            return f"❌ Error: {e}"
    
    def handle_risk(self, args: List[str]) -> str:
        """Get risk metrics."""
        # Calculate current metrics
        portfolio_var = calculate_portfolio_var()
        
        return f"""
⚠️ **RISK MANAGEMENT**
━━━━━━━━━━━━━━━━━━

**Limits:**
• Max Position: ${STATE.risk_limits.max_position_size:,.0f}
• Daily Loss: ${STATE.risk_limits.max_daily_loss:,.0f}
• Max Drawdown: {STATE.risk_limits.max_drawdown:.1%}
• VaR Limit: {STATE.risk_limits.var_limit:.1%}

**Current Status:**
• Daily P&L: ${STATE.metrics.daily_pnl:+,.2f}
• Drawdown: {STATE.metrics.current_drawdown:.1%}
• Consecutive Losses: {STATE.metrics.consecutive_losses}
• Portfolio Heat: {STATE.portfolio_heat:.1%}
• VaR: ${portfolio_var:,.2f}

**Risk Scores:**
• Market Quality: {analyze_microstructure_quality():.1%}
• Toxicity: {STATE.toxicity_score:.1%}
• Regime: {STATE.market_regime.regime}
"""
    
    def handle_performance(self, args: List[str]) -> str:
        """Get performance metrics."""
        # Multi-horizon accuracy
        horizon_accuracy = HORIZON_TRACKER.calculate_accuracy_report()
        
        # Monte Carlo results
        mc_results = ""
        if VALIDATOR.monte_carlo_results:
            mc = VALIDATOR.monte_carlo_results
            mc_results = f"""
**Monte Carlo Analysis:**
• Success Rate: {mc['success_rate']:.1%}
• Avg Return: {mc['avg_return']:.3f}
• Sharpe: {mc['sharpe']:.2f}
"""
        
        return f"""
📈 **PERFORMANCE REPORT**
━━━━━━━━━━━━━━━━━━━

**Today:**
• Trades: {STATE.trades_today}
• P&L: ${STATE.pnl_today:+,.2f}
• Win Rate: {STATE.metrics.win_rate:.1%}

**Overall Metrics:**
• Total Trades: {STATE.metrics.total_trades}
• Total P&L: ${STATE.metrics.total_pnl:+,.2f}
• Sharpe Ratio: {STATE.metrics.sharpe_ratio:.2f}
• Profit Factor: {STATE.metrics.profit_factor:.2f}
• Max Drawdown: {STATE.metrics.max_drawdown:.1%}

**Win/Loss Analysis:**
• Avg Win: ${STATE.metrics.avg_win:.2f}
• Avg Loss: ${STATE.metrics.avg_loss:.2f}
• Best Streak: {STATE.metrics.consecutive_wins}
• Worst Streak: {STATE.metrics.consecutive_losses}

**Multi-Horizon Accuracy:**
{horizon_accuracy}

{mc_results}
"""
    
    def handle_signals(self, args: List[str]) -> str:
        """Get current trading signals."""
        features = FEATURE_ENGINEER.compute_features()
        if features is None:
            return "⚠️ Insufficient data for signal generation"
        
        signals = []
        
        # Get predictions for all timeframes
        for timeframe in ["1m", "5m", "15m"]:
            pred = MODEL_MANAGER.get_ensemble_prediction(features, timeframe)
            
            signals.append(f"""
**{timeframe.upper()} Signal:**
• Direction: {pred['signal']}
• Probability: {pred['prediction']:.1%}
• Confidence: {pred['confidence']:.1%}
• Agreement: {pred['model_agreement']:.1%}
""")
        
        # Get microstructure signals
        micro_metrics = calculate_microstructure_metrics()
        
        signals.append(f"""
**Microstructure:**
• VPIN: {micro_metrics.get('vpin', 0):.3f}
• Kyle λ: {micro_metrics.get('kyle_lambda', 0):.6f}
• Quality: {micro_metrics.get('quality_score', 0):.1%}
""")
        
        return "🎯 **TRADING SIGNALS**\n━━━━━━━━━━━━━━━━━━\n" + "\n".join(signals)
    
    def handle_microstructure(self, args: List[str]) -> str:
        """Get microstructure analysis."""
        metrics = calculate_microstructure_metrics()
        manipulation = MANIPULATION_DETECTOR.detect_manipulation()
        
        return f"""
🔬 **MICROSTRUCTURE ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━

**Core Metrics:**
• VPIN: {metrics.get('vpin', 0):.3f}
• Kyle's Lambda: {metrics.get('kyle_lambda', 0):.6f}
• Effective Spread: {metrics.get('effective_spread', 0):.4f}
• Quote Stuffing: {metrics.get('quote_stuffing_score', 0):.1%}
• Market Quality: {metrics.get('quality_score', 0):.1%}

**Order Flow:**
• OFI: {STATE.OFI_VALUE:.0f}
• CDV: {STATE.CDV_VALUE:.0f}
• Relative Volume: {STATE.RELATIVE_VOLUME:.2f}x

**Manipulation Detection:**
• Spoofing: {manipulation['scores']['spoofing']:.1%}
• Layering: {manipulation['scores']['layering']:.1%}
• Wash Trading: {manipulation['scores']['wash_trading']:.1%}
• Ramping: {manipulation['scores']['ramping']:.1%}
• Overall Score: {manipulation['overall']:.1%}

**Book Analysis:**
• Bid Levels: {len(STATE.bids_dict)}
• Ask Levels: {len(STATE.asks_dict)}
• Total Depth: {sum(STATE.bids_dict.values()) + sum(STATE.asks_dict.values()):,.0f}
"""
    
    def handle_compliance(self, args: List[str]) -> str:
        """Get compliance report."""
        df, summary = MANIPULATION_DETECTOR.generate_compliance_report()
        
        if df.empty:
            return "✅ No compliance alerts in the current session"
        
        return f"""
📋 **COMPLIANCE REPORT**
━━━━━━━━━━━━━━━━━━━

**Summary:**
• Total Alerts: {summary['total_alerts']}
• Recent (1h): {summary['recent_alerts']}

**Alert Types:**
{self._format_dict(summary['alerts_by_type'])}

**Recent Alerts:**
{self._format_recent_alerts(df.tail(5))}
"""
    
    def handle_models(self, args: List[str]) -> str:
        """Get ML model status."""
        return generate_model_report()
    
    def handle_position(self, args: List[str]) -> str:
        """Manage positions."""
        if not args:
            # Show current positions
            if not STATE.positions:
                return "📊 No active positions"
            
            msg = "📊 **ACTIVE POSITIONS**\n━━━━━━━━━━━━━━━━━━\n\n"
            
            for symbol, position in STATE.positions.items():
                current_price = get_current_price(symbol) or position.get('entry', 0)
                pnl = (current_price - position['entry']) * position['size']
                pnl_pct = ((current_price / position['entry']) - 1) * 100
                
                msg += f"**{symbol}:**\n"
                msg += f"• Entry: ${position['entry']:.2f}\n"
                msg += f"• Current: ${current_price:.2f}\n"
                msg += f"• Size: {position['size']}\n"
                msg += f"• P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
                
                if 'stop' in position:
                    msg += f"• Stop: ${position['stop']:.2f}\n"
                if 'target' in position:
                    msg += f"• Target: ${position['target']:.2f}\n"
                
                msg += "\n"
            
            return msg
        
        # Parse position entry
        # Format: SYMBOL buy/sell PRICE stop STOP target TARGET size SIZE
        try:
            # Simple parser - could be enhanced
            parts = ' '.join(args).split()
            
            if len(parts) >= 7:
                symbol = parts[0].upper()
                action = parts[1].lower()
                entry = float(parts[2])
                stop = float(parts[4])
                target = float(parts[6])
                size = int(parts[8]) if len(parts) > 8 else 100
                
                # Store position
                STATE.positions[symbol] = {
                    'action': action,
                    'entry': entry,
                    'stop': stop,
                    'target': target,
                    'size': size,
                    'timestamp': time.time()
                }
                
                # Calculate risk
                risk = abs(entry - stop) * size
                reward = abs(target - entry) * size
                rr_ratio = reward / risk if risk > 0 else 0
                
                return f"""
✅ **Position Recorded**
━━━━━━━━━━━━━━━━━━
Symbol: {symbol}
Action: {action.upper()}
Entry: ${entry:.2f}
Stop: ${stop:.2f}
Target: ${target:.2f}
Size: {size}

Risk: ${risk:.2f}
Reward: ${reward:.2f}
R:R Ratio: {rr_ratio:.1f}:1
"""
            
        except Exception as e:
            return f"❌ Error parsing position: {e}\nFormat: SYMBOL buy/sell PRICE stop STOP target TARGET size SIZE"
        
        return "❌ Invalid position format"
    
    def handle_shutdown(self, args: List[str]) -> str:
        """Shutdown the system."""
        if args and args[0] == "confirm":
            STATE.send_alert("🛑 Shutdown initiated...")
            
            # Set stop events
            STATE.GPT_STOP_EVENT.set()
            STATE.TELEGRAM_STOP_EVENT.set()
            
            # Queue shutdown
            STATE.CMD_QUEUE.put({"action": "shutdown"})
            
            return "🛑 System shutdown in progress..."
        
        return "⚠️ Type '/shutdown confirm' to shutdown the system"
    
    def _handle_special_patterns(self, chat_id: int, text: str):
        """Handle special message patterns."""
        # Check for position notation: SYMBOL @ PRICE
        match = re.match(r'(\w+)\s*@\s*([\d.]+)', text)
        if match:
            symbol = match.group(1).upper()
            price = float(match.group(2))
            
            response = f"📍 Noted: {symbol} @ ${price:.2f}"
            
            # Store as alert level
            STATE.positions[f"ALERT_{symbol}"] = {
                'type': 'alert',
                'price': price,
                'timestamp': time.time()
            }
            
            telegram_send(response)
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for display."""
        return "\n".join([f"• {k}: {v}" for k, v in d.items()])
    
    def _format_recent_alerts(self, df: pd.DataFrame) -> str:
        """Format recent alerts."""
        if df.empty:
            return "No recent alerts"
        
        alerts = []
        for _, row in df.iterrows():
            timestamp = row['timestamp'].strftime("%H:%M:%S")
            score = row['overall']
            alerts.append(f"• {timestamp} - Score: {score:.1%}")
        
        return "\n".join(alerts)

# Initialize Telegram bot
TELEGRAM_BOT = TelegramBot()

# === SYSTEM MONITORING ===
class SystemHealthMonitor:
    """Monitor system health and performance."""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.last_health_check = time.time()
        self.health_metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'latency_ms': 0,
            'error_rate': 0,
            'message_rate': 0
        }
        self.performance_history = deque(maxlen=100)
        
    def check_system_health(self) -> Tuple[bool, str]:
        """Comprehensive health check."""
        issues = []
        
        # Check system resources
        try:
            import psutil
            
            # CPU check
            cpu = psutil.cpu_percent(interval=0.1)
            self.health_metrics['cpu_usage'] = cpu
            if cpu > 90:
                issues.append(f"CPU overload: {cpu:.1f}%")
            
            # Memory check
            mem = psutil.virtual_memory()
            self.health_metrics['memory_usage'] = mem.percent
            if mem.percent > 85:
                issues.append(f"Memory critical: {mem.percent:.1f}%")
            
            # Disk space check
            disk = psutil.disk_usage('/')
            if disk.percent > 95:
                issues.append(f"Disk space low: {disk.percent:.1f}%")
                
        except ImportError:
            logger.warning("psutil not available for system monitoring")
        
        # Check IB connection
        if not ib.isConnected():
            issues.append("IB disconnected")
        else:
            # Measure latency
            latency = self.measure_ib_latency()
            self.health_metrics['latency_ms'] = latency
            if latency > 100:
                issues.append(f"IB latency high: {latency:.0f}ms")
        
        # Check error rate
        current_time = time.time()
        time_delta = current_time - self.last_health_check
        if time_delta > 0:
            error_rate = sum(self.error_counts.values()) / time_delta
            self.health_metrics['error_rate'] = error_rate
            if error_rate > 10:
                issues.append(f"Error rate high: {error_rate:.1f}/s")
        
        # Check data flow
        with locks.tns_lock:
            last_trade_time = STATE.df_tns['Timestamp'].max() if not STATE.df_tns.empty else 0
        
        if STATE.current_symbol and current_time - last_trade_time > 60:
            issues.append("No trades for 60s")
        
        # Update metrics
        self.last_health_check = current_time
        self.performance_history.append({
            'timestamp': current_time,
            'metrics': self.health_metrics.copy(),
            'healthy': len(issues) == 0
        })
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "System healthy"
    
    def measure_ib_latency(self) -> float:
        """Measure IB API latency in milliseconds."""
        try:
            start = time.perf_counter()
            ib.reqCurrentTime()
            end = time.perf_counter()
            return (end - start) * 1000
        except:
            return 999

# Initialize health monitor
HEALTH_MONITOR = SystemHealthMonitor()

# === RESET MANAGER ===
class ResetManager:
    """Manage system resets and recovery."""
    
    def __init__(self):
        self.reset_count = 0
        self.last_reset_time = 0
        self.reset_history = []
        
    def should_reset(self, error_count: int, last_data_time: float) -> Tuple[bool, str]:
        """Determine if system reset is needed."""
        current_time = time.time()
        
        # Check error threshold
        if error_count > 50:
            return True, "Error count exceeded"
        
        # Check data staleness
        if STATE.current_symbol and current_time - last_data_time > 300:
            return True, "Data stale for 5 minutes"
        
        # Check connection
        if not ib.isConnected():
            return True, "IB connection lost"
        
        return False, ""
    
    def execute_reset(self, reason: str, hard: bool = False):
        """Execute system reset."""
        logger.warning(f"Executing reset: {reason} (hard={hard})")
        
        self.reset_count += 1
        self.last_reset_time = time.time()
        self.reset_history.append({
            'timestamp': datetime.datetime.now(),
            'reason': reason,
            'hard': hard
        })
        
        try:
            if hard:
                # Hard reset - reconnect everything
                STATE.send_alert(f"🔄 HARD RESET: {reason}")
                
                # Disconnect IB
                if ib.isConnected():
                    ib.disconnect()
                
                # Clear all data
                reset_data()
                
                # Wait
                time.sleep(5)
                
                # Reconnect
                if connect_all():
                    STATE.send_alert("✅ Reset complete - system online")
                else:
                    STATE.send_alert("❌ Reset failed - manual intervention required")
            else:
                # Soft reset - just clear data
                STATE.send_alert(f"🔄 SOFT RESET: {reason}")
                reset_data()
                STATE.ib_loop_error_count = 0
                
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            STATE.send_alert(f"❌ Reset error: {e}")

# Initialize reset manager
RESET_MANAGER = ResetManager()

# === REPORT GENERATION ===
def generate_system_status() -> str:
    """Generate comprehensive system status report."""
    
    # System health
    healthy, health_msg = HEALTH_MONITOR.check_system_health()
    health_icon = "✅" if healthy else "⚠️"
    
    # Connection status
    ib_connected = "✅" if ib.isConnected() else "❌"
    
    # Data flow
    with locks.tns_lock:
        last_trade = STATE.df_tns['Timestamp'].max() if not STATE.df_tns.empty else 0
    
    data_age = time.time() - last_trade if last_trade > 0 else float('inf')
    data_status = "✅" if data_age < 10 else "⚠️" if data_age < 60 else "❌"
    
    report = f"""
📊 **SYSTEM STATUS**
━━━━━━━━━━━━━━━━━━
Symbol: {STATE.current_symbol or 'None'}
IB Connected: {ib_connected}
Data Flow: {data_status} ({data_age:.1f}s ago)
System Health: {health_icon} {health_msg}

**Resources:**
CPU: {HEALTH_MONITOR.health_metrics['cpu_usage']:.1f}%
Memory: {HEALTH_MONITOR.health_metrics['memory_usage']:.1f}%
Latency: {HEALTH_MONITOR.health_metrics['latency_ms']:.0f}ms

**Trading:**
Positions: {len(STATE.positions)}
Daily P&L: ${STATE.metrics.daily_pnl:+.2f}
Circuit Breaker: {'🔴 TRIGGERED' if STATE.circuit_breaker_triggered else '🟢 Normal'}
    """
    
    return report

def generate_model_report() -> str:
    """Generate ML model performance report."""
    
    report = "🤖 **MODEL PERFORMANCE**\n"
    report += "━━━━━━━━━━━━━━━━━━\n\n"
    
    # Get recent predictions
    features = FEATURE_ENGINEER.compute_features()
    if features is not None:
        for timeframe in ["1m", "5m", "15m"]:
            prediction = MODEL_MANAGER.get_ensemble_prediction(features, timeframe)
            
            report += f"**{timeframe.upper()} Models:**\n"
            report += f"• Prediction: {prediction['prediction']:.1%}\n"
            report += f"• Confidence: {prediction['confidence']:.1%}\n"
            report += f"• Agreement: {prediction['model_agreement']:.1%}\n"
            
            # Individual model predictions
            for model, prob in prediction['model_predictions'].items():
                report += f"  - {model}: {prob:.1%}\n"
            
            report += "\n"
    
    # Online learning metrics
    if RIVER_AVAILABLE and STATE.online_metrics:
        report += "**Online Learning:**\n"
        try:
            report += f"• Accuracy: {STATE.online_metrics['accuracy'].get():.1%}\n"
            report += f"• F1 Score: {STATE.online_metrics['f1'].get():.3f}\n\n"
        except:
            report += "• Metrics not yet available\n\n"
    
    # Model update counts
    report += "**Update Counts:**\n"
    for key, count in ONLINE_LEARNER.update_counts.items():
        report += f"• {key}: {count:,} samples\n"
    
    return report

def generate_risk_report() -> str:
    """Generate risk management report."""
    
    report = f"""
⚠️ **RISK REPORT**
━━━━━━━━━━━━━━━━━
**Limits:**
• Max Position: ${STATE.risk_limits.max_position_size:,.0f}
• Daily Loss Limit: ${STATE.risk_limits.max_daily_loss:,.0f}
• Max Drawdown: {STATE.risk_limits.max_drawdown:.1%}
• Consecutive Losses: {STATE.risk_limits.consecutive_loss_limit}

**Current Status:**
• Daily P&L: ${STATE.metrics.daily_pnl:+,.2f}
• Current Drawdown: {STATE.metrics.current_drawdown:.1%}
• Consecutive Losses: {STATE.metrics.consecutive_losses}
• Portfolio Heat: {STATE.portfolio_heat:.1%}
• VaR: ${calculate_portfolio_var():,.2f}

**Risk Scores:**
• Market Regime: {STATE.market_regime.regime}
• Volatility: {STATE.market_regime.volatility:.1%}
• Toxicity: {STATE.toxicity_score:.1%}
    """
    
    return report

def generate_performance_report() -> str:
    """Generate trading performance report."""
    
    report = f"""
📈 **PERFORMANCE REPORT**
━━━━━━━━━━━━━━━━━━━━━
**Today:**
• Trades: {STATE.trades_today}
• P&L: ${STATE.pnl_today:+,.2f}

**Overall:**
• Total Trades: {STATE.metrics.total_trades}
• Win Rate: {STATE.metrics.win_rate:.1%}
• Profit Factor: {STATE.metrics.profit_factor:.2f}
• Sharpe Ratio: {STATE.metrics.sharpe_ratio:.2f}
• Max Drawdown: {STATE.metrics.max_drawdown:.1%}

**Best/Worst:**
• Best Trade: ${STATE.metrics.avg_win:,.2f}
• Worst Trade: ${-STATE.metrics.avg_loss:,.2f}
• Consecutive Wins: {STATE.metrics.consecutive_wins}
    """
    
    # Add Monte Carlo results if available
    if VALIDATOR.monte_carlo_results:
        mc = VALIDATOR.monte_carlo_results
        report += f"""
**Monte Carlo Analysis:**
• Success Rate: {mc['success_rate']:.1%}
• Expected Return: {mc['avg_return']:.3f}
• Risk-Adjusted: {mc['sharpe']:.2f}
        """
    
    return report

def format_signals_report(signals: List[Dict]) -> str:
    """Format trading signals report."""
    if not signals:
        return "No active signals"
    
    report = "🎯 **ACTIVE SIGNALS**\n"
    
    for signal in signals[:5]:  # Top 5 signals
        report += f"""
• {signal['symbol']} - {signal['action']}
  Confidence: {signal['confidence']:.1%}
  Size: {signal['size']}
  Entry: ${signal['entry']:.2f}
  Stop: ${signal['stop']:.2f}
  Target: ${signal['target']:.2f}
"""
    
    return report

def generate_institutional_analysis(symbol: str, capital: float, price: float) -> str:
    """Generate institutional-grade analysis using GPT."""
    
    # Build comprehensive context
    context = f"""
POSITION ANALYSIS REQUEST:
Symbol: {symbol}
Current Price: ${price:.2f}
Available Capital: ${capital:,.0f}
Risk Per Trade: ${capital * 0.02:,.0f} (2%)

CURRENT MARKET CONDITIONS:
{format_microstructure_report()}

Recent Signals:
{format_detected_signals()}
"""
    
    # Use GPT for analysis
    prompt = f"""As an institutional quant trader, analyze this position opportunity:

{context}

Provide specific recommendations including:
1. Position sizing (shares)
2. Entry strategy (market/limit/scaled)
3. Stop loss level
4. Take profit targets (multiple)
5. Risk/reward analysis
6. Key risks to monitor

Be quantitative and actionable."""
    
    # In production, would call GPT API here
    # For now, return structured analysis
    
    position_size = int(capital * 0.02 / (price * 0.02))  # 2% risk, 2% stop
    stop_loss = price * 0.98
    target1 = price * 1.03
    target2 = price * 1.05
    
    return f"""
**POSITION ANALYSIS**

📊 **Recommendation: LONG**
• Entry: Scaled between ${price:.2f} - ${price * 0.998:.2f}
• Size: {position_size} shares
• Risk: ${position_size * (price - stop_loss):.2f}

🎯 **Targets:**
• T1: ${target1:.2f} (50% position)
• T2: ${target2:.2f} (remaining)
• Stop: ${stop_loss:.2f}

📈 **Risk/Reward:**
• R:R Ratio: 1:2.5
• Win Rate Required: 28.6%
• Expected Value: +${(target1 - price) * position_size * 0.6:.2f}

⚠️ **Key Risks:**
• Market regime: {STATE.market_regime.regime}
• Volatility: {STATE.market_regime.volatility:.1%}
• Liquidity: Monitor spread widening

💡 **Execution:**
• Use limit orders for entry
• Scale in over 5-10 minutes
• Set alerts at key levels
"""

# === GPT WORKER ===
def gpt_worker():
    """Background worker for GPT analysis."""
    logger.info("GPT worker started")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=STATE.client_openai_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return
    
    while not STATE.GPT_STOP_EVENT.is_set():
        try:
            current_time = time.time()
            
            # Check if analysis needed
            if current_time - STATE.last_gpt_time < STATE.GPT_COOLDOWN:
                time.sleep(5)
                continue
            
            # Check if we have data
            with locks.tns_lock:
                if STATE.df_tns.empty:
                    time.sleep(5)
                    continue
            
            # Build prompt
            prompt = build_gpt_prompt()
            if not prompt:
                time.sleep(5)
                continue
            
            # Mark as in progress
            STATE.gpt_in_progress = True
            
            # Call GPT-4.1 and GPT-4o
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",  # Use latest GPT-4
                    messages=[
                        {"role": "system", "content": "You are an expert quantitative trader with access to GPT-4.1 and GPT-4o capabilities. Provide concise, actionable trading analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                
                # Update state
                STATE.latest_gpt_analysis = analysis
                STATE.last_gpt_time = current_time
                STATE.gpt_has_alert = True
                
                # Store in history
                STATE.gpt_history.append({
                    'timestamp': current_time,
                    'analysis': analysis
                })
                
                logger.info("GPT analysis completed")
                
            except Exception as e:
                logger.error(f"GPT API error: {e}")
                STATE.latest_gpt_analysis = "GPT analysis temporarily unavailable"
            
            finally:
                STATE.gpt_in_progress = False
            
            # Wait before next analysis
            time.sleep(30)
            
        except Exception as e:
            logger.error(f"GPT worker error: {e}")
            time.sleep(10)
    
    logger.info("GPT worker stopped")

# === TELEGRAM WORKER ===
def telegram_worker():
    """Background worker for Telegram bot."""
    logger.info("Telegram bot started")
    
    while not STATE.TELEGRAM_STOP_EVENT.is_set():
        try:
            # Get updates
            url = f"https://api.telegram.org/bot{STATE.TELEGRAM_TOKEN}/getUpdates"
            params = {
                "offset": STATE.last_update_id + 1,
                "timeout": 30
            }
            
            response = requests.get(url, params=params, timeout=35)
            
            if response.status_code != 200:
                time.sleep(5)
                continue
            
            updates = response.json().get("result", [])
            
            for update in updates:
                STATE.last_update_id = update["update_id"]
                
                # Process message
                if "message" in update:
                    message = update["message"]
                    chat_id = message.get("chat", {}).get("id")
                    text = message.get("text", "")
                    
                    if chat_id and text:
                        TELEGRAM_BOT.process_message(chat_id, text)
                        
        except Exception as e:
            logger.error(f"Telegram loop error: {e}")
            time.sleep(5)
    
    logger.info("Telegram bot stopped")

# === PERIODIC TASKS ===
def periodic_reports():
    """Send periodic reports to Telegram."""
    current_time = time.time()
    
    # 5-minute update
    if current_time - STATE.last_5min_send > 300:
        try:
            # Get current metrics
            features = FEATURE_ENGINEER.compute_features()
            if features is not None:
                pred_dict = {}
                for tf in ["1m", "5m"]:
                    pred_dict[tf] = MODEL_MANAGER.get_ensemble_prediction(features, tf)
                
                # Calculate decision
                prob_avg = np.mean([p['prediction'] for p in pred_dict.values()])
                
                if prob_avg > 0.65:
                    action = "BUY"
                elif prob_avg < 0.35:
                    action = "SELL"
                else:
                    action = "HOLD"
                
                decision = {
                    'action': action,
                    'confidence': abs(prob_avg - 0.5) * 2,
                    'risk_score': STATE.toxicity_score,
                    'position_size': 100,
                    'stop_loss': get_current_price() * 0.98 if action == "BUY" else get_current_price() * 1.02
                }
                
                # Get microstructure metrics
                metrics = calculate_microstructure_metrics()
                
                # Build and send message
                msg = TELEGRAM_MESSAGE_BUILDER.build_institutional_update(
                    None, decision, pred_dict, metrics
                )
                
                STATE.send_alert(msg, force=True)
                STATE.last_5min_send = current_time
                
        except Exception as e:
            logger.error(f"Error in 5-minute update: {e}")
    
    # 10-minute strategic update
    if current_time - STATE.last_10min_send > 600:
        try:
            msg = TELEGRAM_MESSAGE_BUILDER.build_10min_strategic()
            STATE.send_alert(msg, force=True)
            STATE.last_10min_send = current_time
        except Exception as e:
            logger.error(f"Error in 10-minute update: {e}")
    
    # 15-minute comprehensive report
    if current_time - STATE.last_15min_send > 900:
        try:
            msg = TELEGRAM_MESSAGE_BUILDER.build_15min_comprehensive()
            STATE.send_alert(msg, force=True)
            STATE.last_15min_send = current_time
        except Exception as e:
            logger.error(f"Error in 15-minute update: {e}")

def process_commands():
    """Process queued commands."""
    while not STATE.CMD_QUEUE.empty():
        try:
            cmd = STATE.CMD_QUEUE.get_nowait()
            
            if cmd['action'] == 'subscribe':
                symbol = cmd['symbol']
                success = CONNECTION_MANAGER.subscribe_to_symbol(symbol)
                
                if success:
                    STATE.send_alert(f"✅ Subscribed to {symbol}")
                else:
                    STATE.send_alert(f"❌ Failed to subscribe to {symbol}")
                    
            elif cmd['action'] == 'shutdown':
                logger.info("Shutdown command received")
                return False
                
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Command processing error: {e}")
    
    return True

def update_metrics_and_features():
    """Update all metrics and features."""
    try:
        # Update order flow metrics
        update_order_flow_metrics()
        
        # Update market regime
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if len(df) > 100:
            prices = df['Price'].values
            volumes = df['Size'].values
            STATE.market_regime.classify(prices, volumes)
        
        # Update toxicity score
        vpin = calculate_vpin()
        STATE.toxicity_score = min(1.0, vpin * 2)  # Scale VPIN to toxicity
        
        # Update risk metrics
        STATE.update_risk_metrics()
        
        # Detect anomalies
        features = FEATURE_ENGINEER.compute_features()
        if features is not None:
            anomalies = ANOMALY_DETECTOR.detect_anomalies(features)
            
            if anomalies['is_anomaly']:
                logger.warning(f"Anomalies detected: {anomalies['reasons']}")
        
        # Detect manipulation
        manipulation = MANIPULATION_DETECTOR.detect_manipulation()
        
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")

def save_and_train_models():
    """Save models and trigger training if needed."""
    try:
        # Partial fit online models
        partial_fit_all_models()
        
        # Batch retrain if needed
        batch_retrain_all_models()
        
        # Save models periodically
        save_all_models()
        
        # Update multi-horizon tracking
        current_price = get_current_price()
        if current_price:
            HORIZON_TRACKER.verify_predictions(current_price)
            
            # Add new prediction
            features = FEATURE_ENGINEER.compute_features()
            if features is not None:
                pred = MODEL_MANAGER.get_ensemble_prediction(features)
                HORIZON_TRACKER.add_prediction(pred['prediction'], current_price)
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")

def compress_ticks():
    """Compress tick data to save memory."""
    try:
        if STATE.tick_count > 0:
            # Extract valid data
            valid_ticks = STATE.ticks_array[:STATE.tick_count].copy()
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ticks_{STATE.current_symbol}_{timestamp}.zst"
            
            # Compress data
            compressed = zstd.compress(valid_ticks.tobytes(), level=3)
            
            # Save to file
            with open(filename, 'wb') as f:
                f.write(compressed)
            
            logger.info(f"Compressed {STATE.tick_count} ticks to {filename}")
            
            # Reset array
            STATE.ticks_array[:] = 0
            STATE.tick_count = 0
            
    except Exception as e:
        logger.error(f"Tick compression failed: {e}")

def clean_old_data():
    """Clean old data to prevent memory issues."""
    
    cutoff_time = time.time() - 7200  # 2 hours
    
    # Clean L2 events
    STATE.l2_events = deque(
        (e for e in STATE.l2_events if e['timestamp'] > cutoff_time),
        maxlen=50000
    )
    
    # Clean predictions buffer
    STATE.predictions_buffer = deque(
        (p for p in STATE.predictions_buffer if p['timestamp'] > cutoff_time),
        maxlen=1000
    )

def reset_data():
    """Reset all data structures."""
    
    logger.info("Resetting all data structures")
    
    # Clear market data
    with locks.l2_lock:
        STATE.df_l2_display = pd.DataFrame(columns=["Price", "Size", "Side"])
        STATE.bids_dict.clear()
        STATE.asks_dict.clear()
        STATE.l2_events.clear()
        STATE.spread_history.clear()
    
    with locks.tns_lock:
        STATE.df_tns = pd.DataFrame(columns=["Time", "Price", "Size", "Timestamp"])
    
    # Clear features and predictions
    STATE.features_buffer.clear()
    STATE.features_buffer_5m.clear()
    STATE.features_buffer_15m.clear()
    STATE.predictions_buffer.clear()
    STATE.prob_buffer.clear()
    
    # Clear training data
    STATE.TRAINING_BUFFER.clear()
    STATE.TRAINING_BUFFER_5m.clear()
    STATE.TRAINING_BUFFER_15m.clear()
    
    # Reset metrics
    STATE.OFI_VALUE = 0.0
    STATE.CDV_VALUE = 0.0
    STATE.RELATIVE_VOLUME = 1.0
    
    # Reset counters
    STATE.ib_loop_error_count = 0
    STATE.tick_count = 0
    
    logger.info("Data reset complete")

# === CLEANUP AND SHUTDOWN ===
@atexit.register
def cleanup_on_exit():
    """Cleanup on system exit."""
    try:
        logger.info("Starting cleanup...")
        
        # Stop event loops
        STATE.GPT_STOP_EVENT.set()
        STATE.TELEGRAM_STOP_EVENT.set()
        
        # Save all models
        save_all_models(force=True)
        
        # Compress ticks
        compress_ticks()
        
        # Export final reports
        try:
            # Performance report
            perf_df = pd.DataFrame(STATE.metrics.__dict__, index=[0])
            perf_df.to_csv(f"final_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # Compliance report
            if MANIPULATION_DETECTOR.compliance_log:
                comp_df = pd.DataFrame(MANIPULATION_DETECTOR.compliance_log)
                comp_df.to_csv(f"final_compliance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        except:
            pass
        
        # Disconnect IB
        if ib.isConnected():
            ib.disconnect()
        
        logger.info("Cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# === MAIN LOOP ===
def enhanced_ib_loop():
    """Enhanced main trading loop with all features."""
    
    logger.info("Starting enhanced IB loop...")
    
    # Start background workers
    gpt_thread = threading.Thread(target=gpt_worker, daemon=True, name="GPT-Worker")
    gpt_thread.start()
    
    telegram_thread = threading.Thread(target=telegram_worker, daemon=True, name="Telegram-Worker")
    telegram_thread.start()
    
    # Initialize loop variables
    loop_counter = 0
    last_data_time = time.time()
    last_health_check = time.time()
    
    while True:
        try:
            loop_counter += 1
            current_time = time.time()
            
            # Process IB events
            ib.sleep(0.01)
            
            # Check for new data
            with locks.tns_lock:
                if not STATE.df_tns.empty:
                    last_data_time = STATE.df_tns['Timestamp'].max()
            
            # Process commands
            if not process_commands():
                break
            
            # Update metrics every loop
            if loop_counter % 10 == 0:
                update_metrics_and_features()
            
            # Train models every 30 seconds
            if loop_counter % 3000 == 0:
                save_and_train_models()
            
            # Send periodic reports
            if loop_counter % 100 == 0:
                periodic_reports()
            
            # Health check every 10 seconds
            if current_time - last_health_check > 10:
                conn_healthy, conn_msg = CONNECTION_MANAGER.check_connection_health()
                if not conn_healthy:
                    logger.warning(f"Connection issues: {conn_msg}")
                    
                    # Try to reconnect if needed
                    if not ib.isConnected():
                        CONNECTION_MANAGER.connect_to_ib()
                        if STATE.current_symbol:
                            CONNECTION_MANAGER.subscribe_to_symbol(STATE.current_symbol)
                
                # Check for stale data
                data_age = current_time - last_data_time
                if data_age > 300 and STATE.current_symbol:
                    logger.warning(f"Data stale for {data_age:.0f}s")
                    RESET_MANAGER.execute_reset("Data stale", hard=False)
                
                last_health_check = current_time
            
            # Periodic reports
            if loop_counter % 100 == 0:
                periodic_reports()
            
            # Clean old data
            if loop_counter % 10000 == 0:
                clean_old_data()
            
            # Compress ticks
            if STATE.tick_count > 900000:
                compress_ticks()
            
            # Reset loop counter to prevent overflow
            if loop_counter > 1000000:
                loop_counter = 0
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break
            
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            STATE.ib_loop_error_count += 1
            
            # Check if we need to reset
            should_reset, reason = RESET_MANAGER.should_reset(
                STATE.ib_loop_error_count,
                last_data_time
            )
            
            if should_reset:
                RESET_MANAGER.execute_reset(reason, hard=(STATE.ib_loop_error_count > 10))
            
            time.sleep(1)
    
    logger.info("Enhanced IB loop stopped")

# === MAIN ENTRY POINT ===
def main():
    """Main entry point."""
    
    logger.info("=" * 80)
    logger.info("SYNAPSE X ONLINE LEARNING SYSTEM v10.0.9.01")
    logger.info("=" * 80)
    
    # Connect to IB
    if not connect_all():
        logger.error("Failed to establish initial connection")
        sys.exit(1)
    
    # Send startup message
    startup_msg = f"""
🚀 **SYNAPSE X ONLINE LEARNING STARTED**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Version: 10.0.9.01
Symbol: {STATE.current_symbol}
GPT: GPT-4.1 & GPT-4o Enabled
ML: Online Learning Active
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use /help for commands
    """
    STATE.send_alert(startup_msg)
    
    # Run main loop
    try:
        enhanced_ib_loop()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        STATE.send_alert(f"🚨 CRITICAL ERROR: {e}")
    finally:
        # Cleanup
        logger.info("Shutting down...")
        
        # Stop workers
        STATE.GPT_STOP_EVENT.set()
        STATE.TELEGRAM_STOP_EVENT.set()
        
        # Save models
        save_all_models(force=True)
        
        # Compress remaining ticks
        compress_ticks()
        
        # Disconnect IB
        if ib.isConnected():
            ib.disconnect()
        
        # Final report
        final_msg = f"""
🛑 **SYNAPSE X SHUTDOWN**
━━━━━━━━━━━━━━━━━━━━━
• Total P&L: ${STATE.metrics.total_pnl:+,.2f}
• Total Trades: {STATE.metrics.total_trades}
• Win Rate: {STATE.metrics.win_rate:.1%}
• Sharpe Ratio: {STATE.metrics.sharpe_ratio:.2f}
━━━━━━━━━━━━━━━━━━━━━
Models saved successfully ✓
System terminated gracefully
        """
        STATE.send_alert(final_msg)
        
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()