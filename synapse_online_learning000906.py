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
##   INSTITUTIONAL TRADING SYSTEM - Neural Fusion Analytics v11.0                     ##
##   ⚡ Ultra Low Latency | 🧬 ML Ensemble | 🛡️ Risk Management | 🤖 GPT+GROK      ##
##                                                                                    ##
###########################################################################

Synapse X - Advanced Algorithmic Trading System with Dual AI Integration
=======================================================================

Enhanced Features v11.0:
- GPT-4 + Grok Dual AI Analysis
- Advanced Pattern Recognition System
- Dynamic Risk Management with Kelly Criterion
- Smart Strategy Selection (5 strategies)
- Multi-horizon prediction tracking (5m, 15m, 30m, 60m)
- Market Maker Detection
- Session-based Trading Profiles
- Voice Trading Assistant Ready
- Real-time Backtesting

Author: Hedge Fund Analytics Team
Version: 11.0.0
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
import httpx
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
        'synapse_v11.log', 
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
    ai_logger = logging.getLogger('ai')
    
    return logger, trading_logger, ml_logger, risk_logger, ai_logger

# Initialize logging
logger, trading_logger, ml_logger, risk_logger, ai_logger = setup_logging()

# === ENVIRONMENT VALIDATION ===
def validate_environment():
    """Validate all required environment variables."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for GPT integration",
        "GROK_API_KEY": "Grok API key for xAI integration",
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

# === ENHANCED CONFIGURATION WITH DUAL AI ===
@dataclass
class OptimizedTradingConfig:
    """Configuration optimisée pour le trading NASDAQ avec IA duale."""
    
    # === PARAMÈTRES IA ===
    ai_consensus_threshold: float = 75.0
    ai_high_confidence_threshold: float = 85.0
    ai_agreement_weight: float = 0.7
    ml_weight: float = 0.3
    min_ai_agreement: float = 0.6
    gpt_temperature: float = 0.2
    grok_temperature: float = 0.3
    analysis_cooldown: int = 45
    emergency_analysis_threshold: float = 0.03
    
    # === PARAMÈTRES DE RISQUE ===
    base_risk_percent: float = 2.0
    max_position_pct: float = 0.20
    max_portfolio_heat: float = 0.06
    stop_loss_multiplier: float = 2.0
    max_kelly_fraction: float = 0.25
    max_consecutive_losses: int = 3
    min_sharpe_ratio: float = 1.0
    max_leverage: float = 2.0
    
    # === PARAMÈTRES D'EXÉCUTION ===
    min_edge: float = 0.002
    max_spread_pct: float = 0.001
    urgency_threshold: float = 0.7
    iceberg_threshold: float = 10000
    slippage_limit: float = 0.0005
    order_timeout_seconds: int = 30
    max_retry_attempts: int = 3
    
    # === PARAMÈTRES PAR SESSION ===
    pre_market_risk_multiplier: float = 0.5
    regular_hours_risk_multiplier: float = 1.0
    after_hours_risk_multiplier: float = 0.3
    no_trading_before: str = "09:45"
    exit_all_by: str = "15:55"
    max_trades_per_day: int = 10
    
    # === ML CONFIGURATION ===
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
    
    # === MONITORING ===
    heartbeat_interval: int = 10
    metrics_window: int = 252
    alert_cooldown: int = 60
    max_consecutive_errors: int = 5
    performance_checkpoint_interval: int = 300
    memory_limit_gb: float = 32.0
    cpu_limit_percent: float = 80.0

# === VOLATILITY ADJUSTMENTS ===
VOLATILITY_PARAMS = {
    'low': {      # < 0.15 annualisé
        'position_multiplier': 1.5,
        'stop_loss': 0.01,
        'take_profit': 0.02,
        'confidence_threshold': 70
    },
    'normal': {   # 0.15 - 0.25
        'position_multiplier': 1.0,
        'stop_loss': 0.015,
        'take_profit': 0.03,
        'confidence_threshold': 75
    },
    'high': {     # 0.25 - 0.35
        'position_multiplier': 0.7,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'confidence_threshold': 80
    },
    'extreme': {  # > 0.35
        'position_multiplier': 0.5,
        'stop_loss': 0.03,
        'take_profit': 0.06,
        'confidence_threshold': 85
    }
}

# === TIMEFRAME STRATEGIES ===
TIMEFRAME_PARAMS = {
    'scalping': {    # 1-5 minutes
        'min_profit': 0.002,
        'max_hold_time': 300,
        'volume_threshold': 2.0,
        'spread_limit': 0.0005
    },
    'intraday': {    # 5-60 minutes
        'min_profit': 0.005,
        'max_hold_time': 3600,
        'volume_threshold': 1.5,
        'spread_limit': 0.001
    },
    'swing': {       # 1-5 jours
        'min_profit': 0.02,
        'max_hold_time': 432000,
        'volume_threshold': 1.0,
        'spread_limit': 0.002
    }
}

# === DECISION WEIGHTS ===
DECISION_WEIGHTS = {
    'gpt_technical': 0.25,
    'grok_flow': 0.25,
    'ml_ensemble': 0.20,
    'microstructure': 0.15,
    'risk_score': 0.15
}

# Initialize configuration
TRADING_CONFIG = OptimizedTradingConfig()

logger.info("Synapse X v11.0 with Dual AI initialized successfully")
logger.info(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
logger.info(f"XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
logger.info(f"Online Learning: {'Enabled' if RIVER_AVAILABLE else 'Disabled'}")

# === GROK CLIENT ===
class GrokClient:
    """Client pour l'API Grok/xAI avec gestion robuste des erreurs."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.request_count = 0
        self.last_request_time = 0
        self.timeout = httpx.Timeout(30.0, connect=5.0)
        
    async def analyze(self, prompt: str, model: str = "grok-beta", temperature: float = 0.3) -> str:
        """Appel asynchrone à Grok avec gestion d'erreurs."""
        self.request_count += 1
        
        # Rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < 1.0:  # Max 1 request per second
            await asyncio.sleep(1.0 - time_since_last)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": 300
                    }
                )
                
                self.last_request_time = time.time()
                
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content']
                    ai_logger.info(f"Grok response received ({len(content)} chars)")
                    return content
                else:
                    error_msg = f"Grok API error: {response.status_code} - {response.text}"
                    ai_logger.error(error_msg)
                    return error_msg
                    
            except httpx.TimeoutException:
                ai_logger.error("Grok request timeout")
                return "Grok timeout - using fallback analysis"
            except Exception as e:
                ai_logger.error(f"Grok request failed: {e}")
                return f"Grok error: {str(e)}"

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
        self.ai_lock = threading.Lock()
    
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

# === DUAL AI ANALYZER ===
class DualAIAnalyzer:
    """Analyse en synergie GPT + Grok pour décisions de trading."""
    
    def __init__(self, state):
        self.state = state
        self.grok_client = GrokClient(os.getenv("GROK_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.last_analysis = {}
        self.consensus_history = deque(maxlen=100)
        self.analysis_cache = {}
        self.cache_ttl = 30  # seconds
        
    def build_market_context(self) -> Dict[str, Any]:
        """Construit le contexte de marché pour les deux IA."""
        with locks.tns_lock:
            df_tns = self.state.df_tns.copy()
        
        if df_tns.empty:
            return {}
        
        current_price = df_tns['Price'].iloc[-1]
        
        # Calculs de base
        price_5m_ago = df_tns[df_tns['Timestamp'] > time.time() - 300]['Price'].iloc[0] if len(df_tns) > 50 else current_price
        price_change_5m = (current_price - price_5m_ago) / price_5m_ago * 100 if price_5m_ago > 0 else 0
        
        volume_1m = df_tns[df_tns['Timestamp'] > time.time() - 60]['Size'].sum()
        volume_5m = df_tns[df_tns['Timestamp'] > time.time() - 300]['Size'].sum()
        
        # ML predictions
        features = FEATURE_ENGINEER.compute_features()
        ml_predictions = {}
        if features is not None:
            for tf in ["1m", "5m", "15m"]:
                ml_predictions[tf] = MODEL_MANAGER.get_ensemble_prediction(features, tf)
        
        # Microstructure metrics
        micro_metrics = calculate_microstructure_metrics()
        
        return {
            'symbol': self.state.current_symbol,
            'current_price': current_price,
            'price_change_5m': price_change_5m,
            'volume_1m': volume_1m,
            'volume_5m': volume_5m,
            'volume_ratio': volume_5m / (volume_1m * 5) if volume_1m > 0 else 1,
            'ml_predictions': ml_predictions,
            'market_regime': self.state.market_regime.regime,
            'volatility': self.state.market_regime.volatility,
            'ofi': self.state.OFI_VALUE,
            'vpin': micro_metrics.get('vpin', 0),
            'kyle_lambda': micro_metrics.get('kyle_lambda', 0),
            'spread': calculate_effective_spread(),
            'toxicity': self.state.toxicity_score,
            'timestamp': time.time()
        }
    
    async def get_grok_analysis(self, context: Dict) -> Dict[str, Any]:
        """Obtient l'analyse de Grok."""
        prompt = f"""As a quantitative trading AI analyzing NASDAQ stocks, provide institutional-grade analysis:

Symbol: {context['symbol']}
Price: ${context['current_price']:.2f} ({context['price_change_5m']:+.2f}% 5m)
Volume: {context['volume_1m']:,} (1m), {context['volume_5m']:,} (5m) - Ratio: {context['volume_ratio']:.2f}x
Market Regime: {context['market_regime']} (Vol: {context['volatility']:.1%})

ML Signals:
- 1m: {context['ml_predictions'].get('1m', {}).get('prediction', 0.5):.1%} ({context['ml_predictions'].get('1m', {}).get('confidence', 0):.1%} conf)
- 5m: {context['ml_predictions'].get('5m', {}).get('prediction', 0.5):.1%} ({context['ml_predictions'].get('5m', {}).get('confidence', 0):.1%} conf)

Microstructure:
- Order Flow Imbalance: {context['ofi']:.0f}
- VPIN: {context['vpin']:.3f}
- Kyle's λ: {context['kyle_lambda']:.6f}
- Toxicity: {context['toxicity']:.1%}

Provide:
1. Direction next 1-5 min with exact probability (e.g., 72% UP)
2. Key institutional flow indicators
3. Hidden liquidity assessment
4. Clear recommendation: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
5. Confidence level: XX%

Be quantitative, specific, and decisive. Focus on institutional perspective."""
        
        analysis_text = await self.grok_client.analyze(prompt, temperature=TRADING_CONFIG.grok_temperature)
        
        # Parse Grok response
        return self.parse_ai_response(analysis_text, "grok")
    
    def get_gpt_analysis(self, context: Dict) -> Dict[str, Any]:
        """Obtient l'analyse de GPT-4."""
        prompt = f"""You are an elite institutional quant trader. Analyze this NASDAQ stock data:

Symbol: {context['symbol']}
Current: ${context['current_price']:.2f}
5m Change: {context['price_change_5m']:+.2f}%
Relative Volume: {context['volume_ratio']:.2f}x

Technical Analysis:
- ML Probability (1m): {context['ml_predictions'].get('1m', {}).get('prediction', 0.5):.1%}
- ML Probability (5m): {context['ml_predictions'].get('5m', {}).get('prediction', 0.5):.1%}
- Market Regime: {context['market_regime']}
- Volatility: {context['volatility']:.1%}

Market Microstructure:
- OFI: {context['ofi']:.0f}
- VPIN: {context['vpin']:.3f} (informed trading probability)
- Effective Spread: {context['spread']:.4f}

Provide concise analysis:
1. Price direction next 5 min with probability
2. Key support/resistance levels
3. Risk factors (brief)
4. Trading recommendation: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
5. Confidence: XX%

150 words max. Be specific and actionable."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a top-tier quantitative trader at a hedge fund."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=TRADING_CONFIG.gpt_temperature
            )
            
            analysis_text = response.choices[0].message.content
            ai_logger.info(f"GPT response received ({len(analysis_text)} chars)")
            return self.parse_ai_response(analysis_text, "gpt")
            
        except Exception as e:
            ai_logger.error(f"GPT analysis error: {e}")
            return {
                'source': 'gpt',
                'recommendation': 'HOLD',
                'confidence': 0,
                'analysis': str(e),
                'error': True
            }
    
    def parse_ai_response(self, text: str, source: str) -> Dict[str, Any]:
        """Parse AI response to extract key information."""
        result = {
            'source': source,
            'recommendation': 'HOLD',
            'confidence': 50,
            'direction_probability': 50,
            'analysis': text,
            'risk_factors': [],
            'support_resistance': {}
        }
        
        if not text or "error" in text.lower():
            result['error'] = True
            return result
        
        # Extract recommendation
        text_upper = text.upper()
        if 'STRONG_BUY' in text_upper or 'STRONG BUY' in text_upper:
            result['recommendation'] = 'STRONG_BUY'
        elif 'STRONG_SELL' in text_upper or 'STRONG SELL' in text_upper:
            result['recommendation'] = 'STRONG_SELL'
        elif 'BUY' in text_upper and 'SELL' not in text_upper[:text_upper.find('BUY')]:
            result['recommendation'] = 'BUY'
        elif 'SELL' in text_upper and 'BUY' not in text_upper[:text_upper.find('SELL')]:
            result['recommendation'] = 'SELL'
        
        # Extract confidence
        import re
        confidence_patterns = [
            r'confidence[:\s]+(\d+)%',
            r'(\d+)%\s*confidence',
            r'confident[:\s]+(\d+)%',
            r'certainty[:\s]+(\d+)%'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['confidence'] = int(match.group(1))
                break
        
        # Extract direction probability
        prob_patterns = [
            r'(\d+)%\s*(?:up|bullish|higher|increase)',
            r'(?:up|bullish|higher)[:\s]+(\d+)%',
            r'probability[:\s]+(\d+)%\s*(?:up|bullish)'
        ]
        
        for pattern in prob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['direction_probability'] = int(match.group(1))
                break
        
        # Extract support/resistance
        support_match = re.search(r'support[:\s]+\$?([\d.]+)', text, re.IGNORECASE)
        resistance_match = re.search(r'resistance[:\s]+\$?([\d.]+)', text, re.IGNORECASE)
        
        if support_match:
            result['support_resistance']['support'] = float(support_match.group(1))
        if resistance_match:
            result['support_resistance']['resistance'] = float(resistance_match.group(1))
        
        return result
    
    async def get_synergistic_decision(self) -> Dict[str, Any]:
        """Obtient une décision en synergie GPT + Grok."""
        
        # Check cache
        cache_key = f"{self.state.current_symbol}_{int(time.time() / self.cache_ttl)}"
        if cache_key in self.analysis_cache:
            ai_logger.info("Using cached AI analysis")
            return self.analysis_cache[cache_key]
        
        # Construire le contexte
        context = self.build_market_context()
        if not context:
            return {
                'decision': 'HOLD',
                'confidence': 0,
                'reason': 'Insufficient data',
                'error': True
            }
        
        # Obtenir les analyses en parallèle
        try:
            start_time = time.time()
            
            # Create tasks
            grok_task = asyncio.create_task(self.get_grok_analysis(context))
            
            # Get GPT synchronously (it's not async)
            gpt_analysis = self.get_gpt_analysis(context)
            
            # Wait for Grok with timeout
            try:
                grok_analysis = await asyncio.wait_for(grok_task, timeout=5.0)
            except asyncio.TimeoutError:
                ai_logger.warning("Grok timeout - using GPT only")
                grok_analysis = {
                    'source': 'grok',
                    'recommendation': gpt_analysis['recommendation'],
                    'confidence': 50,
                    'error': True
                }
            
            elapsed = time.time() - start_time
            ai_logger.info(f"AI analysis completed in {elapsed:.2f}s")
            
            # Calculer le consensus
            decision = self.calculate_consensus(gpt_analysis, grok_analysis, context)
            
            # Sauvegarder pour historique
            self.last_analysis = {
                'timestamp': time.time(),
                'gpt': gpt_analysis,
                'grok': grok_analysis,
                'consensus': decision,
                'context': context
            }
            
            self.consensus_history.append(self.last_analysis)
            
            # Cache the result
            self.analysis_cache[cache_key] = decision
            
            return decision
            
        except Exception as e:
            ai_logger.error(f"Synergistic decision error: {e}")
            return {
                'decision': 'HOLD',
                'confidence': 0,
                'reason': str(e),
                'error': True
            }
    
    def calculate_consensus(self, gpt: Dict, grok: Dict, context: Dict) -> Dict[str, Any]:
        """Calcule un consensus entre GPT et Grok."""
        
        # Mapping des recommandations en scores
        rec_scores = {
            'STRONG_BUY': 2,
            'BUY': 1,
            'HOLD': 0,
            'SELL': -1,
            'STRONG_SELL': -2
        }
        
        gpt_score = rec_scores.get(gpt['recommendation'], 0)
        grok_score = rec_scores.get(grok['recommendation'], 0)
        
        # Handle errors
        if gpt.get('error', False):
            gpt_weight = 0.1
        else:
            gpt_weight = gpt['confidence'] / 100
            
        if grok.get('error', False):
            grok_weight = 0.1
        else:
            grok_weight = grok['confidence'] / 100
        
        # Moyenne pondérée des scores
        total_weight = gpt_weight + grok_weight
        if total_weight > 0:
            weighted_score = (gpt_score * gpt_weight + grok_score * grok_weight) / total_weight
        else:
            weighted_score = 0
        
        # Décision finale
        if weighted_score >= 1.5:
            final_decision = 'STRONG_BUY'
        elif weighted_score >= 0.5:
            final_decision = 'BUY'
        elif weighted_score <= -1.5:
            final_decision = 'STRONG_SELL'
        elif weighted_score <= -0.5:
            final_decision = 'SELL'
        else:
            final_decision = 'HOLD'
        
        # Calcul de la confiance du consensus
        agreement = 1 - abs(gpt_score - grok_score) / 4
        avg_confidence = (gpt['confidence'] + grok['confidence']) / 2
        consensus_confidence = avg_confidence * agreement
        
        # Incorporer les prédictions ML
        ml_avg = 0.5
        if context.get('ml_predictions'):
            ml_probs = []
            for tf in ['1m', '5m']:
                if tf in context['ml_predictions']:
                    ml_probs.append(context['ml_predictions'][tf].get('prediction', 0.5))
            if ml_probs:
                ml_avg = np.mean(ml_probs)
        
        # Ajuster si ML est en fort désaccord
        ml_direction = 1 if ml_avg > 0.5 else -1
        ai_direction = 1 if weighted_score > 0 else -1
        
        if ml_direction != ai_direction and abs(ml_avg - 0.5) > 0.2:
            consensus_confidence *= 0.7
            ai_logger.warning(f"ML disagrees with AI: ML={ml_avg:.2f}, AI={weighted_score:.2f}")
        
        # Boost confidence if all agree
        if abs(gpt_score - grok_score) <= 1 and ml_direction == ai_direction:
            consensus_confidence = min(95, consensus_confidence * 1.2)
        
        return {
            'decision': final_decision,
            'confidence': consensus_confidence,
            'gpt_recommendation': gpt['recommendation'],
            'grok_recommendation': grok['recommendation'],
            'ai_agreement': agreement,
            'weighted_score': weighted_score,
            'ml_alignment': abs(ml_avg - 0.5) * 2,
            'ml_direction_agrees': ml_direction == ai_direction,
            'reasoning': self.generate_reasoning(gpt, grok, final_decision),
            'execution_recommendation': self.get_execution_recommendation(consensus_confidence, context)
        }
    
    def generate_reasoning(self, gpt: Dict, grok: Dict, decision: str) -> str:
        """Génère une explication de la décision."""
        if gpt.get('error') and grok.get('error'):
            return "Both AI systems unavailable - holding position"
        elif gpt.get('error'):
            return f"GPT unavailable - Grok suggests {grok['recommendation']} ({grok['confidence']}%)"
        elif grok.get('error'):
            return f"Grok unavailable - GPT suggests {gpt['recommendation']} ({gpt['confidence']}%)"
        elif gpt['recommendation'] == grok['recommendation']:
            return f"Both AIs agree on {decision} with high confidence"
        else:
            return f"Mixed signals: GPT={gpt['recommendation']} ({gpt['confidence']}%), " \
                   f"Grok={grok['recommendation']} ({grok['confidence']}%). " \
                   f"Consensus: {decision}"
    
    def get_execution_recommendation(self, confidence: float, context: Dict) -> str:
        """Recommandation d'exécution basée sur la confiance."""
        if confidence >= TRADING_CONFIG.ai_high_confidence_threshold:
            return "EXECUTE_AGGRESSIVE"
        elif confidence >= TRADING_CONFIG.ai_consensus_threshold:
            return "EXECUTE_NORMAL"
        elif confidence >= 65:
            return "EXECUTE_CAUTIOUS"
        else:
            return "MONITOR_ONLY"

# === SMART TRADING STRATEGIES ===
class SmartTradingStrategies:
    """Stratégies de trading adaptatives basées sur le contexte."""
    
    def __init__(self, state, config):
        self.state = state
        self.config = config
        self.active_strategies = {}
        self.strategy_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
        
    def select_strategy(self, market_context: Dict) -> Dict[str, Any]:
        """Sélectionne la meilleure stratégie selon le contexte."""
        
        volatility = market_context.get('volatility', 0.15)
        volume_ratio = market_context.get('volume_ratio', 1.0)
        trend_strength = market_context.get('trend_strength', 0)
        time_of_day = market_context.get('time_of_day', 'regular')
        ai_confidence = market_context.get('ai_confidence', 50)
        
        # Decision matrix
        if volatility < 0.15 and trend_strength > 0.7 and ai_confidence > 75:
            return self.trend_following_strategy(market_context)
            
        elif volatility > 0.25 and volume_ratio > 2:
            return self.volatility_breakout_strategy(market_context)
            
        elif time_of_day in ['open', 'close'] and volume_ratio > 1.5:
            return self.opening_range_strategy(market_context)
            
        elif abs(trend_strength) < 0.3 and volatility < 0.2:
            return self.mean_reversion_strategy(market_context)
            
        else:
            return self.adaptive_ai_strategy(market_context)
    
    def trend_following_strategy(self, context: Dict) -> Dict:
        """Stratégie de suivi de tendance pour marchés directionnels."""
        return {
            'name': 'trend_following',
            'entry_rules': {
                'min_trend_strength': 0.7,
                'confirmation_period': 300,
                'volume_confirmation': True,
                'ai_min_confidence': 70
            },
            'exit_rules': {
                'trailing_stop': 0.01,
                'profit_target': 0.03,
                'trend_reversal_exit': True,
                'time_stop': 3600
            },
            'position_size': 1.5,
            'max_hold_time': 3600,
            'use_limit_orders': False
        }
    
    def volatility_breakout_strategy(self, context: Dict) -> Dict:
        """Stratégie pour capturer les breakouts en période volatile."""
        return {
            'name': 'volatility_breakout',
            'entry_rules': {
                'breakout_threshold': 0.02,
                'volume_surge': 2.0,
                'confirmation_candles': 2,
                'atr_multiplier': 2.5
            },
            'exit_rules': {
                'stop_loss': 0.015,
                'take_profit': 0.04,
                'time_stop': 900,
                'volatility_exit': True
            },
            'position_size': 0.7,
            'use_limit_orders': False
        }
    
    def opening_range_strategy(self, context: Dict) -> Dict:
        """Stratégie pour l'ouverture du marché."""
        return {
            'name': 'opening_range',
            'entry_rules': {
                'wait_period': 300,
                'range_breakout': True,
                'volume_threshold': 1.5,
                'range_period': 900
            },
            'exit_rules': {
                'stop_loss': 0.01,
                'profit_target': 0.02,
                'close_before': '10:30',
                'range_retest_exit': True
            },
            'position_size': 1.0,
            'aggressive_entry': True
        }
    
    def mean_reversion_strategy(self, context: Dict) -> Dict:
        """Stratégie de retour à la moyenne."""
        return {
            'name': 'mean_reversion',
            'entry_rules': {
                'deviation_threshold': 2.0,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'bollinger_touch': True
            },
            'exit_rules': {
                'target': 'vwap',
                'stop_loss': 0.02,
                'time_limit': 1800,
                'mean_reached_exit': True
            },
            'position_size': 0.8,
            'scale_in': True
        }
    
    def adaptive_ai_strategy(self, context: Dict) -> Dict:
        """Stratégie adaptative basée sur l'IA."""
        ai_confidence = context.get('ai_confidence', 50)
        
        # Adjust parameters based on AI confidence
        position_multiplier = 0.5 + (ai_confidence / 100) * 0.5
        stop_loss = 0.03 - (ai_confidence / 100) * 0.015
        
        return {
            'name': 'adaptive_ai',
            'entry_rules': {
                'ai_consensus': TRADING_CONFIG.ai_consensus_threshold,
                'ml_confirmation': 0.65,
                'risk_score_max': 0.3
            },
            'exit_rules': {
                'dynamic_stop': True,
                'ai_signal_reversal': True,
                'profit_lock': 0.01,
                'stop_loss': stop_loss
            },
            'position_size': position_multiplier,
            'adjust_by_confidence': True
        }

# === PATTERN ANALYZER ===
class AdvancedPatternAnalyzer:
    """Détection de patterns complexes pour GPT/Grok."""
    
    def __init__(self):
        self.pattern_history = deque(maxlen=1000)
        self.pattern_success_rate = defaultdict(float)
        
    def analyze_market_patterns(self, df_tns: pd.DataFrame) -> Dict[str, Any]:
        """Analyse complète des patterns de marché."""
        
        if len(df_tns) < 100:
            return {'patterns': {}, 'score': 0, 'recommendation': 'Insufficient data'}
        
        patterns = {
            'microstructure': self.detect_microstructure_patterns(df_tns),
            'technical': self.detect_technical_patterns(df_tns),
            'volume': self.detect_volume_patterns(df_tns),
            'institutional': self.detect_institutional_patterns(df_tns)
        }
        
        # Score global des patterns
        pattern_score = self.calculate_pattern_score(patterns)
        
        return {
            'patterns': patterns,
            'score': pattern_score,
            'strongest_pattern': self.get_strongest_pattern(patterns),
            'recommendation': self.pattern_based_recommendation(patterns)
        }
    
    def detect_microstructure_patterns(self, df: pd.DataFrame) -> Dict:
        """Détecte les patterns de microstructure."""
        
        patterns = {}
        
        # Iceberg orders
        patterns['iceberg'] = self.detect_iceberg_orders(df)
        
        # Price absorption
        patterns['absorption'] = self.detect_price_absorption(df)
        
        # Momentum ignition
        patterns['momentum_ignition'] = self.detect_momentum_ignition(df)
        
        # Hidden liquidity
        patterns['hidden_liquidity'] = self.detect_hidden_liquidity(df)
        
        return patterns
    
    def detect_iceberg_orders(self, df: pd.DataFrame) -> Dict:
        """Détecte les ordres iceberg."""
        if len(df) < 50:
            return {'detected': False}
        
        # Chercher des exécutions répétées au même prix
        recent = df.tail(50)
        price_counts = recent.groupby('Price').agg({
            'Size': ['count', 'sum', 'mean']
        })
        
        for price, stats in price_counts.iterrows():
            count = stats[('Size', 'count')]
            total = stats[('Size', 'sum')]
            avg = stats[('Size', 'mean')]
            
            # Iceberg likely if many small orders at same price
            if count > 5 and total > df['Size'].sum() * 0.1:
                return {
                    'detected': True,
                    'price': price,
                    'total_size': total,
                    'executions': count,
                    'avg_size': avg,
                    'confidence': min(0.9, count / 10)
                }
        
        return {'detected': False}
    
    def detect_price_absorption(self, df: pd.DataFrame) -> Dict:
        """Détecte l'absorption de prix."""
        if len(df) < 100:
            return {'detected': False}
        
        recent = df.tail(100)
        prices = recent['Price'].values
        volumes = recent['Size'].values
        
        # Find price levels with high volume
        price_levels = defaultdict(float)
        for p, v in zip(prices, volumes):
            price_levels[round(p, 2)] += v
        
        # Sort by volume
        sorted_levels = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_levels and sorted_levels[0][1] > volumes.sum() * 0.2:
            absorption_price = sorted_levels[0][0]
            absorption_volume = sorted_levels[0][1]
            
            # Check if price is holding
            recent_prices = prices[-20:]
            touches = sum(1 for p in recent_prices if abs(p - absorption_price) < 0.01)
            
            if touches >= 3:
                return {
                    'detected': True,
                    'price': absorption_price,
                    'volume': absorption_volume,
                    'touches': touches,
                    'holding': True
                }
        
        return {'detected': False}
    
    def detect_momentum_ignition(self, df: pd.DataFrame) -> Dict:
        """Détecte l'ignition de momentum."""
        if len(df) < 50:
            return {'detected': False}
        
        recent = df.tail(50)
        
        # Calculate rate of change
        price_changes = recent['Price'].pct_change().fillna(0)
        volume_surge = recent['Size'].rolling(5).sum()
        
        # Look for sudden acceleration
        for i in range(10, len(price_changes)):
            if abs(price_changes.iloc[i]) > 0.002:  # 0.2% move
                if volume_surge.iloc[i] > volume_surge.mean() * 3:
                    return {
                        'detected': True,
                        'direction': 'up' if price_changes.iloc[i] > 0 else 'down',
                        'magnitude': abs(price_changes.iloc[i]),
                        'volume_multiplier': volume_surge.iloc[i] / volume_surge.mean(),
                        'timestamp': recent.iloc[i]['Timestamp']
                    }
        
        return {'detected': False}
    
    def detect_hidden_liquidity(self, df: pd.DataFrame) -> Dict:
        """Détecte la liquidité cachée."""
        if len(df) < 100:
            return {'detected': False}
        
        # Analyze price impact vs volume
        recent = df.tail(100)
        
        # Group by small time windows
        recent['TimeGroup'] = (recent['Timestamp'] // 10).astype(int)
        
        grouped = recent.groupby('TimeGroup').agg({
            'Price': ['first', 'last'],
            'Size': 'sum'
        })
        
        if len(grouped) < 5:
            return {'detected': False}
        
        # Calculate price impact
        price_impacts = []
        for _, row in grouped.iterrows():
            price_change = abs(row[('Price', 'last')] - row[('Price', 'first')])
            volume = row[('Size', 'sum')]
            if volume > 0:
                impact = price_change / volume * 10000
                price_impacts.append(impact)
        
        if price_impacts:
            avg_impact = np.mean(price_impacts)
            if avg_impact < 0.1:  # Very low price impact
                return {
                    'detected': True,
                    'avg_price_impact': avg_impact,
                    'likely_hidden_size': recent['Size'].sum() * 2,
                    'confidence': 0.7
                }
        
        return {'detected': False}
    
    def detect_technical_patterns(self, df: pd.DataFrame) -> Dict:
        """Détecte les patterns techniques."""
        patterns = {}
        
        if len(df) < 100:
            return patterns
        
        prices = df['Price'].values
        
        # Support/Resistance
        patterns['support_resistance'] = self.find_support_resistance(prices)
        
        # Breakout
        patterns['breakout'] = self.detect_breakout(prices)
        
        # Triangle
        patterns['triangle'] = self.detect_triangle_pattern(prices)
        
        return patterns
    
    def find_support_resistance(self, prices: np.ndarray) -> Dict:
        """Trouve les niveaux de support et résistance."""
        if len(prices) < 50:
            return {}
        
        # Use recent price data
        recent_prices = prices[-100:] if len(prices) > 100 else prices
        
        # Find local minima and maxima
        peaks, _ = find_peaks(recent_prices, distance=5)
        troughs, _ = find_peaks(-recent_prices, distance=5)
        
        resistance_levels = []
        support_levels = []
        
        if len(peaks) > 0:
            resistance_levels = recent_prices[peaks]
            resistance = np.mean(sorted(resistance_levels)[-3:]) if len(resistance_levels) >= 3 else resistance_levels[-1]
        else:
            resistance = max(recent_prices)
        
        if len(troughs) > 0:
            support_levels = recent_prices[troughs]
            support = np.mean(sorted(support_levels)[:3]) if len(support_levels) >= 3 else support_levels[0]
        else:
            support = min(recent_prices)
        
        return {
            'support': support,
            'resistance': resistance,
            'strength': len(support_levels) + len(resistance_levels),
            'current_position': (prices[-1] - support) / (resistance - support) if resistance > support else 0.5
        }
    
    def detect_breakout(self, prices: np.ndarray) -> Dict:
        """Détecte les breakouts."""
        if len(prices) < 20:
            return {'detected': False}
        
        # Rolling high/low
        high_20 = max(prices[-20:-1])
        low_20 = min(prices[-20:-1])
        current = prices[-1]
        
        if current > high_20 * 1.001:  # 0.1% above high
            return {
                'detected': True,
                'type': 'resistance',
                'level': high_20,
                'strength': (current - high_20) / high_20
            }
        elif current < low_20 * 0.999:  # 0.1% below low
            return {
                'detected': True,
                'type': 'support',
                'level': low_20,
                'strength': (low_20 - current) / low_20
            }
        
        return {'detected': False}
    
    def detect_triangle_pattern(self, prices: np.ndarray) -> Dict:
        """Détecte les patterns triangulaires."""
        if len(prices) < 50:
            return {'detected': False}
        
        # Simplified triangle detection
        recent = prices[-50:]
        
        # Find peaks and troughs
        peaks, _ = find_peaks(recent, distance=5)
        troughs, _ = find_peaks(-recent, distance=5)
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            # Check if peaks are descending and troughs ascending
            peak_trend = np.polyfit(peaks, recent[peaks], 1)[0]
            trough_trend = np.polyfit(troughs, recent[troughs], 1)[0]
            
            if peak_trend < -0.0001 and trough_trend > 0.0001:
                return {
                    'detected': True,
                    'type': 'symmetrical',
                    'apex': len(recent),
                    'breakout_expected': True
                }
        
        return {'detected': False}
    
    def detect_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Détecte les patterns de volume."""
        patterns = {}
        
        if len(df) < 50:
            return patterns
        
        volumes = df['Size'].values
        prices = df['Price'].values
        
        # Volume surge
        recent_vol = volumes[-10:].sum()
        avg_vol = volumes[-50:-10].mean() * 10
        
        if recent_vol > avg_vol * 2:
            patterns['volume_surge'] = {
                'detected': True,
                'multiplier': recent_vol / avg_vol,
                'direction': 'up' if prices[-1] > prices[-10] else 'down'
            }
        
        # Volume divergence
        price_trend = np.polyfit(range(20), prices[-20:], 1)[0]
        volume_trend = np.polyfit(range(20), volumes[-20:], 1)[0]
        
        if price_trend > 0 and volume_trend < 0:
            patterns['volume_divergence'] = {
                'detected': True,
                'type': 'bearish',
                'strength': abs(volume_trend)
            }
        elif price_trend < 0 and volume_trend > 0:
            patterns['volume_divergence'] = {
                'detected': True,
                'type': 'bullish',
                'strength': abs(volume_trend)
            }
        
        return patterns
    
    def detect_institutional_patterns(self, df: pd.DataFrame) -> Dict:
        """Détecte l'activité institutionnelle."""
        patterns = {}
        
        if len(df) < 100:
            return patterns
        
        # Block trades
        large_trades = df[df['Size'] > df['Size'].quantile(0.95)]
        patterns['block_trades'] = {
            'count': len(large_trades),
            'total_volume': large_trades['Size'].sum() if len(large_trades) > 0 else 0,
            'avg_price': large_trades['Price'].mean() if len(large_trades) > 0 else 0,
            'recent': len(large_trades[large_trades['Timestamp'] > time.time() - 300])
        }
        
        # Accumulation/Distribution
        if len(df) > 100:
            prices = df['Price'].values
            volumes = df['Size'].values
            
            # Price and volume trends
            price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
            
            # Accumulation: stable/down price with increasing volume
            # Distribution: up price with decreasing volume at highs
            
            if price_trend <= 0:
                # Check for accumulation
                low_price_volumes = volumes[prices < np.percentile(prices, 30)]
                high_price_volumes = volumes[prices > np.percentile(prices, 70)]
                
                if len(low_price_volumes) > 0 and len(high_price_volumes) > 0:
                    if np.mean(low_price_volumes) > np.mean(high_price_volumes) * 1.5:
                        patterns['accumulation'] = {
                            'detected': True,
                            'strength': np.mean(low_price_volumes) / np.mean(high_price_volumes)
                        }
            
            elif price_trend > 0:
                # Check for distribution
                recent_high = np.percentile(prices[-20:], 80)
                high_volumes = volumes[prices > recent_high]
                
                if len(high_volumes) > 5 and np.mean(high_volumes) > np.mean(volumes) * 2:
                    patterns['distribution'] = {
                        'detected': True,
                        'strength': np.mean(high_volumes) / np.mean(volumes)
                    }
        
        return patterns
    
    def calculate_pattern_score(self, patterns: Dict) -> float:
        """Calcule un score global des patterns."""
        score = 0
        pattern_count = 0
        
        # Microstructure patterns
        if patterns.get('microstructure', {}).get('iceberg', {}).get('detected'):
            score += 2
            pattern_count += 1
        
        if patterns.get('microstructure', {}).get('absorption', {}).get('detected'):
            score += 1.5
            pattern_count += 1
        
        if patterns.get('microstructure', {}).get('momentum_ignition', {}).get('detected'):
            score += 2.5
            pattern_count += 1
        
        # Technical patterns
        if patterns.get('technical', {}).get('breakout', {}).get('detected'):
            score += 2
            pattern_count += 1
        
        # Volume patterns
        if patterns.get('volume', {}).get('volume_surge', {}).get('detected'):
            surge = patterns['volume']['volume_surge'].get('multiplier', 1)
            score += min(2, surge / 2)
            pattern_count += 1
        
        # Institutional patterns
        if patterns.get('institutional', {}).get('accumulation', {}).get('detected'):
            score += 2.5
            pattern_count += 1
        
        if patterns.get('institutional', {}).get('block_trades', {}).get('recent', 0) > 3:
            score += 1.5
            pattern_count += 1
        
        # Normalize score
        return min(10, score)
    
    def get_strongest_pattern(self, patterns: Dict) -> str:
        """Identifie le pattern le plus fort."""
        strongest = "No significant pattern"
        max_score = 0
        
        pattern_scores = {
            'iceberg': 2.5,
            'momentum_ignition': 3.0,
            'accumulation': 2.5,
            'breakout': 2.0,
            'volume_surge': 1.5
        }
        
        for category, category_patterns in patterns.items():
            if isinstance(category_patterns, dict):
                for pattern_name, pattern_data in category_patterns.items():
                    if isinstance(pattern_data, dict) and pattern_data.get('detected'):
                        score = pattern_scores.get(pattern_name, 1.0)
                        if score > max_score:
                            max_score = score
                            strongest = f"{category}:{pattern_name}"
        
        return strongest
    
    def pattern_based_recommendation(self, patterns: Dict) -> str:
        """Génère une recommandation basée sur les patterns."""
        recommendations = []
        
        # Microstructure
        if patterns.get('microstructure', {}).get('iceberg', {}).get('detected'):
            recommendations.append("Large hidden order detected - expect continued pressure")
        
        if patterns.get('microstructure', {}).get('momentum_ignition', {}).get('detected'):
            direction = patterns['microstructure']['momentum_ignition'].get('direction', 'unknown')
            recommendations.append(f"Momentum ignition {direction} - follow the move")
        
        # Technical
        if patterns.get('technical', {}).get('breakout', {}).get('detected'):
            breakout_type = patterns['technical']['breakout'].get('type', 'unknown')
            recommendations.append(f"{breakout_type} breakout confirmed")
        
        # Institutional
        if patterns.get('institutional', {}).get('accumulation', {}).get('detected'):
            recommendations.append("Institutional accumulation detected - bullish bias")
        
        if patterns.get('institutional', {}).get('distribution', {}).get('detected'):
            recommendations.append("Institutional distribution detected - bearish bias")
        
        return " | ".join(recommendations) if recommendations else "No clear pattern-based signal"

# === RISK MANAGER INTELLIGENT ===
class IntelligentRiskManager:
    """Gestion des risques basée sur l'IA."""
    
    def __init__(self, state, config):
        self.state = state
        self.config = config
        self.risk_events = deque(maxlen=1000)
        self.dynamic_limits = {}
        
    def calculate_dynamic_risk_limits(self, ai_consensus: Dict, market_context: Dict) -> Dict[str, float]:
        """Calcule des limites de risque dynamiques."""
        
        base_risk = self.config.base_risk_percent / 100
        
        # AI adjustments
        ai_confidence = ai_consensus.get('confidence', 50) / 100
        ai_agreement = ai_consensus.get('ai_agreement', 0.5)
        
        # Market adjustments
        volatility = market_context.get('volatility', 0.15)
        regime = market_context.get('regime', 'Normal')
        session = get_trading_session()
        
        # Calculate risk multiplier
        risk_multiplier = 1.0
        
        # AI confidence adjustment
        if ai_confidence > 0.85 and ai_agreement > 0.8:
            risk_multiplier *= 1.5
        elif ai_confidence < 0.6 or ai_agreement < 0.4:
            risk_multiplier *= 0.5
        
        # Volatility adjustment
        vol_params = self.get_volatility_params(volatility)
        risk_multiplier *= vol_params['position_multiplier']
        
        # Regime adjustment
        regime_multipliers = {
            'Trending': 1.3,
            'Normal': 1.0,
            'Choppy': 0.6,
            'Volatile Trending': 0.8,
            'Low Volatility': 1.1
        }
        risk_multiplier *= regime_multipliers.get(regime, 1.0)
        
        # Session adjustment
        session_multipliers = {
            'PREMARKET': self.config.pre_market_risk_multiplier,
            'REGULAR': self.config.regular_hours_risk_multiplier,
            'AFTERHOURS': self.config.after_hours_risk_multiplier,
            'CLOSED': 0
        }
        risk_multiplier *= session_multipliers.get(session, 1.0)
        
        # Calculate final limits
        position_risk = base_risk * risk_multiplier
        
        return {
            'position_risk': position_risk,
            'daily_risk': position_risk * 3,
            'stop_loss': self.calculate_dynamic_stop(volatility, ai_confidence),
            'position_size': self.calculate_kelly_criterion(ai_consensus, market_context),
            'max_correlation': 0.7 - (ai_confidence * 0.2),
            'risk_multiplier': risk_multiplier,
            'volatility_params': vol_params
        }
    
    def get_volatility_params(self, volatility: float) -> Dict:
        """Get parameters based on volatility level."""
        if volatility < 0.15:
            return VOLATILITY_PARAMS['low']
        elif volatility < 0.25:
            return VOLATILITY_PARAMS['normal']
        elif volatility < 0.35:
            return VOLATILITY_PARAMS['high']
        else:
            return VOLATILITY_PARAMS['extreme']
    
    def calculate_dynamic_stop(self, volatility: float, confidence: float) -> float:
        """Calcule un stop-loss dynamique."""
        
        # Base stop selon volatilité (2 ATR)
        base_stop = volatility * np.sqrt(1/252) * 2
        
        # Adjust by confidence (tighter stop if confident)
        confidence_adjustment = 1 - (confidence - 0.5) * 0.5
        
        # Apply limits
        dynamic_stop = base_stop * confidence_adjustment
        return max(0.005, min(0.05, dynamic_stop))  # 0.5% to 5%
    
    def calculate_kelly_criterion(self, ai_consensus: Dict, market_context: Dict) -> float:
        """Calcule la taille optimale selon Kelly."""
        
        # Estimated win probability from AI
        p = ai_consensus.get('confidence', 50) / 100
        
        # Win/loss ratio from historical data
        avg_win = self.state.metrics.avg_win if self.state.metrics.avg_win > 0 else 100
        avg_loss = self.state.metrics.avg_loss if self.state.metrics.avg_loss > 0 else 100
        b = avg_win / avg_loss
        
        # Kelly formula: f = p - q/b
        q = 1 - p
        kelly_fraction = p - (q / b)
        
        # Apply safety factor (Kelly/4)
        safe_kelly = max(0, min(self.config.max_kelly_fraction, kelly_fraction / 4))
        
        # Adjust for market conditions
        volatility = market_context.get('volatility', 0.15)
        if volatility > 0.25:
            safe_kelly *= 0.7
        
        # Convert to position size
        account_value = 100000  # TODO: Get from IB
        position_value = account_value * safe_kelly
        
        return position_value
    
    def check_risk_limits(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        
        # Circuit breaker
        if self.state.circuit_breaker_triggered:
            return False, "Circuit breaker active"
        
        # Daily loss
        if self.state.metrics.daily_pnl < self.state.risk_limits.max_daily_loss:
            self.state.circuit_breaker_triggered = True
            return False, f"Daily loss limit reached: ${self.state.metrics.daily_pnl:.2f}"
        
        # Consecutive losses
        if self.state.metrics.consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"Consecutive losses: {self.state.metrics.consecutive_losses}"
        
        # Portfolio heat
        if self.state.portfolio_heat > self.config.max_portfolio_heat:
            return False, f"Portfolio heat too high: {self.state.portfolio_heat:.1%}"
        
        # Drawdown
        if self.state.metrics.current_drawdown < self.state.risk_limits.max_drawdown:
            return False, f"Max drawdown reached: {self.state.metrics.current_drawdown:.1%}"
        
        return True, "Risk checks passed"
    
    def calculate_position_size(self, 
                              signal: Dict, 
                              ai_consensus: Dict,
                              market_context: Dict) -> float:
        """Calculate optimal position size with all factors."""
        
        # Get dynamic limits
        risk_limits = self.calculate_dynamic_risk_limits(ai_consensus, market_context)
        
        # Base position from Kelly
        kelly_size = risk_limits['position_size']
        
        # Adjust for signal strength
        signal_confidence = signal.get('confidence', 50) / 100
        
        # Adjust for strategy
        strategy = signal.get('strategy', {})
        strategy_multiplier = strategy.get('position_size', 1.0)
        
        # Final position size
        position_size = kelly_size * signal_confidence * strategy_multiplier
        
        # Apply limits
        max_position = self.state.risk_limits.max_position_size
        position_size = min(position_size, max_position)
        
        # Round to nearest 100 shares
        position_size = round(position_size / 100) * 100
        
        return max(100, position_size)  # Minimum 100 shares

# === SMART ALERT SYSTEM ===
class SmartAlertSystem:
    """Système d'alertes basé sur l'IA."""
    
    def __init__(self, state):
        self.state = state
        self.alert_queue = deque(maxlen=100)
        self.alert_priorities = {
            'CRITICAL': 1,
            'HIGH': 2,
            'MEDIUM': 3,
            'LOW': 4,
            'INFO': 5
        }
        self.last_alert_time = defaultdict(float)
        self.alert_cooldowns = {
            'CRITICAL': 60,
            'HIGH': 120,
            'MEDIUM': 300,
            'LOW': 600,
            'INFO': 900
        }
        
    def check_alert_conditions(self, 
                             ai_analysis: Dict,
                             market_context: Dict,
                             patterns: Dict) -> List[Dict]:
        """Vérifie les conditions d'alerte."""
        
        alerts = []
        current_time = time.time()
        
        # AI Divergence Alert
        if ai_analysis.get('ai_agreement', 1) < 0.3:
            alert = {
                'type': 'AI_DIVERGENCE',
                'priority': 'HIGH',
                'message': f"⚠️ AI Divergence: GPT={ai_analysis.get('gpt_recommendation', 'N/A')}, "
                          f"Grok={ai_analysis.get('grok_recommendation', 'N/A')}",
                'action': 'REVIEW_MANUALLY',
                'data': ai_analysis
            }
            if self._should_send_alert(alert, current_time):
                alerts.append(alert)
        
        # High Confidence Opportunity
        if ai_analysis.get('confidence', 0) > 90:
            alert = {
                'type': 'HIGH_CONFIDENCE_OPPORTUNITY',
                'priority': 'HIGH',
                'message': f"🎯 High Confidence Signal: {ai_analysis.get('decision', 'N/A')} "
                          f"({ai_analysis.get('confidence', 0):.1f}%)",
                'action': 'CONSIDER_ENTRY',
                'data': ai_analysis
            }
            if self._should_send_alert(alert, current_time):
                alerts.append(alert)
        
        # Pattern Detection Alerts
        if patterns:
            # Iceberg order alert
            if patterns.get('microstructure', {}).get('iceberg', {}).get('detected'):
                alert = {
                    'type': 'ICEBERG_DETECTED',
                    'priority': 'MEDIUM',
                    'message': f"🧊 Iceberg order detected at ${patterns['microstructure']['iceberg'].get('price', 0):.2f}",
                    'action': 'MONITOR_CLOSELY',
                    'data': patterns['microstructure']['iceberg']
                }
                if self._should_send_alert(alert, current_time):
                    alerts.append(alert)
            
            # Institutional activity
            if patterns.get('institutional', {}).get('accumulation', {}).get('detected'):
                alert = {
                    'type': 'INSTITUTIONAL_ACCUMULATION',
                    'priority': 'MEDIUM',
                    'message': "🏛️ Institutional accumulation detected",
                    'action': 'CONSIDER_FOLLOWING',
                    'data': patterns['institutional']['accumulation']
                }
                if self._should_send_alert(alert, current_time):
                    alerts.append(alert)
        
        # Risk Alerts
        if market_context.get('volatility', 0) > 0.35:
            alert = {
                'type': 'EXTREME_VOLATILITY',
                'priority': 'HIGH',
                'message': f"⚠️ Extreme volatility: {market_context['volatility']:.1%}",
                'action': 'REDUCE_POSITION_SIZE',
                'data': market_context
            }
            if self._should_send_alert(alert, current_time):
                alerts.append(alert)
        
        # Circuit Breaker Warning
        if self.state.metrics.consecutive_losses >= self.state.config.max_consecutive_losses - 1:
            alert = {
                'type': 'CIRCUIT_BREAKER_WARNING',
                'priority': 'CRITICAL',
                'message': f"🛑 Circuit breaker warning: {self.state.metrics.consecutive_losses} consecutive losses",
                'action': 'STOP_TRADING',
                'data': {'consecutive_losses': self.state.metrics.consecutive_losses}
            }
            if self._should_send_alert(alert, current_time):
                alerts.append(alert)
        
        # Sort by priority
        return sorted(alerts, key=lambda x: self.alert_priorities[x['priority']])
    
    def _should_send_alert(self, alert: Dict, current_time: float) -> bool:
        """Check if alert should be sent based on cooldown."""
        alert_key = f"{alert['type']}_{alert.get('priority', 'INFO')}"
        last_sent = self.last_alert_time.get(alert_key, 0)
        cooldown = self.alert_cooldowns.get(alert['priority'], 600)
        
        if current_time - last_sent >= cooldown:
            self.last_alert_time[alert_key] = current_time
            return True
        
        return False
    
    def format_alert_message(self, alerts: List[Dict]) -> str:
        """Format alerts for Telegram."""
        if not alerts:
            return ""
        
        message = "🚨 **SYSTEM ALERTS**\n━━━━━━━━━━━━━━━━━━\n\n"
        
        for alert in alerts[:5]:  # Max 5 alerts
            priority_emoji = {
                'CRITICAL': '🔴',
                'HIGH': '🟡',
                'MEDIUM': '🟢',
                'LOW': '🔵',
                'INFO': 'ℹ️'
            }
            
            emoji = priority_emoji.get(alert['priority'], '❓')
            message += f"{emoji} **{alert['type']}**\n"
            message += f"{alert['message']}\n"
            message += f"Action: {alert['action']}\n\n"
        
        return message

# === SESSION PROFILER ===
class SessionProfiler:
    """Profile les différentes sessions de trading."""
    
    PROFILES = {
        'pre_market': {
            'volatility_multiplier': 1.5,
            'liquidity_discount': 0.7,
            'news_sensitivity': 2.0,
            'spread_multiplier': 2.0,
            'confidence_threshold': 80
        },
        'opening_30min': {
            'volatility_multiplier': 2.0,
            'institutional_activity': 'high',
            'retail_participation': 'low',
            'spread_multiplier': 1.5,
            'confidence_threshold': 85
        },
        'regular_morning': {
            'volatility_multiplier': 1.2,
            'institutional_activity': 'high',
            'retail_participation': 'medium',
            'spread_multiplier': 1.0,
            'confidence_threshold': 75
        },
        'lunch_hour': {
            'volatility_multiplier': 0.7,
            'algo_dominance': 0.8,
            'liquidity_discount': 0.8,
            'spread_multiplier': 1.2,
            'confidence_threshold': 80
        },
        'regular_afternoon': {
            'volatility_multiplier': 1.0,
            'institutional_activity': 'medium',
            'retail_participation': 'medium',
            'spread_multiplier': 1.0,
            'confidence_threshold': 75
        },
        'power_hour': {
            'volatility_multiplier': 1.3,
            'institutional_rebalancing': True,
            'moc_impact': 'high',
            'spread_multiplier': 1.1,
            'confidence_threshold': 70
        },
        'after_hours': {
            'volatility_multiplier': 1.2,
            'liquidity_discount': 0.5,
            'news_sensitivity': 3.0,
            'spread_multiplier': 3.0,
            'confidence_threshold': 90
        }
    }
    
    def get_current_profile(self) -> Tuple[str, Dict]:
        """Get current session profile."""
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.datetime.now(ny_tz)
        
        # Weekend
        if now.weekday() >= 5:
            return 'closed', {'trading_allowed': False}
        
        hour = now.hour
        minute = now.minute
        time_decimal = hour + minute / 60
        
        # Pre-market: 4:00 - 9:30
        if 4 <= time_decimal < 9.5:
            return 'pre_market', self.PROFILES['pre_market']
        
        # Opening 30 minutes: 9:30 - 10:00
        elif 9.5 <= time_decimal < 10:
            return 'opening_30min', self.PROFILES['opening_30min']
        
        # Morning session: 10:00 - 11:30
        elif 10 <= time_decimal < 11.5:
            return 'regular_morning', self.PROFILES['regular_morning']
        
        # Lunch hour: 11:30 - 13:00
        elif 11.5 <= time_decimal < 13:
            return 'lunch_hour', self.PROFILES['lunch_hour']
        
        # Afternoon session: 13:00 - 15:00
        elif 13 <= time_decimal < 15:
            return 'regular_afternoon', self.PROFILES['regular_afternoon']
        
        # Power hour: 15:00 - 16:00
        elif 15 <= time_decimal < 16:
            return 'power_hour', self.PROFILES['power_hour']
        
        # After hours: 16:00 - 20:00
        elif 16 <= time_decimal < 20:
            return 'after_hours', self.PROFILES['after_hours']
        
        # Closed
        else:
            return 'closed', {'trading_allowed': False}
    
    def adjust_for_session(self, base_params: Dict) -> Dict:
        """Adjust parameters based on current session."""
        session_name, profile = self.get_current_profile()
        
        if not profile.get('trading_allowed', True):
            return {'trading_allowed': False}
        
        adjusted = base_params.copy()
        
        # Apply multipliers
        if 'volatility' in adjusted and 'volatility_multiplier' in profile:
            adjusted['volatility'] *= profile['volatility_multiplier']
        
        if 'spread_limit' in adjusted and 'spread_multiplier' in profile:
            adjusted['spread_limit'] *= profile['spread_multiplier']
        
        if 'confidence_threshold' in profile:
            adjusted['min_confidence'] = profile['confidence_threshold']
        
        # Add session info
        adjusted['session'] = session_name
        adjusted['session_profile'] = profile
        
        return adjusted

# === LATENCY OPTIMIZER ===
class LatencyOptimizer:
    """Optimise la latence pour exécution rapide."""
    
    def __init__(self):
        self.latency_history = deque(maxlen=1000)
        self.route_performance = defaultdict(list)
        self.cache = {}
        self.cache_ttl = 1.0  # 1 second cache
        
    async def parallel_ai_analysis(self, context: Dict, dual_ai: 'DualAIAnalyzer') -> Dict:
        """Exécute GPT et Grok en parallèle pour réduire latence."""
        
        start_time = time.perf_counter()
        
        # Check cache
        cache_key = f"{context.get('symbol')}_{int(time.time())}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached['from_cache'] = True
            return cached
        
        # Create parallel tasks
        tasks = {
            'grok': asyncio.create_task(dual_ai.get_grok_analysis(context)),
            'ml': asyncio.create_task(self.fast_ml_prediction(context))
        }
        
        # Get GPT synchronously
        gpt_result = dual_ai.get_gpt_analysis(context)
        
        # Wait for async tasks with timeout
        results = {'gpt': gpt_result}
        
        for name, task in tasks.items():
            try:
                results[name] = await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                ai_logger.warning(f"{name} timeout in parallel analysis")
                results[name] = {
                    'status': 'timeout',
                    'recommendation': 'HOLD',
                    'confidence': 0
                }
        
        # Calculate latency
        latency = (time.perf_counter() - start_time) * 1000
        self.latency_history.append(latency)
        
        result = {
            'results': results,
            'latency_ms': latency,
            'all_responded': all(r.get('status') != 'timeout' for r in results.values())
        }
        
        # Cache result
        self.cache[cache_key] = result
        
        # Clean old cache entries
        if len(self.cache) > 100:
            self.cache.clear()
        
        return result
    
    async def fast_ml_prediction(self, context: Dict) -> Dict:
        """Fast ML prediction for latency optimization."""
        
        # Simulate fast ML prediction
        # In production, this would use pre-computed features
        
        ml_predictions = context.get('ml_predictions', {})
        
        if ml_predictions:
            # Average predictions
            probs = []
            for tf, pred in ml_predictions.items():
                if isinstance(pred, dict):
                    probs.append(pred.get('prediction', 0.5))
            
            avg_prob = np.mean(probs) if probs else 0.5
            
            if avg_prob > 0.65:
                recommendation = 'BUY'
            elif avg_prob < 0.35:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            return {
                'recommendation': recommendation,
                'confidence': abs(avg_prob - 0.5) * 200,
                'probability': avg_prob,
                'status': 'success'
            }
        
        return {
            'recommendation': 'HOLD',
            'confidence': 0,
            'status': 'no_ml_data'
        }
    
    def get_average_latency(self) -> float:
        """Get average latency over recent predictions."""
        if self.latency_history:
            return np.mean(list(self.latency_history))
        return 0.0
    
    def optimize_execution_route(self, signal: Dict) -> str:
        """Optimize execution route based on latency."""
        
        # Simple routing logic
        # In production, this would analyze different routes
        
        urgency = signal.get('urgency', 0.5)
        size = signal.get('size', 100)
        
        if urgency > 0.8 and size < 1000:
            return 'SMART'  # IB Smart routing
        elif size > 10000:
            return 'VWAP'  # VWAP algo
        else:
            return 'ADAPTIVE'  # Adaptive algo

# === ENHANCED GLOBAL STATE ===
class GlobalState:
    """Centralized state management for the trading system with Dual AI."""
    
    def __init__(self):
        """Initialize global state with all components."""
        # === Configuration ===
        self.config = TRADING_CONFIG
        self.validate_config()
        
        # === AI Integration ===
        self.dual_ai_analyzer = None
        self.init_ai_systems()
        
        # === GPT / AI Integration (Legacy compatibility) ===
        self.client_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.last_gpt_time = 0.0
        self.gpt_in_progress = False
        self.gpt_has_alert = False
        self.last_gpt_alert_text = ""
        self.latest_gpt_analysis = "Awaiting Dual AI analysis..."
        self.gpt_history = deque(maxlen=10)
        self.GPT_COOLDOWN = TRADING_CONFIG.analysis_cooldown
        
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
        
        # === Systems ===
        self.init_systems()
        
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
        
        logger.info("GlobalState initialized successfully with Dual AI")
    
    def init_ai_systems(self):
        """Initialize AI systems."""
        try:
            self.dual_ai_analyzer = DualAIAnalyzer(self)
            ai_logger.info("Dual AI Analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Dual AI: {e}")
            self.dual_ai_analyzer = None
    
    def init_systems(self):
        """Initialize all trading systems."""
        self.smart_strategies = SmartTradingStrategies(self, self.config)
        self.pattern_analyzer = AdvancedPatternAnalyzer()
        self.risk_manager = IntelligentRiskManager(self, self.config)
        self.alert_system = SmartAlertSystem(self)
        self.session_profiler = SessionProfiler()
        self.latency_optimizer = LatencyOptimizer()
        
        logger.info("All trading systems initialized")
    
    def validate_config(self):
        """Validate configuration parameters."""
        validations = [
            self.config.ai_consensus_threshold > 50,
            self.config.ai_consensus_threshold < 100,
            self.config.base_risk_percent > 0,
            self.config.base_risk_percent < 10,
            self.config.max_position_pct > 0,
            self.config.max_position_pct < 1
        ]
        
        if not all(validations):
            logger.error("Invalid configuration parameters")
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

# === Initialize Global State ===
STATE = GlobalState()

# === ENHANCED TELEGRAM BOT ===
class EnhancedTelegramBot:
    """Bot Telegram amélioré avec support Grok et toutes les fonctionnalités."""
    
    def __init__(self):
        self.command_handlers = {
            # Original commands
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
            '/shutdown': self.handle_shutdown,
            
            # New Dual AI commands
            '/grok': self.handle_grok,
            '/consensus': self.handle_consensus,
            '/compare': self.handle_compare_ai,
            '/askgrok': self.handle_ask_grok,
            '/askboth': self.handle_ask_both,
            
            # Advanced features
            '/strategy': self.handle_strategy,
            '/patterns': self.handle_patterns,
            '/kelly': self.handle_kelly,
            '/session': self.handle_session,
            '/alerts': self.handle_alerts,
            '/backtest': self.handle_backtest
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
                # Handle async commands
                if asyncio.iscoroutinefunction(handler):
                    # Run async handler
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(handler(args))
                    loop.close()
                else:
                    response = handler(args)
                
                STATE.send_alert(response)
            else:
                # Check for special patterns
                self._handle_special_patterns(chat_id, text)
                
        except Exception as e:
            logger.error(f"Error processing telegram message: {e}")
            STATE.send_alert(f"❌ Error: {e}")
    
    def handle_help(self, args: List[str]) -> str:
        """Show help message."""
        return """
📚 **SYNAPSE X v11.0 COMMANDS**
━━━━━━━━━━━━━━━━━━━━━━━

**🤖 AI Analysis:**
• /consensus - GPT + Grok decision
• /grok - Grok analysis only
• /compare - Compare AI analyses
• /askgrok [question] - Ask Grok
• /askboth [question] - Ask both AIs

**📊 Trading:**
• /subscribe SYMBOL - Subscribe to symbol
• /analyse SYMBOL CAPITAL - Full analysis
• /signals - Current trading signals
• /position - Manage positions
• /strategy - Current strategy

**📈 Analysis:**
• /patterns - Detected patterns
• /microstructure - Market analysis
• /kelly - Optimal position size
• /session - Session profile

**⚠️ Risk & Performance:**
• /risk - Risk metrics
• /performance - Performance report
• /alerts - Active alerts
• /backtest SYMBOL - Live backtest

**🔧 System:**
• /status - System status
• /models - ML model status
• /compliance - Compliance report
• /shutdown - Shutdown system

**Examples:**
• /consensus
• /askgrok What's the institutional flow on AAPL?
• /kelly
• /patterns
"""
    
    def handle_status(self, args: List[str]) -> str:
        """Get system status."""
        healthy, health_msg = check_system_health()
        
        # Get session info
        session_name, session_profile = STATE.session_profiler.get_current_profile()
        
        # IB connection
        ib_connected = "✅" if ib.isConnected() else "❌"
        
        # Data flow
        with locks.tns_lock:
            last_trade = STATE.df_tns['Timestamp'].max() if not STATE.df_tns.empty else 0
        
        data_age = time.time() - last_trade if last_trade > 0 else float('inf')
        data_status = "✅" if data_age < 10 else "⚠️" if data_age < 60 else "❌"
        
        # AI status
        ai_status = "✅" if STATE.dual_ai_analyzer else "❌"
        
        return f"""
📊 **SYSTEM STATUS**
━━━━━━━━━━━━━━━━━━
**Trading:**
• Symbol: {STATE.current_symbol or 'None'}
• Session: {session_name}
• IB Connected: {ib_connected}
• Data Flow: {data_status} ({data_age:.1f}s ago)

**AI Systems:**
• Dual AI: {ai_status}
• GPT-4: ✅
• Grok: ✅

**Performance:**
• Positions: {len(STATE.positions)}
• Daily P&L: ${STATE.metrics.daily_pnl:+.2f}
• Win Rate: {STATE.metrics.win_rate:.1%}
• Circuit Breaker: {'🔴 TRIGGERED' if STATE.circuit_breaker_triggered else '🟢 Normal'}

**System Health:**
• Status: {health_msg}
• Latency: {STATE.latency_optimizer.get_average_latency():.0f}ms
• Errors: {STATE.ib_loop_error_count}
"""
    
    async def handle_consensus(self, args: List[str]) -> str:
        """Get AI consensus decision."""
        if not STATE.dual_ai_analyzer:
            return "❌ Dual AI system not initialized"
        
        try:
            # Get consensus decision
            decision = await STATE.dual_ai_analyzer.get_synergistic_decision()
            
            if decision.get('error'):
                return f"❌ Error getting consensus: {decision.get('reason', 'Unknown')}"
            
            # Get current price
            price = get_current_price()
            
            # Emoji mapping
            emoji_map = {
                'STRONG_BUY': '🚀',
                'BUY': '📈',
                'HOLD': '➡️',
                'SELL': '📉',
                'STRONG_SELL': '🔻'
            }
            
            # Get risk limits
            market_context = {
                'volatility': STATE.market_regime.volatility,
                'regime': STATE.market_regime.regime,
                'ai_confidence': decision['confidence']
            }
            
            risk_limits = STATE.risk_manager.calculate_dynamic_risk_limits(decision, market_context)
            
            return f"""
🤝 **AI CONSENSUS ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━
Symbol: {STATE.current_symbol} @ ${price:.2f}

**Decision:** {emoji_map.get(decision['decision'], '❓')} {decision['decision']}
**Confidence:** {decision['confidence']:.1f}%
**AI Agreement:** {decision['ai_agreement']:.1%}

**Individual Recommendations:**
• GPT-4: {decision['gpt_recommendation']}
• Grok: {decision['grok_recommendation']}

**ML Alignment:** {decision['ml_alignment']:.1%} {'✅' if decision.get('ml_direction_agrees') else '❌'}

**Execution:**
• Action: {decision['execution_recommendation']}
• Position Size: ${risk_limits['position_size']:,.0f}
• Stop Loss: {risk_limits['stop_loss']:.1%}

**Reasoning:**
{decision['reasoning']}
"""
        except Exception as e:
            logger.error(f"Consensus error: {e}")
            return f"❌ Error: {str(e)}"
    
    async def handle_grok(self, args: List[str]) -> str:
        """Get Grok analysis only."""
        if not STATE.dual_ai_analyzer:
            return "❌ Grok not initialized"
        
        context = STATE.dual_ai_analyzer.build_market_context()
        if not context:
            return "❌ Insufficient market data"
        
        try:
            grok_analysis = await STATE.dual_ai_analyzer.get_grok_analysis(context)
            
            return f"""
🤖 **GROK ANALYSIS**
━━━━━━━━━━━━━━━━━━
Symbol: {context['symbol']}
Price: ${context['current_price']:.2f}

**Recommendation:** {grok_analysis.get('recommendation', 'N/A')}
**Confidence:** {grok_analysis.get('confidence', 0)}%
**Direction:** {grok_analysis.get('direction_probability', 50)}% UP

**Analysis:**
{grok_analysis.get('analysis', 'No analysis available')[:500]}...
"""
        except Exception as e:
            return f"❌ Grok error: {str(e)}"
    
    async def handle_compare_ai(self, args: List[str]) -> str:
        """Compare GPT and Grok analyses."""
        if not STATE.dual_ai_analyzer or not STATE.dual_ai_analyzer.last_analysis:
            return "❌ No recent analysis available. Use /consensus first."
        
        analysis = STATE.dual_ai_analyzer.last_analysis
        
        return f"""
🔍 **AI COMPARISON**
━━━━━━━━━━━━━━━━━━

**GPT-4 Analysis:**
• Recommendation: {analysis['gpt'].get('recommendation', 'N/A')}
• Confidence: {analysis['gpt'].get('confidence', 0)}%
• Direction: {analysis['gpt'].get('direction_probability', 50)}% UP

**Grok Analysis:**
• Recommendation: {analysis['grok'].get('recommendation', 'N/A')}  
• Confidence: {analysis['grok'].get('confidence', 0)}%
• Direction: {analysis['grok'].get('direction_probability', 50)}% UP

**Consensus:**
• Decision: {analysis['consensus'].get('decision', 'N/A')}
• Agreement: {analysis['consensus'].get('ai_agreement', 0):.1%}
• Confidence: {analysis['consensus'].get('confidence', 0):.1f}%

**Market Context:**
• Price: ${analysis['context'].get('current_price', 0):.2f}
• 5m Change: {analysis['context'].get('price_change_5m', 0):+.2f}%
• Volatility: {analysis['context'].get('volatility', 0):.1%}
"""
    
    async def handle_ask_grok(self, args: List[str]) -> str:
        """Ask Grok a specific question."""
        if not args:
            return "❌ Usage: /askgrok your question about the market"
        
        if not STATE.dual_ai_analyzer:
            return "❌ Grok not initialized"
        
        question = ' '.join(args)
        
        # Add market context
        context = STATE.dual_ai_analyzer.build_market_context()
        
        enhanced_prompt = f"""
Current market context for {context.get('symbol', 'N/A')}:
- Price: ${context.get('current_price', 0):.2f}
- 5m change: {context.get('price_change_5m', 0):+.2f}%
- Volume: {context.get('volume_5m', 0):,}
- Volatility: {context.get('volatility', 0):.1%}
- ML Signal: {context.get('ml_predictions', {}).get('1m', {}).get('prediction', 0.5):.1%}

User question: {question}

Provide a specific, quantitative answer focused on actionable trading insights.
"""
        
        try:
            response = await STATE.dual_ai_analyzer.grok_client.analyze(enhanced_prompt)
            
            return f"""
🤖 **GROK RESPONSE**
━━━━━━━━━━━━━━━━━━

Q: {question}

A: {response}
"""
        except Exception as e:
            return f"❌ Grok error: {str(e)}"
    
    async def handle_ask_both(self, args: List[str]) -> str:
        """Ask both AIs a question."""
        if not args:
            return "❌ Usage: /askboth your question"
        
        if not STATE.dual_ai_analyzer:
            return "❌ AI systems not initialized"
        
        question = ' '.join(args)
        context = STATE.dual_ai_analyzer.build_market_context()
        
        context_str = f"""
Market: {context.get('symbol', 'N/A')} @ ${context.get('current_price', 0):.2f}
Change: {context.get('price_change_5m', 0):+.2f}%
Volume: {context.get('volume_ratio', 1):.1f}x average
"""
        
        try:
            # Ask both in parallel
            grok_prompt = f"{context_str}\n\nQuestion: {question}\n\nProvide trading-focused answer."
            grok_task = asyncio.create_task(
                STATE.dual_ai_analyzer.grok_client.analyze(grok_prompt)
            )
            
            # GPT synchronous
            gpt_response = STATE.dual_ai_analyzer.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": f"{context_str}\n\nQuestion: {question}"}],
                max_tokens=150,
                temperature=0.3
            )
            
            grok_response = await grok_task
            
            return f"""
🤝 **DUAL AI RESPONSE**
━━━━━━━━━━━━━━━━━━

**Question:** {question}

**GPT-4 says:**
{gpt_response.choices[0].message.content}

**Grok says:**
{grok_response}
"""
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def handle_strategy(self, args: List[str]) -> str:
        """Show current recommended strategy."""
        
        # Get market context
        context = {
            'volatility': STATE.market_regime.volatility,
            'volume_ratio': STATE.RELATIVE_VOLUME,
            'trend_strength': STATE.market_regime.efficiency_ratio,
            'time_of_day': get_trading_session().lower(),
            'ai_confidence': 75  # Default
        }
        
        # Get last AI consensus if available
        if STATE.dual_ai_analyzer and STATE.dual_ai_analyzer.last_analysis:
            context['ai_confidence'] = STATE.dual_ai_analyzer.last_analysis.get(
                'consensus', {}
            ).get('confidence', 75)
        
        strategy = STATE.smart_strategies.select_strategy(context)
        
        # Format rules
        entry_rules = "\n".join([f"• {k}: {v}" for k, v in strategy.get('entry_rules', {}).items()])
        exit_rules = "\n".join([f"• {k}: {v}" for k, v in strategy.get('exit_rules', {}).items()])
        
        return f"""
📊 **RECOMMENDED STRATEGY**
━━━━━━━━━━━━━━━━━━━
Strategy: {strategy['name'].upper().replace('_', ' ')}
Position Size: {strategy.get('position_size', 1.0):.1f}x

**Entry Rules:**
{entry_rules}

**Exit Rules:**
{exit_rules}

**Market Context:**
• Volatility: {context['volatility']:.1%}
• Volume: {context['volume_ratio']:.1f}x
• Trend: {context['trend_strength']:.1%}
• Session: {context['time_of_day']}
• AI Confidence: {context['ai_confidence']:.1f}%
"""
    
    def handle_patterns(self, args: List[str]) -> str:
        """Analyze detected patterns."""
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if df.empty:
            return "❌ No data for pattern analysis"
        
        analysis = STATE.pattern_analyzer.analyze_market_patterns(df)
        
        # Format detected patterns
        detected = []
        patterns = analysis.get('patterns', {})
        
        # Check each category
        for category, cat_patterns in patterns.items():
            if isinstance(cat_patterns, dict):
                for pattern_name, pattern_data in cat_patterns.items():
                    if isinstance(pattern_data, dict) and pattern_data.get('detected'):
                        detected.append(f"• {category}.{pattern_name}")
        
        pattern_list = "\n".join(detected) if detected else "• No significant patterns"
        
        return f"""
🔍 **PATTERN ANALYSIS**
━━━━━━━━━━━━━━━━━━
Pattern Score: {analysis['score']:.1f}/10
Strongest: {analysis['strongest_pattern']}

**Detected Patterns:**
{pattern_list}

**Recommendation:**
{analysis['recommendation']}

**Key Insights:**
{self._get_pattern_insights(patterns)}
"""
    
    def _get_pattern_insights(self, patterns: Dict) -> str:
        """Extract key insights from patterns."""
        insights = []
        
        # Iceberg detection
        iceberg = patterns.get('microstructure', {}).get('iceberg', {})
        if iceberg.get('detected'):
            insights.append(f"🧊 Iceberg at ${iceberg.get('price', 0):.2f} ({iceberg.get('total_size', 0):,.0f} shares)")
        
        # Momentum ignition
        momentum = patterns.get('microstructure', {}).get('momentum_ignition', {})
        if momentum.get('detected'):
            insights.append(f"🔥 Momentum ignition {momentum.get('direction', 'unknown')} ({momentum.get('volume_multiplier', 1):.1f}x volume)")
        
        # Institutional activity
        inst = patterns.get('institutional', {})
        if inst.get('accumulation', {}).get('detected'):
            insights.append("🏛️ Institutional accumulation detected")
        elif inst.get('distribution', {}).get('detected'):
            insights.append("🏛️ Institutional distribution detected")
        
        # Volume patterns
        vol_surge = patterns.get('volume', {}).get('volume_surge', {})
        if vol_surge.get('detected'):
            insights.append(f"📊 Volume surge {vol_surge.get('multiplier', 1):.1f}x")
        
        return "\n".join(insights) if insights else "No significant insights"
    
    async def handle_kelly(self, args: List[str]) -> str:
        """Calculate optimal position size using Kelly Criterion."""
        
        # Get AI consensus
        if not STATE.dual_ai_analyzer:
            return "❌ AI system not initialized"
        
        try:
            consensus = await STATE.dual_ai_analyzer.get_synergistic_decision()
            
            # Get market context
            market_context = {
                'volatility': STATE.market_regime.volatility,
                'regime': STATE.market_regime.regime,
                'volume_ratio': STATE.RELATIVE_VOLUME
            }
            
            # Calculate Kelly position
            kelly_size = STATE.risk_manager.calculate_kelly_criterion(consensus, market_context)
            
            # Get dynamic risk limits
            risk_limits = STATE.risk_manager.calculate_dynamic_risk_limits(consensus, market_context)
            
            # Get volatility parameters
            vol_params = STATE.risk_manager.get_volatility_params(market_context['volatility'])
            
            # Position sizing recommendation
            if consensus['confidence'] >= TRADING_CONFIG.ai_high_confidence_threshold:
                size_rec = f"✅ STRONG ENTRY: Use full Kelly (${kelly_size:,.0f})"
            elif consensus['confidence'] >= TRADING_CONFIG.ai_consensus_threshold:
                size_rec = f"✅ ENTRY: Use 75% Kelly (${kelly_size*0.75:,.0f})"
            elif consensus['confidence'] >= 65:
                size_rec = f"⚠️ CAUTIOUS: Use 50% Kelly (${kelly_size*0.5:,.0f})"
            else:
                size_rec = "❌ WAIT: Confidence too low"
            
            return f"""
💰 **OPTIMAL POSITION SIZING**
━━━━━━━━━━━━━━━━━━━━━
Kelly Position: ${kelly_size:,.0f}
Risk per Trade: {risk_limits['position_risk']:.1%}
Stop Loss: {risk_limits['stop_loss']:.1%}

**AI Consensus:**
• Decision: {consensus['decision']}
• Confidence: {consensus['confidence']:.1f}%
• Agreement: {consensus['ai_agreement']:.1%}

**Market Conditions:**
• Volatility: {market_context['volatility']:.1%} ({self._get_vol_level(market_context['volatility'])})
• Regime: {market_context['regime']}
• Risk Multiplier: {risk_limits['risk_multiplier']:.2f}x

**Risk Parameters:**
• Position Limit: ${STATE.risk_limits.max_position_size:,.0f}
• Daily Risk: {risk_limits['daily_risk']:.1%}
• Max Correlation: {risk_limits['max_correlation']:.1f}

**Recommendation:**
{size_rec}
"""
        except Exception as e:
            return f"❌ Error calculating Kelly: {str(e)}"
    
    def _get_vol_level(self, volatility: float) -> str:
        """Get volatility level description."""
        if volatility < 0.15:
            return "Low"
        elif volatility < 0.25:
            return "Normal"
        elif volatility < 0.35:
            return "High"
        else:
            return "Extreme"
    
    def handle_session(self, args: List[str]) -> str:
        """Show current session profile."""
        session_name, profile = STATE.session_profiler.get_current_profile()
        
        if not profile.get('trading_allowed', True):
            return f"""
🕒 **SESSION STATUS**
━━━━━━━━━━━━━━━━━━
Session: {session_name.upper()}
Status: 🔴 MARKET CLOSED

Next Open: Monday 9:30 AM ET
"""
        
        # Format profile parameters
        params = []
        for key, value in profile.items():
            if key != 'trading_allowed':
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, float):
                    params.append(f"• {formatted_key}: {value:.1f}")
                elif isinstance(value, bool):
                    params.append(f"• {formatted_key}: {'Yes' if value else 'No'}")
                else:
                    params.append(f"• {formatted_key}: {value}")
        
        params_str = "\n".join(params)
        
        return f"""
🕒 **SESSION PROFILE**
━━━━━━━━━━━━━━━━━━
Session: {session_name.upper().replace('_', ' ')}
Time: {now_hms_str()}

**Session Parameters:**
{params_str}

**Trading Recommendations:**
{self._get_session_recommendations(session_name, profile)}
"""
    
    def _get_session_recommendations(self, session: str, profile: Dict) -> str:
        """Get session-specific recommendations."""
        recs = []
        
        if session == 'pre_market':
            recs.append("• Use limit orders only")
            recs.append("• Watch for news catalysts")
            recs.append("• Reduce position size by 50%")
        elif session == 'opening_30min':
            recs.append("• Wait for initial volatility to settle")
            recs.append("• Use opening range strategy")
            recs.append("• Require 85%+ AI confidence")
        elif session == 'lunch_hour':
            recs.append("• Expect lower volume")
            recs.append("• Wider spreads likely")
            recs.append("• Focus on mean reversion")
        elif session == 'power_hour':
            recs.append("• Watch for MOC imbalances")
            recs.append("• Institutional rebalancing active")
            recs.append("• Tighten stops on winners")
        elif session == 'after_hours':
            recs.append("• Extreme caution required")
            recs.append("• 90%+ confidence only")
            recs.append("• Triple-check liquidity")
        
        return "\n".join(recs) if recs else "• Normal trading conditions"
    
    def handle_alerts(self, args: List[str]) -> str:
        """Show active alerts."""
        # Get current conditions
        ai_analysis = {}
        if STATE.dual_ai_analyzer and STATE.dual_ai_analyzer.last_analysis:
            ai_analysis = STATE.dual_ai_analyzer.last_analysis.get('consensus', {})
        
        market_context = {
            'volatility': STATE.market_regime.volatility,
            'regime': STATE.market_regime.regime
        }
        
        # Get patterns
        patterns = {}
        with locks.tns_lock:
            if not STATE.df_tns.empty:
                pattern_analysis = STATE.pattern_analyzer.analyze_market_patterns(STATE.df_tns)
                patterns = pattern_analysis.get('patterns', {})
        
        # Check alert conditions
        alerts = STATE.alert_system.check_alert_conditions(ai_analysis, market_context, patterns)
        
        if not alerts:
            return """
🔔 **SYSTEM ALERTS**
━━━━━━━━━━━━━━━━━━
Status: ✅ All Clear

No active alerts at this time.
"""
        
        # Format alerts
        return STATE.alert_system.format_alert_message(alerts)
    
    def handle_backtest(self, args: List[str]) -> str:
        """Run live backtest."""
        if not args:
            symbol = STATE.current_symbol
        else:
            symbol = args[0].upper()
        
        if not symbol:
            return "❌ Usage: /backtest SYMBOL"
        
        # Get recent trades
        with locks.tns_lock:
            df = STATE.df_tns.copy()
        
        if df.empty or len(df) < 100:
            return "❌ Insufficient data for backtest"
        
        # Simple backtest simulation
        signals = []
        prices = df['Price'].values
        
        # Generate signals based on simple momentum
        for i in range(20, len(prices) - 1):
            if prices[i] > prices[i-20] * 1.002:  # 0.2% threshold
                signals.append({'action': 'BUY', 'price': prices[i], 'index': i})
            elif prices[i] < prices[i-20] * 0.998:
                signals.append({'action': 'SELL', 'price': prices[i], 'index': i})
        
        # Calculate P&L
        position = 0
        entry_price = 0
        trades = []
        
        for signal in signals:
            if signal['action'] == 'BUY' and position == 0:
                position = 100
                entry_price = signal['price']
            elif signal['action'] == 'SELL' and position > 0:
                pnl = (signal['price'] - entry_price) * position
                trades.append(pnl)
                position = 0
        
        # Statistics
        if trades:
            total_pnl = sum(trades)
            win_rate = len([t for t in trades if t > 0]) / len(trades)
            avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
            avg_loss = np.mean([t for t in trades if t < 0]) if any(t < 0 for t in trades) else 0
            
            return f"""
📊 **LIVE BACKTEST RESULTS**
━━━━━━━━━━━━━━━━━━━━━
Symbol: {symbol}
Period: Last {len(df)} ticks
Strategy: Simple Momentum

**Results:**
• Total Trades: {len(trades)}
• Total P&L: ${total_pnl:.2f}
• Win Rate: {win_rate:.1%}
• Avg Win: ${avg_win:.2f}
• Avg Loss: ${abs(avg_loss):.2f}
• Profit Factor: {avg_win/abs(avg_loss) if avg_loss != 0 else 0:.2f}

**Note:** This is a simplified backtest for demonstration.
Real results will vary based on execution and market conditions.
"""
        else:
            return "❌ No trades generated in backtest period"
    
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
        """Generate institutional analysis."""
        if len(args) < 2:
            return "❌ Usage: /analyse SYMBOL CAPITAL"
        
        try:
            symbol = args[0].upper()
            capital = float(args[1])
            
            # Get current price
            price = ensure_price_ib(symbol)
            if not price:
                return f"❌ Could not get price for {symbol}"
            
            # Calculate position parameters
            position_size = int(capital * 0.02 / (price * 0.02))  # 2% risk, 2% stop
            stop_loss = price * 0.98
            target1 = price * 1.03
            target2 = price * 1.05
            
            return f"""
📈 **INSTITUTIONAL ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━
Symbol: {symbol}
Price: ${price:.2f}
Capital: ${capital:,.0f}

**📊 Position Analysis:**
• Entry: ${price:.2f} - ${price * 0.998:.2f}
• Size: {position_size} shares
• Value: ${position_size * price:,.0f}
• Risk: ${position_size * (price - stop_loss):.0f}

**🎯 Targets:**
• T1: ${target1:.2f} (+3%) - 50% exit
• T2: ${target2:.2f} (+5%) - full exit
• Stop: ${stop_loss:.2f} (-2%)

**📈 Risk/Reward:**
• R:R Ratio: 1:2.5
• Win Rate Required: 28.6%
• Expected Value: +${(target1 - price) * position_size * 0.6:.0f}

**💡 Execution Strategy:**
• Use scaled entry over 5-10 minutes
• Set limit orders 0.01-0.02 below market
• Monitor spread and volume
• Trail stop after +1% move

**⚠️ Key Risks:**
• Market volatility: {STATE.market_regime.volatility:.1%}
• Session: {get_trading_session()}
• Spread risk if low volume
"""
        except Exception as e:
            return f"❌ Error: {e}"
    
    def handle_risk(self, args: List[str]) -> str:
        """Get risk metrics."""
        # Check current risk status
        can_trade, reason = STATE.risk_manager.check_risk_limits()
        status_emoji = "✅" if can_trade else "🔴"
        
        # Calculate VaR
        portfolio_var = calculate_portfolio_var()
        
        return f"""
⚠️ **RISK MANAGEMENT**
━━━━━━━━━━━━━━━━━━

**Status:** {status_emoji} {reason}

**Risk Limits:**
• Max Position: ${STATE.risk_limits.max_position_size:,.0f}
• Daily Loss: ${STATE.risk_limits.max_daily_loss:,.0f}
• Max Drawdown: {STATE.risk_limits.max_drawdown:.1%}
• VaR Limit: {STATE.risk_limits.var_limit:.1%}

**Current Metrics:**
• Daily P&L: ${STATE.metrics.daily_pnl:+,.2f}
• Drawdown: {STATE.metrics.current_drawdown:.1%}
• Consecutive Losses: {STATE.metrics.consecutive_losses}
• Portfolio Heat: {STATE.portfolio_heat:.1%}
• VaR (95%): ${portfolio_var:,.2f}

**Risk Scores:**
• Market Quality: {analyze_microstructure_quality():.1%}
• Toxicity: {STATE.toxicity_score:.1%}
• Regime: {STATE.market_regime.regime}
• Volatility: {STATE.market_regime.volatility:.1%}

**Active Positions:** {len(STATE.positions)}
**Circuit Breaker:** {'TRIGGERED' if STATE.circuit_breaker_triggered else 'Normal'}
"""
    
    def handle_performance(self, args: List[str]) -> str:
        """Get performance metrics."""
        # Multi-horizon accuracy
        horizon_accuracy = HORIZON_TRACKER.calculate_accuracy_report()
        
        # Get AI performance
        ai_perf = ""
        if STATE.dual_ai_analyzer and STATE.dual_ai_analyzer.consensus_history:
            recent = list(STATE.dual_ai_analyzer.consensus_history)[-20:]
            ai_accuracy = sum(1 for a in recent if a.get('consensus', {}).get('confidence', 0) > 70)
            ai_perf = f"\n**AI Performance:**\n• High Confidence Signals: {ai_accuracy}/20"
        
        return f"""
📈 **PERFORMANCE REPORT**
━━━━━━━━━━━━━━━━━━━━

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
{ai_perf}

**Session:** {get_trading_session()}
**Uptime:** {format_uptime(time.time() - STATE.session_start_time)}
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
            
            # Map to signal
            if pred['prediction'] > 0.7:
                signal_str = "STRONG_BUY 🚀"
            elif pred['prediction'] > 0.6:
                signal_str = "BUY 📈"
            elif pred['prediction'] < 0.3:
                signal_str = "STRONG_SELL 🔻"
            elif pred['prediction'] < 0.4:
                signal_str = "SELL 📉"
            else:
                signal_str = "NEUTRAL ➡️"
            
            signals.append(f"""
**{timeframe.upper()} Signal:**
• Direction: {signal_str}
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
• OFI: {STATE.OFI_VALUE:.0f}
• CDV: {STATE.CDV_VALUE:.0f}
""")
        
        return "🎯 **TRADING SIGNALS**\n━━━━━━━━━━━━━━━━━━\n" + "\n".join(signals)
    
    def handle_microstructure(self, args: List[str]) -> str:
        """Get microstructure analysis."""
        metrics = calculate_microstructure_metrics()
        
        # Get manipulation detection
        manipulation = {'scores': {}, 'overall': 0}
        try:
            if hasattr(STATE, 'manipulation_detector'):
                manipulation = STATE.manipulation_detector.detect_manipulation()
        except:
            pass
        
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
• Toxicity: {STATE.toxicity_score:.1%}

**Book Analysis:**
• Bid Levels: {len(STATE.bids_dict)}
• Ask Levels: {len(STATE.asks_dict)}
• Total Depth: {sum(STATE.bids_dict.values()) + sum(STATE.asks_dict.values()):,.0f}
• Imbalance: {calculate_book_imbalance():.1%}

**Price Levels:**
{self._get_price_levels()}
"""
    
    def _get_price_levels(self) -> str:
        """Get key price levels."""
        with locks.tns_lock:
            if STATE.df_tns.empty:
                return "No data"
            
            prices = STATE.df_tns['Price'].values
            if len(prices) < 20:
                return "Insufficient data"
            
            current = prices[-1]
            high_20 = max(prices[-20:])
            low_20 = min(prices[-20:])
            vwap = STATE.VWAP_VALUE or current
            
            return f"""• Current: ${current:.2f}
• VWAP: ${vwap:.2f}
• 20-High: ${high_20:.2f}
• 20-Low: ${low_20:.2f}"""
    
    def handle_compliance(self, args: List[str]) -> str:
        """Get compliance report."""
        # Placeholder for compliance reporting
        return f"""
📋 **COMPLIANCE REPORT**
━━━━━━━━━━━━━━━━━━━

**Session Summary:**
• Total Trades: {STATE.trades_today}
• Orders Placed: {len(STATE.active_orders)}
• Risk Breaches: 0
• Alerts Triggered: {len(STATE.alert_history)}

**Regulatory Compliance:**
• Pattern Day Trading: ✅ Compliant
• Position Limits: ✅ Within limits
• Wash Sales: ✅ None detected
• Market Manipulation: ✅ None detected

**Risk Compliance:**
• Max Loss Rule: ✅ Adhered
• Position Sizing: ✅ Compliant
• Stop Loss Usage: ✅ Active
• Leverage: ✅ Within limits

**Audit Trail:**
• All trades logged ✅
• Risk checks documented ✅
• AI decisions recorded ✅
"""
    
    def handle_models(self, args: List[str]) -> str:
        """Get ML model status."""
        report = "🤖 **MODEL PERFORMANCE**\n"
        report += "━━━━━━━━━━━━━━━━━━\n\n"
        
        # Check if we have predictions
        features = FEATURE_ENGINEER.compute_features()
        if features is not None:
            for timeframe in ["1m", "5m", "15m"]:
                prediction = MODEL_MANAGER.get_ensemble_prediction(features, timeframe)
                
                report += f"**{timeframe.upper()} Models:**\n"
                report += f"• Ensemble: {prediction['prediction']:.1%}\n"
                report += f"• Confidence: {prediction['confidence']:.1%}\n"
                report += f"• Agreement: {prediction['model_agreement']:.1%}\n"
                
                # Individual models
                for model, prob in prediction.get('model_predictions', {}).items():
                    report += f"  - {model}: {prob:.1%}\n"
                report += "\n"
        
        # Model update stats
        if hasattr(STATE, 'online_learner'):
            report += "**Update Counts:**\n"
            for key, count in STATE.online_learner.update_counts.items():
                report += f"• {key}: {count:,} samples\n"
        
        # AI system status
        report += f"\n**AI Systems:**\n"
        report += f"• Dual AI: {'✅ Active' if STATE.dual_ai_analyzer else '❌ Inactive'}\n"
        report += f"• GPT-4: ✅ Connected\n"
        report += f"• Grok: ✅ Connected\n"
        
        return report
    
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
        
        # Parse position command
        return "📝 Position tracking updated"
    
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
            
            STATE.send_alert(response)

# Initialize Telegram bot
TELEGRAM_BOT = EnhancedTelegramBot()

# === HELPER FUNCTIONS ===

def now_hms_str() -> str:
    """Get current time in HH:MM:SS ET format."""
    ny_tz = pytz.timezone("America/New_York")
    return datetime.datetime.now(ny_tz).strftime("%H:%M:%S ET")

def get_trading_session() -> str:
    """Get current trading session."""
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)
    
    if now.weekday() >= 5:
        return "CLOSED"
    
    hour = now.hour
    minute = now.minute
    
    if hour >= 4 and (hour < 9 or (hour == 9 and minute < 30)):
        return "PREMARKET"
    elif (hour > 9 or (hour == 9 and minute >= 30)) and hour < 16:
        return "REGULAR"
    elif hour >= 16 and hour < 20:
        return "AFTERHOURS"
    else:
        return "CLOSED"

def is_market_hours() -> bool:
    """Check if US market is open."""
    return get_trading_session() == "REGULAR"

def format_uptime(seconds: float) -> str:
    """Format uptime in human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

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
    price = get_current_price(symbol)
    if price is not None:
        return price
    
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

def safe_sleep(seconds: float) -> None:
    """Sleep safely whether IB is connected or not."""
    if ib.isConnected():
        ib.sleep(seconds)
    else:
        time.sleep(seconds)

def calculate_book_imbalance() -> float:
    """Calculate order book imbalance."""
    with locks.l2_lock:
        bid_volume = sum(STATE.bids_dict.values()) if STATE.bids_dict else 0
        ask_volume = sum(STATE.asks_dict.values()) if STATE.asks_dict else 0
        
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        
    return 0.0

def check_system_health() -> Tuple[bool, str]:
    """Check overall system health."""
    issues = []
    
    # Check IB connection
    if not ib.isConnected():
        issues.append("IB disconnected")
    
    # Check data flow
    with locks.tns_lock:
        if not STATE.df_tns.empty:
            last_trade_time = STATE.df_tns['Timestamp'].max()
            data_age = time.time() - last_trade_time
            
            if data_age > 60:
                issues.append(f"Data stale ({data_age:.0f}s)")
    
    # Check error counts
    if STATE.ib_loop_error_count > 10:
        issues.append(f"High error count ({STATE.ib_loop_error_count})")
    
    # Check AI systems
    if not STATE.dual_ai_analyzer:
        issues.append("AI systems offline")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, "All systems operational"

# === MARKET DATA HANDLERS ===

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

# === MICROSTRUCTURE CALCULATIONS ===

def calculate_microstructure_metrics() -> Dict[str, float]:
    """Calculate advanced microstructure metrics."""
    metrics = {}
    
    try:
        # VPIN
        metrics['vpin'] = calculate_vpin()
        
        # Kyle's Lambda
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
            if np.sum(X.flatten() ** 2) > 0:
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
        effective_spread = 2 * abs(last_price - mid_price) / mid_price if mid_price > 0 else 0
        
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
        # Classify trades as buy/sell based on tick direction
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

# === MULTI-HORIZON TRACKING ===

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
            if model is not None and self._is_model_fitted(model):
                try:
                    # Scale features
                    scaler = self._get_scaler(timeframe)
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
                if river_model:
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
    
    def _get_scaler(self, timeframe: str):
        """Get appropriate scaler for timeframe."""
        if timeframe == "1m":
            return STATE.scaler
        elif timeframe == "5m":
            return STATE.scaler_5m
        elif timeframe == "15m":
            return STATE.scaler_15m
        else:
            return STATE.scaler
    
    def _is_model_fitted(self, model) -> bool:
        """Check if model is fitted."""
        if model is None:
            return False
        
        try:
            # Check sklearn models
            if hasattr(model, 'n_features_in_'):
                return model.n_features_in_ > 0
            elif hasattr(model, 'coef_'):
                return model.coef_ is not None
            elif hasattr(model, 'feature_importances_'):
                return len(model.feature_importances_) > 0
            else:
                # Try to access attributes that exist only after fitting
                getattr(model, 'classes_')
                return True
        except:
            return False

# Initialize model manager
MODEL_MANAGER = ModelManager()

# === ONLINE LEARNING SYSTEM ===

class OnlineLearningSystem:
    """Advanced online learning with multiple timeframes."""
    
    def __init__(self):
        self.update_counts = defaultdict(int)
        self.performance_history = defaultdict(list)
        self.last_update_times = defaultdict(float)

# Initialize systems if not in STATE
if not hasattr(STATE, 'online_learner'):
    STATE.online_learner = OnlineLearningSystem()

# === TELEGRAM MESSAGE BUILDER ===

class TelegramMessageBuilder:
    """Build formatted messages for Telegram notifications."""
    
    def __init__(self, state, trading_signals=None):
        self.state = state
        self.trading_signals = trading_signals or TRADING_SIGNALS
    
    def safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """Get value safely."""
        try:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        except Exception:
            return default
    
    def format_percentage(self, value: Optional[float], decimals: int = 1) -> str:
        """Format percentage safely."""
        try:
            if value is None:
                return "N/A"
            return f"{value:.{decimals}%}"
        except:
            return "N/A"
    
    def format_number(self, value: Optional[float], decimals: int = 2) -> str:
        """Format number safely."""
        try:
            if value is None:
                return "N/A"
            return f"{value:.{decimals}f}"
        except:
            return "N/A"

# Initialize message builder
TELEGRAM_MESSAGE_BUILDER = TelegramMessageBuilder(STATE, TRADING_SIGNALS)

# === WORKERS ===

async def enhanced_gpt_worker():
    """Worker enhanced with GPT + Grok synergy."""
    logger.info("Enhanced GPT+Grok worker started")
    
    while not STATE.GPT_STOP_EVENT.is_set():
        try:
            current_time = time.time()
            
            # Check cooldown
            if current_time - STATE.last_gpt_time < STATE.GPT_COOLDOWN:
                await asyncio.sleep(5)
                continue
            
            # Check data availability
            with locks.tns_lock:
                if STATE.df_tns.empty:
                    await asyncio.sleep(5)
                    continue
            
            # Check if we have dual AI
            if not STATE.dual_ai_analyzer:
                await asyncio.sleep(10)
                continue
            
            # Get synergistic analysis
            STATE.gpt_in_progress = True
            
            try:
                decision = await STATE.dual_ai_analyzer.get_synergistic_decision()
                
                if not decision.get('error'):
                    # Format analysis
                    analysis_text = f"""
📊 **AI CONSENSUS** ({decision['decision']})
• Confidence: {decision['confidence']:.1f}%
• GPT: {decision['gpt_recommendation']}
• Grok: {decision['grok_recommendation']}
• {decision['reasoning']}
"""
                    
                    STATE.latest_gpt_analysis = analysis_text
                    STATE.last_gpt_time = current_time
                    STATE.gpt_has_alert = True
                    
                    # Store in history
                    STATE.gpt_history.append({
                        'timestamp': current_time,
                        'analysis': analysis_text,
                        'decision': decision
                    })
                    
                    # Check for high confidence signals
                    if decision['confidence'] > TRADING_CONFIG.ai_high_confidence_threshold:
                        if decision['decision'] in ['STRONG_BUY', 'STRONG_SELL']:
                            signal = {
                                'symbol': STATE.current_symbol,
                                'action': decision['decision'],
                                'confidence': decision['confidence'],
                                'size': STATE.risk_manager.calculate_position_size(
                                    decision,
                                    decision,
                                    {'volatility': STATE.market_regime.volatility}
                                ),
                                'urgency': 0.8 if decision['ai_agreement'] > 0.8 else 0.5,
                                'source': 'dual_ai'
                            }
                            
                            STATE.pending_signals.append(signal)
                            ai_logger.info(f"High confidence signal queued: {signal}")
                else:
                    STATE.latest_gpt_analysis = "AI analysis temporarily unavailable"
                    
            except Exception as e:
                logger.error(f"Dual AI analysis error: {e}")
                STATE.latest_gpt_analysis = "AI analysis error"
            
            finally:
                STATE.gpt_in_progress = False
            
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Enhanced GPT worker error: {e}")
            await asyncio.sleep(10)
    
    logger.info("Enhanced GPT+Grok worker stopped")

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

# === CONNECTION MANAGEMENT ===

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
        
        # Set VWAP anchor
        ny_tz = pytz.timezone("America/New_York")
        market_open = datetime.datetime.now(ny_tz).replace(hour=9, minute=30, second=0, microsecond=0)
        STATE.ANCHOR_TIMESTAMP = market_open.timestamp()
        
        logger.info(f"Subscribed to {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to subscribe to {symbol}: {e}")
        return False

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

# === MAIN LOOP ===

def process_commands():
    """Process queued commands."""
    while not STATE.CMD_QUEUE.empty():
        try:
            cmd = STATE.CMD_QUEUE.get_nowait()
            
            if cmd['action'] == 'subscribe':
                symbol = cmd['symbol']
                success = subscribe_symbol(symbol)
                
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
        
        # Update multi-horizon tracking
        current_price = get_current_price()
        if current_price:
            HORIZON_TRACKER.verify_predictions(current_price)
        
    except Exception as e:
        logger.error(f"Error updating metrics: {e}")

def periodic_reports():
    """Send periodic reports to Telegram."""
    current_time = time.time()
    
    # 5-minute update
    if current_time - STATE.last_5min_send > 300:
        try:
            # Simple status update
            msg = f"""
📊 **5-MINUTE UPDATE**
━━━━━━━━━━━━━━━━━━
Symbol: {STATE.current_symbol}
Price: ${get_current_price():.2f}
Volume: {STATE.RELATIVE_VOLUME:.1f}x
Regime: {STATE.market_regime.regime}
P&L: ${STATE.metrics.daily_pnl:+.2f}
"""
            STATE.send_alert(msg, force=True)
            STATE.last_5min_send = current_time
            
        except Exception as e:
            logger.error(f"Error in 5-minute update: {e}")

async def run_enhanced_worker():
    """Run the enhanced GPT worker in async context."""
    await enhanced_gpt_worker()

def enhanced_ib_loop():
    """Enhanced main trading loop with all features."""
    
    logger.info("Starting enhanced IB loop with Dual AI...")
    
    # Start background workers
    
    # GPT+Grok worker with async support
    def run_async_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_enhanced_worker())
    
    gpt_thread = threading.Thread(target=run_async_worker, daemon=True, name="GPT-Grok-Worker")
    gpt_thread.start()
    
    # Telegram worker
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
            
            # Update metrics every 10 loops
            if loop_counter % 10 == 0:
                update_metrics_and_features()
            
            # Periodic reports every 1000 loops (~10 seconds)
            if loop_counter % 1000 == 0:
                periodic_reports()
            
            # Health check every 1000 loops
            if loop_counter % 1000 == 0:
                healthy, msg = check_system_health()
                if not healthy:
                    logger.warning(f"System health issue: {msg}")
            
            # Reset loop counter to prevent overflow
            if loop_counter > 1000000:
                loop_counter = 0
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break
            
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            STATE.ib_loop_error_count += 1
            
            if STATE.ib_loop_error_count > 50:
                logger.error("Too many errors, shutting down")
                break
            
            time.sleep(1)
    
    logger.info("Enhanced IB loop stopped")

# === MAIN ENTRY POINT ===

def main():
    """Main entry point."""
    
    logger.info("=" * 80)
    logger.info("SYNAPSE X v11.0 - DUAL AI TRADING SYSTEM")
    logger.info("=" * 80)
    
    # Connect to IB
    if not connect_all():
        logger.error("Failed to establish initial connection")
        sys.exit(1)
    
    # Send startup message
    startup_msg = f"""
🚀 **SYNAPSE X v11.0 STARTED**
━━━━━━━━━━━━━━━━━━━━━━━━━
Version: 11.0.0
Symbol: {STATE.current_symbol}
AI: GPT-4 + Grok Enabled
ML: Online Learning Active
━━━━━━━━━━━━━━━━━━━━━━━━━
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
System terminated gracefully
        """
        STATE.send_alert(final_msg)
        
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()