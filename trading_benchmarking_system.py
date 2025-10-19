"""
Trading AI Benchmarking System - Enterprise Edition
Specialized benchmarking for AI models in trading and financial markets
"""

import os
import json
import time
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import statistics
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingMetrics:
    """Comprehensive metrics for trading AI model evaluation"""
    
    # Performance Metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Risk Metrics
    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    expected_shortfall: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    
    # Trading Metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    trading_frequency: float = 0.0
    average_holding_period: float = 0.0
    turnover_rate: float = 0.0
    
    # AI-Specific Metrics
    prediction_accuracy: float = 0.0
    directional_accuracy: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error
    rmse: float = 0.0  # Root Mean Square Error
    information_ratio: float = 0.0
    
    # Advanced Metrics
    deflated_sharpe_ratio: float = 0.0
    probability_backtest_overfitting: float = 0.0
    white_reality_check_pvalue: float = 0.0
    spa_test_statistic: float = 0.0
    consistency_score: float = 0.0

@dataclass
class TradingTest:
    """Trading-specific test case"""
    test_id: str
    category: str
    subcategory: str
    market_data: Dict[str, Any]
    expected_performance: Dict[str, float]
    evaluation_criteria: List[str]
    difficulty_level: str = "medium"
    domain: str = "equity"
    weight: float = 1.0

@dataclass
class TradingBenchmarkResult:
    """Complete trading benchmark result"""
    provider: str
    model: str
    test: TradingTest
    predictions: List[float]
    actual_returns: List[float]
    trading_signals: List[int]  # 1 for buy, -1 for sell, 0 for hold
    metrics: TradingMetrics
    timestamp: datetime
    execution_time: float
    success: bool = True
    error_message: Optional[str] = None

class TradingBenchmarkingSystem:
    """
    Advanced Trading AI Benchmarking System
    Specialized for financial markets and trading strategies
    """
    
    def __init__(self):
        self.logger = logger
        self.benchmark_results = []
        self.trading_test_suites = {}
        self.market_data_cache = {}
        self.load_trading_test_suites()
        self.load_evaluation_frameworks()
        
        # Trading-specific configurations
        self.trading_config = {
            "risk_free_rate": 0.02,  # 2% annual risk-free rate
            "benchmark_return": 0.08,  # 8% annual benchmark return
            "transaction_cost": 0.001,  # 0.1% transaction cost
            "max_position_size": 0.1,  # 10% max position size
            "lookback_period": 252,  # 1 year of trading days
            "evaluation_periods": [30, 90, 180, 365]  # Evaluation periods in days
        }
        
        self.logger.info("🚀 Trading Benchmarking System initialized")
    
    def load_trading_test_suites(self):
        """Load comprehensive trading test suites"""
        
        self.trading_test_suites = {
            "comprehensive": [
                # Price Prediction Tests
                TradingTest(
                    test_id="price_pred_001",
                    category="price_prediction",
                    subcategory="short_term",
                    market_data={
                        "asset": "AAPL",
                        "period": "1D",
                        "features": ["price", "volume", "rsi", "macd", "bollinger"],
                        "lookback": 60
                    },
                    expected_performance={
                        "sharpe_ratio": 1.5,
                        "max_drawdown": 0.15,
                        "win_rate": 0.55
                    },
                    evaluation_criteria=["prediction_accuracy", "directional_accuracy", "sharpe_ratio"],
                    difficulty_level="medium",
                    domain="equity",
                    weight=1.5
                ),
                
                TradingTest(
                    test_id="price_pred_002",
                    category="price_prediction",
                    subcategory="long_term",
                    market_data={
                        "asset": "SPY",
                        "period": "1W",
                        "features": ["price", "volume", "sector_rotation", "macro_indicators"],
                        "lookback": 52
                    },
                    expected_performance={
                        "sharpe_ratio": 1.2,
                        "max_drawdown": 0.20,
                        "win_rate": 0.52
                    },
                    evaluation_criteria=["prediction_accuracy", "consistency", "risk_metrics"],
                    difficulty_level="hard",
                    domain="equity",
                    weight=2.0
                ),
                
                # Volatility Prediction Tests
                TradingTest(
                    test_id="vol_pred_001",
                    category="volatility_prediction",
                    subcategory="garch_modeling",
                    market_data={
                        "asset": "VIX",
                        "period": "1D",
                        "features": ["volatility", "vix", "fear_greed", "market_sentiment"],
                        "lookback": 100
                    },
                    expected_performance={
                        "sharpe_ratio": 1.8,
                        "max_drawdown": 0.12,
                        "win_rate": 0.58
                    },
                    evaluation_criteria=["volatility_accuracy", "risk_management", "profit_factor"],
                    difficulty_level="expert",
                    domain="volatility",
                    weight=2.5
                ),
                
                # Sentiment Analysis Tests
                TradingTest(
                    test_id="sentiment_001",
                    category="sentiment_analysis",
                    subcategory="news_impact",
                    market_data={
                        "asset": "TSLA",
                        "period": "1H",
                        "features": ["news_sentiment", "social_media", "analyst_ratings"],
                        "lookback": 168  # 1 week of hourly data
                    },
                    expected_performance={
                        "sharpe_ratio": 1.3,
                        "max_drawdown": 0.18,
                        "win_rate": 0.53
                    },
                    evaluation_criteria=["sentiment_accuracy", "event_detection", "market_timing"],
                    difficulty_level="hard",
                    domain="sentiment",
                    weight=1.8
                ),
                
                # Portfolio Optimization Tests
                TradingTest(
                    test_id="portfolio_001",
                    category="portfolio_optimization",
                    subcategory="multi_asset",
                    market_data={
                        "assets": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                        "period": "1D",
                        "features": ["returns", "correlation", "volatility", "momentum"],
                        "lookback": 252
                    },
                    expected_performance={
                        "sharpe_ratio": 1.6,
                        "max_drawdown": 0.14,
                        "win_rate": 0.56
                    },
                    evaluation_criteria=["diversification", "risk_adjusted_returns", "rebalancing_efficiency"],
                    difficulty_level="expert",
                    domain="portfolio",
                    weight=2.2
                ),
                
                # High-Frequency Trading Tests
                TradingTest(
                    test_id="hft_001",
                    category="high_frequency",
                    subcategory="market_making",
                    market_data={
                        "asset": "BTC-USD",
                        "period": "1M",
                        "features": ["bid_ask", "order_book", "trade_flow", "microstructure"],
                        "lookback": 1440  # 1 day of minute data
                    },
                    expected_performance={
                        "sharpe_ratio": 2.0,
                        "max_drawdown": 0.08,
                        "win_rate": 0.62
                    },
                    evaluation_criteria=["latency", "execution_quality", "market_impact"],
                    difficulty_level="expert",
                    domain="crypto",
                    weight=3.0
                )
            ],
            
            "performance": [
                TradingTest(
                    test_id="perf_001",
                    category="performance",
                    subcategory="speed_test",
                    market_data={
                        "asset": "SPY",
                        "period": "1M",
                        "features": ["price"],
                        "lookback": 1000
                    },
                    expected_performance={
                        "response_time": 0.1,
                        "throughput": 1000
                    },
                    evaluation_criteria=["speed", "throughput", "latency"],
                    difficulty_level="easy",
                    domain="performance",
                    weight=1.0
                )
            ],
            
            "domain_specific": {
                "equity": [
                    TradingTest(
                        test_id="equity_001",
                        category="equity_trading",
                        subcategory="sector_rotation",
                        market_data={
                            "sectors": ["technology", "healthcare", "finance", "energy"],
                            "period": "1D",
                            "features": ["sector_performance", "rotation_signals", "macro_factors"],
                            "lookback": 252
                        },
                        expected_performance={
                            "sharpe_ratio": 1.4,
                            "max_drawdown": 0.16,
                            "win_rate": 0.54
                        },
                        evaluation_criteria=["sector_timing", "rotation_accuracy", "risk_management"],
                        difficulty_level="hard",
                        domain="equity",
                        weight=1.8
                    )
                ],
                "crypto": [
                    TradingTest(
                        test_id="crypto_001",
                        category="crypto_trading",
                        subcategory="momentum_strategy",
                        market_data={
                            "assets": ["BTC", "ETH", "ADA", "DOT"],
                            "period": "4H",
                            "features": ["price", "volume", "sentiment", "on_chain_metrics"],
                            "lookback": 168
                        },
                        expected_performance={
                            "sharpe_ratio": 1.7,
                            "max_drawdown": 0.25,
                            "win_rate": 0.57
                        },
                        evaluation_criteria=["momentum_detection", "crypto_specific_metrics", "volatility_handling"],
                        difficulty_level="hard",
                        domain="crypto",
                        weight=2.0
                    )
                ],
                "forex": [
                    TradingTest(
                        test_id="forex_001",
                        category="forex_trading",
                        subcategory="carry_trade",
                        market_data={
                            "pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"],
                            "period": "1D",
                            "features": ["interest_rates", "economic_indicators", "central_bank_policy"],
                            "lookback": 365
                        },
                        expected_performance={
                            "sharpe_ratio": 1.2,
                            "max_drawdown": 0.20,
                            "win_rate": 0.51
                        },
                        evaluation_criteria=["interest_rate_differentials", "economic_calendar", "risk_management"],
                        difficulty_level="expert",
                        domain="forex",
                        weight=2.3
                    )
                ]
            }
        }
        
        self.logger.info(f"📚 Loaded {len(self.trading_test_suites)} trading test suites")
    
    def load_evaluation_frameworks(self):
        """Load trading-specific evaluation frameworks"""
        
        self.evaluation_frameworks = {
            "performance": {
                "sharpe_ratio": self._calculate_sharpe_ratio,
                "max_drawdown": self._calculate_max_drawdown,
                "win_rate": self._calculate_win_rate,
                "profit_factor": self._calculate_profit_factor,
                "calmar_ratio": self._calculate_calmar_ratio,
                "sortino_ratio": self._calculate_sortino_ratio
            },
            "risk": {
                "var_95": self._calculate_var_95,
                "var_99": self._calculate_var_99,
                "expected_shortfall": self._calculate_expected_shortfall,
                "volatility": self._calculate_volatility,
                "beta": self._calculate_beta
            },
            "trading": {
                "total_return": self._calculate_total_return,
                "annualized_return": self._calculate_annualized_return,
                "trading_frequency": self._calculate_trading_frequency,
                "average_holding_period": self._calculate_average_holding_period,
                "turnover_rate": self._calculate_turnover_rate
            },
            "ai_specific": {
                "prediction_accuracy": self._calculate_prediction_accuracy,
                "directional_accuracy": self._calculate_directional_accuracy,
                "mape": self._calculate_mape,
                "rmse": self._calculate_rmse,
                "information_ratio": self._calculate_information_ratio
            },
            "advanced": {
                "deflated_sharpe_ratio": self._calculate_deflated_sharpe_ratio,
                "probability_backtest_overfitting": self._calculate_pbo,
                "white_reality_check": self._calculate_white_reality_check,
                "spa_test": self._calculate_spa_test,
                "consistency_score": self._calculate_consistency_score
            }
        }
    
    async def run_trading_benchmark(self, provider: str, model: str, 
                                  test_suite: str = "comprehensive") -> List[TradingBenchmarkResult]:
        """
        Run comprehensive trading benchmark for a model
        """
        self.logger.info(f"🚀 Starting trading benchmark for {provider}/{model}")
        
        results = []
        tests = self.trading_test_suites.get(test_suite, [])
        
        if not tests:
            self.logger.error(f"❌ No tests found for suite: {test_suite}")
            return results
        
        # Run tests
        for test in tests:
            try:
                result = await self._run_single_trading_test(provider, model, test)
                if result:
                    results.append(result)
                    self.benchmark_results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Test {test.test_id} failed: {e}")
        
        self.logger.info(f"✅ Completed trading benchmark: {len(results)} tests passed")
        return results
    
    async def _run_single_trading_test(self, provider: str, model: str, 
                                     test: TradingTest) -> TradingBenchmarkResult:
        """Run a single trading test"""
        
        start_time = time.time()
        
        try:
            # Simulate market data generation
            market_data = self._generate_market_data(test)
            
            # Simulate AI model predictions
            predictions = await self._simulate_ai_predictions(provider, model, test, market_data)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(predictions, test)
            
            # Calculate actual returns
            actual_returns = self._calculate_actual_returns(market_data, trading_signals)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_trading_metrics(predictions, actual_returns, trading_signals)
            
            execution_time = time.time() - start_time
            
            return TradingBenchmarkResult(
                provider=provider,
                model=model,
                test=test,
                predictions=predictions,
                actual_returns=actual_returns,
                trading_signals=trading_signals,
                metrics=metrics,
                timestamp=datetime.now(),
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"❌ Trading test {test.test_id} failed: {e}")
            
            return TradingBenchmarkResult(
                provider=provider,
                model=model,
                test=test,
                predictions=[],
                actual_returns=[],
                trading_signals=[],
                metrics=TradingMetrics(),
                timestamp=datetime.now(),
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_market_data(self, test: TradingTest) -> Dict[str, Any]:
        """Generate simulated market data for testing"""
        
        # This would integrate with real market data APIs
        # For now, generate realistic simulated data
        
        lookback = test.market_data.get("lookback", 252)
        
        # Generate price data
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.0005, 0.02, lookback)  # Daily returns
        prices = 100 * np.cumprod(1 + returns)  # Starting price of 100
        
        # Generate volume data
        volumes = np.random.lognormal(10, 0.5, lookback)
        
        # Generate technical indicators
        rsi = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, lookback))
        macd = np.random.normal(0, 0.5, lookback)
        
        market_data = {
            "prices": prices.tolist(),
            "volumes": volumes.tolist(),
            "returns": returns.tolist(),
            "rsi": rsi.tolist(),
            "macd": macd.tolist(),
            "timestamp": [datetime.now() - timedelta(days=i) for i in range(lookback, 0, -1)]
        }
        
        return market_data
    
    async def _simulate_ai_predictions(self, provider: str, model: str, 
                                     test: TradingTest, market_data: Dict[str, Any]) -> List[float]:
        """Simulate AI model predictions"""
        
        # This would call the actual AI provider
        # For now, generate realistic predictions based on test difficulty
        
        base_accuracy = {
            "easy": 0.7,
            "medium": 0.6,
            "hard": 0.55,
            "expert": 0.5
        }
        
        accuracy = base_accuracy.get(test.difficulty_level, 0.6)
        
        # Generate predictions with controlled accuracy
        np.random.seed(hash(f"{provider}{model}{test.test_id}") % 2**32)
        
        actual_returns = np.array(market_data["returns"])
        noise_level = (1 - accuracy) * np.std(actual_returns)
        
        predictions = actual_returns + np.random.normal(0, noise_level, len(actual_returns))
        
        return predictions.tolist()
    
    def _generate_trading_signals(self, predictions: List[float], test: TradingTest) -> List[int]:
        """Generate trading signals from predictions"""
        
        signals = []
        threshold = 0.001  # 0.1% threshold for trading
        
        for pred in predictions:
            if pred > threshold:
                signals.append(1)  # Buy signal
            elif pred < -threshold:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)  # Hold signal
        
        return signals
    
    def _calculate_actual_returns(self, market_data: Dict[str, Any], 
                                trading_signals: List[int]) -> List[float]:
        """Calculate actual returns from trading signals"""
        
        prices = np.array(market_data["prices"])
        returns = np.array(market_data["returns"])
        
        trading_returns = []
        position = 0
        
        for i, signal in enumerate(trading_signals):
            if i == 0:
                trading_returns.append(0)
                continue
            
            # Update position based on signal
            if signal == 1 and position <= 0:  # Buy signal
                position = 1
            elif signal == -1 and position >= 0:  # Sell signal
                position = -1
            elif signal == 0:  # Hold signal
                pass
            
            # Calculate return for this period
            if position != 0:
                period_return = position * returns[i]
                # Apply transaction costs
                if position != 0:
                    period_return -= self.trading_config["transaction_cost"]
            else:
                period_return = 0
            
            trading_returns.append(period_return)
        
        return trading_returns
    
    def _calculate_trading_metrics(self, predictions: List[float], 
                                 actual_returns: List[float], 
                                 trading_signals: List[int]) -> TradingMetrics:
        """Calculate comprehensive trading metrics"""
        
        metrics = TradingMetrics()
        
        if not actual_returns:
            return metrics
        
        returns_array = np.array(actual_returns)
        
        # Performance Metrics
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns_array)
        metrics.max_drawdown = self._calculate_max_drawdown(returns_array)
        metrics.win_rate = self._calculate_win_rate(returns_array)
        metrics.profit_factor = self._calculate_profit_factor(returns_array)
        metrics.calmar_ratio = self._calculate_calmar_ratio(returns_array)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns_array)
        
        # Risk Metrics
        metrics.var_95 = self._calculate_var_95(returns_array)
        metrics.var_99 = self._calculate_var_99(returns_array)
        metrics.expected_shortfall = self._calculate_expected_shortfall(returns_array)
        metrics.volatility = self._calculate_volatility(returns_array)
        metrics.beta = self._calculate_beta(returns_array)
        
        # Trading Metrics
        metrics.total_return = self._calculate_total_return(returns_array)
        metrics.annualized_return = self._calculate_annualized_return(returns_array)
        metrics.trading_frequency = self._calculate_trading_frequency(trading_signals)
        metrics.average_holding_period = self._calculate_average_holding_period(trading_signals)
        metrics.turnover_rate = self._calculate_turnover_rate(trading_signals)
        
        # AI-Specific Metrics
        if predictions:
            metrics.prediction_accuracy = self._calculate_prediction_accuracy(predictions, actual_returns)
            metrics.directional_accuracy = self._calculate_directional_accuracy(predictions, actual_returns)
            metrics.mape = self._calculate_mape(predictions, actual_returns)
            metrics.rmse = self._calculate_rmse(predictions, actual_returns)
            metrics.information_ratio = self._calculate_information_ratio(returns_array)
        
        # Advanced Metrics
        metrics.deflated_sharpe_ratio = self._calculate_deflated_sharpe_ratio(returns_array)
        metrics.probability_backtest_overfitting = self._calculate_pbo(returns_array)
        metrics.white_reality_check_pvalue = self._calculate_white_reality_check(returns_array)
        metrics.spa_test_statistic = self._calculate_spa_test(returns_array)
        metrics.consistency_score = self._calculate_consistency_score(returns_array)
        
        return metrics
    
    # Performance Metrics Calculations
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return (np.mean(returns) - self.trading_config["risk_free_rate"]/252) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return float('inf') if len(profits) > 0 else 0.0
        
        return abs(np.sum(profits) / np.sum(losses))
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        annual_return = self._calculate_annualized_return(returns)
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return (np.mean(returns) - self.trading_config["risk_free_rate"]/252) / downside_std * np.sqrt(252)
    
    # Risk Metrics Calculations
    def _calculate_var_95(self, returns: np.ndarray) -> float:
        """Calculate Value at Risk 95%"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, 5)
    
    def _calculate_var_99(self, returns: np.ndarray) -> float:
        """Calculate Value at Risk 99%"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, 1)
    
    def _calculate_expected_shortfall(self, returns: np.ndarray) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0.0
        var_95 = self._calculate_var_95(returns)
        return np.mean(returns[returns <= var_95])
    
    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility"""
        if len(returns) == 0:
            return 0.0
        return np.std(returns) * np.sqrt(252)
    
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """Calculate beta (simplified - would need market returns)"""
        # Simplified beta calculation
        return 1.0  # Would need market benchmark data
    
    # Trading Metrics Calculations
    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return"""
        if len(returns) == 0:
            return 0.0
        return np.prod(1 + returns) - 1
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        total_return = self._calculate_total_return(returns)
        years = len(returns) / 252
        return (1 + total_return) ** (1/years) - 1 if years > 0 else 0.0
    
    def _calculate_trading_frequency(self, signals: List[int]) -> float:
        """Calculate trading frequency"""
        if len(signals) == 0:
            return 0.0
        return np.sum(np.abs(np.diff([0] + signals))) / len(signals)
    
    def _calculate_average_holding_period(self, signals: List[int]) -> float:
        """Calculate average holding period"""
        if len(signals) == 0:
            return 0.0
        
        positions = []
        current_position = 0
        holding_start = 0
        
        for i, signal in enumerate(signals):
            if signal != 0 and current_position == 0:  # New position
                current_position = signal
                holding_start = i
            elif signal == 0 and current_position != 0:  # Close position
                positions.append(i - holding_start)
                current_position = 0
        
        return np.mean(positions) if positions else 0.0
    
    def _calculate_turnover_rate(self, signals: List[int]) -> float:
        """Calculate turnover rate"""
        if len(signals) == 0:
            return 0.0
        return np.sum(np.abs(np.diff([0] + signals))) / len(signals)
    
    # AI-Specific Metrics Calculations
    def _calculate_prediction_accuracy(self, predictions: List[float], actual: List[float]) -> float:
        """Calculate prediction accuracy"""
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0.0
        
        # Calculate correlation coefficient
        corr = np.corrcoef(predictions, actual)[0, 1]
        return max(0, corr) if not np.isnan(corr) else 0.0
    
    def _calculate_directional_accuracy(self, predictions: List[float], actual: List[float]) -> float:
        """Calculate directional accuracy"""
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0.0
        
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actual)
        
        return np.sum(pred_direction == actual_direction) / len(predictions)
    
    def _calculate_mape(self, predictions: List[float], actual: List[float]) -> float:
        """Calculate Mean Absolute Percentage Error"""
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0.0
        
        actual_array = np.array(actual)
        pred_array = np.array(predictions)
        
        # Avoid division by zero
        mask = actual_array != 0
        if np.sum(mask) == 0:
            return 0.0
        
        return np.mean(np.abs((actual_array[mask] - pred_array[mask]) / actual_array[mask])) * 100
    
    def _calculate_rmse(self, predictions: List[float], actual: List[float]) -> float:
        """Calculate Root Mean Square Error"""
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0.0
        
        return np.sqrt(np.mean((np.array(predictions) - np.array(actual)) ** 2))
    
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate Information Ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) - self.trading_config["benchmark_return"]/252
        tracking_error = np.std(returns)
        
        return excess_return / tracking_error * np.sqrt(252) if tracking_error > 0 else 0.0
    
    # Advanced Metrics Calculations
    def _calculate_deflated_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Deflated Sharpe Ratio (Bailey & López de Prado)"""
        sharpe = self._calculate_sharpe_ratio(returns)
        n = len(returns)
        
        if n <= 2:
            return 0.0
        
        # Calculate deflated Sharpe ratio
        gamma = 0.5772  # Euler-Mascheroni constant
        dsr = sharpe * np.sqrt((n - 1) / (n - 3)) - gamma
        
        return max(0, dsr)
    
    def _calculate_pbo(self, returns: np.ndarray) -> float:
        """Calculate Probability of Backtest Overfitting (simplified)"""
        if len(returns) < 10:
            return 0.5
        
        # Simplified PBO calculation
        # In practice, this would require multiple backtests
        sharpe = self._calculate_sharpe_ratio(returns)
        
        # Simplified PBO based on Sharpe ratio
        if sharpe > 2.0:
            return 0.1  # Low probability of overfitting
        elif sharpe > 1.5:
            return 0.3
        elif sharpe > 1.0:
            return 0.5
        else:
            return 0.8  # High probability of overfitting
    
    def _calculate_white_reality_check(self, returns: np.ndarray) -> float:
        """Calculate White's Reality Check p-value (simplified)"""
        # Simplified implementation
        sharpe = self._calculate_sharpe_ratio(returns)
        
        # Convert Sharpe ratio to approximate p-value
        if sharpe > 2.0:
            return 0.01
        elif sharpe > 1.5:
            return 0.05
        elif sharpe > 1.0:
            return 0.10
        else:
            return 0.50
    
    def _calculate_spa_test(self, returns: np.ndarray) -> float:
        """Calculate SPA test statistic (simplified)"""
        # Simplified SPA test
        sharpe = self._calculate_sharpe_ratio(returns)
        
        # Convert to SPA test statistic
        return sharpe / np.sqrt(len(returns)) if len(returns) > 0 else 0.0
    
    def _calculate_consistency_score(self, returns: np.ndarray) -> float:
        """Calculate consistency score"""
        if len(returns) < 10:
            return 0.0
        
        # Calculate rolling Sharpe ratios
        window = min(30, len(returns) // 3)
        rolling_sharpes = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            if len(window_returns) > 0:
                sharpe = self._calculate_sharpe_ratio(window_returns)
                rolling_sharpes.append(sharpe)
        
        if len(rolling_sharpes) == 0:
            return 0.0
        
        # Consistency based on variance of rolling Sharpe ratios
        consistency = 1 - np.std(rolling_sharpes) / (np.mean(rolling_sharpes) + 1e-8)
        return max(0, consistency)
    
    def generate_trading_report(self, results: List[TradingBenchmarkResult]) -> str:
        """Generate comprehensive trading benchmark report"""
        
        if not results:
            return "No trading benchmark results available"
        
        report = "# 🚀 TRADING AI BENCHMARK REPORT\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Total Tests:** {len(results)}\n\n"
        
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            # Overall Statistics
            report += "## 📊 OVERALL TRADING PERFORMANCE\n\n"
            
            avg_sharpe = np.mean([r.metrics.sharpe_ratio for r in successful_results])
            avg_max_dd = np.mean([r.metrics.max_drawdown for r in successful_results])
            avg_win_rate = np.mean([r.metrics.win_rate for r in successful_results])
            avg_profit_factor = np.mean([r.metrics.profit_factor for r in successful_results])
            
            report += f"- **Average Sharpe Ratio:** {avg_sharpe:.2f}\n"
            report += f"- **Average Max Drawdown:** {avg_max_dd:.2%}\n"
            report += f"- **Average Win Rate:** {avg_win_rate:.2%}\n"
            report += f"- **Average Profit Factor:** {avg_profit_factor:.2f}\n\n"
            
            # Provider Comparison
            report += "## 🏆 PROVIDER COMPARISON\n\n"
            
            by_provider = {}
            for result in successful_results:
                if result.provider not in by_provider:
                    by_provider[result.provider] = []
                by_provider[result.provider].append(result)
            
            for provider, provider_results in by_provider.items():
                provider_sharpe = np.mean([r.metrics.sharpe_ratio for r in provider_results])
                provider_dd = np.mean([r.metrics.max_drawdown for r in provider_results])
                provider_win_rate = np.mean([r.metrics.win_rate for r in provider_results])
                
                report += f"### {provider.upper()}\n"
                report += f"- **Tests:** {len(provider_results)}\n"
                report += f"- **Average Sharpe Ratio:** {provider_sharpe:.2f}\n"
                report += f"- **Average Max Drawdown:** {provider_dd:.2%}\n"
                report += f"- **Average Win Rate:** {provider_win_rate:.2%}\n\n"
            
            # Category Performance
            report += "## 📈 CATEGORY PERFORMANCE\n\n"
            
            by_category = {}
            for result in successful_results:
                if result.test.category not in by_category:
                    by_category[result.test.category] = []
                by_category[result.test.category].append(result)
            
            for category, category_results in by_category.items():
                category_sharpe = np.mean([r.metrics.sharpe_ratio for r in category_results])
                category_accuracy = np.mean([r.metrics.prediction_accuracy for r in category_results])
                
                report += f"### {category.title()}\n"
                report += f"- **Tests:** {len(category_results)}\n"
                report += f"- **Average Sharpe Ratio:** {category_sharpe:.2f}\n"
                report += f"- **Average Prediction Accuracy:** {category_accuracy:.2%}\n\n"
        
        return report

# Global instance
trading_benchmarking_system = TradingBenchmarkingSystem()

def get_trading_benchmarking_system():
    """Get the global Trading Benchmarking System instance"""
    return trading_benchmarking_system
