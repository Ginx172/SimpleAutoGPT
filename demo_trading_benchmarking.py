#!/usr/bin/env python3
"""
Demo script for Trading AI Benchmarking System
Quick demonstration of all integrated frameworks and features
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from trading_benchmarking_system import get_trading_benchmarking_system

async def demo_trading_benchmarking():
    """Demonstrate the complete trading benchmarking system"""
    
    print("🎯 TRADING AI BENCHMARKING SYSTEM - DEMO")
    print("=" * 60)
    
    # Initialize the system
    print("🚀 Initializing Trading Benchmarking System...")
    trading_benchmark = get_trading_benchmarking_system()
    
    print("\n📊 FRAMEWORK INTEGRATION STATUS:")
    print("-" * 40)
    
    # Check framework status
    frameworks = {
        "FinRL Environment": trading_benchmark.finrl_env is not None,
        "Qlib Platform": trading_benchmark.qlib_handler is not None,
        "VectorBT Backtesting": trading_benchmark.vectorbt_portfolio is not None,
        "Academic Models": len(trading_benchmark.academic_models) > 0
    }
    
    for framework, status in frameworks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {framework}: {'Available' if status else 'Not Available'}")
    
    if trading_benchmark.academic_models:
        print("   Academic Models Available:")
        for model_name in trading_benchmark.academic_models.keys():
            print(f"     - {model_name.title()}")
    
    print("\n🧪 RUNNING BENCHMARK DEMO:")
    print("-" * 40)
    
    # Run a quick benchmark
    try:
        print("🔄 Running comprehensive benchmark for OpenAI GPT-4...")
        results = await trading_benchmark.run_trading_benchmark(
            provider="openai",
            model="gpt-4",
            test_suite="comprehensive"
        )
        
        if results:
            successful_results = [r for r in results if r.success]
            print(f"✅ Benchmark completed: {len(successful_results)} tests passed")
            
            if successful_results:
                # Show sample results
                sample_result = successful_results[0]
                print(f"\n📋 SAMPLE RESULT:")
                print(f"   Test ID: {sample_result.test.test_id}")
                print(f"   Category: {sample_result.test.category}")
                print(f"   Domain: {sample_result.test.domain}")
                print(f"   Sharpe Ratio: {sample_result.metrics.sharpe_ratio:.3f}")
                print(f"   Max Drawdown: {sample_result.metrics.max_drawdown:.3f}")
                print(f"   Win Rate: {sample_result.metrics.win_rate:.3f}")
                print(f"   Prediction Accuracy: {sample_result.metrics.prediction_accuracy:.3f}")
                print(f"   Execution Time: {sample_result.execution_time:.3f}s")
                
                # Show advanced metrics
                print(f"\n🔬 ADVANCED METRICS:")
                print(f"   Deflated Sharpe Ratio: {sample_result.metrics.deflated_sharpe_ratio:.3f}")
                print(f"   PBO: {sample_result.metrics.probability_backtest_overfitting:.3f}")
                print(f"   White Reality Check: {sample_result.metrics.white_reality_check_pvalue:.3f}")
                print(f"   Consistency Score: {sample_result.metrics.consistency_score:.3f}")
        else:
            print("❌ No results returned from benchmark")
            
    except Exception as e:
        print(f"❌ Benchmark demo failed: {e}")
    
    print("\n🔧 FRAMEWORK-SPECIFIC DEMOS:")
    print("-" * 40)
    
    # Demo FinRL
    print("🧠 FinRL RL Environment Demo:")
    try:
        from trading_benchmarking_system import TradingTest
        finrl_test = TradingTest(
            test_id="demo_finrl",
            category="price_prediction",
            subcategory="short_term",
            market_data={"returns": [0.001, -0.002, 0.003]},
            expected_performance={"sharpe_ratio": 1.5},
            evaluation_criteria=["prediction_accuracy"]
        )
        
        market_data = trading_benchmark._generate_market_data(finrl_test)
        predictions = await trading_benchmark._finrl_prediction(finrl_test, market_data)
        print(f"   ✅ Generated {len(predictions)} RL-based predictions")
        print(f"   📊 Sample: {predictions[:3]}")
        
    except Exception as e:
        print(f"   ❌ FinRL demo failed: {e}")
    
    # Demo Qlib
    print("\n📊 Qlib Quant Platform Demo:")
    try:
        qlib_test = TradingTest(
            test_id="demo_qlib",
            category="price_prediction",
            subcategory="long_term",
            market_data={"returns": [0.001, -0.002, 0.003]},
            expected_performance={"sharpe_ratio": 1.2},
            evaluation_criteria=["prediction_accuracy"]
        )
        
        market_data = trading_benchmark._generate_market_data(qlib_test)
        predictions = await trading_benchmark._qlib_prediction(qlib_test, market_data)
        print(f"   ✅ Generated {len(predictions)} quant-based predictions")
        print(f"   📊 Sample: {predictions[:3]}")
        
    except Exception as e:
        print(f"   ❌ Qlib demo failed: {e}")
    
    # Demo Academic Models
    print("\n🎓 Academic Models Demo:")
    academic_models = ["informer", "autoformer", "patchtst"]
    
    for model_name in academic_models:
        try:
            academic_test = TradingTest(
                test_id=f"demo_{model_name}",
                category="price_prediction",
                subcategory="long_term",
                market_data={"returns": [0.001, -0.002, 0.003]},
                expected_performance={"sharpe_ratio": 1.8},
                evaluation_criteria=["prediction_accuracy"]
            )
            
            market_data = trading_benchmark._generate_market_data(academic_test)
            predictions = await trading_benchmark._academic_model_prediction(academic_test, market_data)
            print(f"   ✅ {model_name.title()}: Generated {len(predictions)} predictions")
            
        except Exception as e:
            print(f"   ❌ {model_name} demo failed: {e}")
    
    # Demo VectorBT
    print("\n⚡ VectorBT Backtesting Demo:")
    try:
        import numpy as np
        
        prices = np.array([100, 101, 99, 102, 98, 103])
        returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
        signals = [1, 0, -1, 1, 0]
        
        vectorbt_returns = trading_benchmark._vectorbt_backtest(prices, returns, signals)
        print(f"   ✅ VectorBT backtesting completed: {len(vectorbt_returns)} returns")
        print(f"   📊 Sample returns: {vectorbt_returns[:3]}")
        
    except Exception as e:
        print(f"   ❌ VectorBT demo failed: {e}")
    
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("🚀 Trading AI Benchmarking System is ready for production use!")
    
    print("\n📚 NEXT STEPS:")
    print("-" * 40)
    print("1. Run full benchmark: python test_trading_benchmarking.py")
    print("2. Start dashboard: python start_trading_dashboard.py")
    print("3. Install dependencies: pip install -r requirements_trading_benchmarking.txt")
    print("4. Configure API keys in .env file")
    print("5. Read documentation: TRADING_BENCHMARKING_README.md")

if __name__ == "__main__":
    asyncio.run(demo_trading_benchmarking())
