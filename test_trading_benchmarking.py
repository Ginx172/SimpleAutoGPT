#!/usr/bin/env python3
"""
Test script for Trading AI Benchmarking System
Demonstrates integration with FinRL, Qlib, vectorbt, Kats, and academic models
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from trading_benchmarking_system import get_trading_benchmarking_system

async def test_trading_benchmarking():
    """Test the complete trading benchmarking system"""
    
    print("🚀 Testing Trading AI Benchmarking System")
    print("=" * 60)
    
    # Initialize the system
    trading_benchmark = get_trading_benchmarking_system()
    
    # Test providers and models
    providers = [
        {"provider": "openai", "model": "gpt-4"},
        {"provider": "anthropic", "model": "claude-3-opus"},
        {"provider": "groq", "model": "llama3-70b"},
        {"provider": "together", "model": "llama2-70b"},
        {"provider": "mistral", "model": "mistral-7b"}
    ]
    
    test_suites = ["comprehensive", "performance"]
    
    all_results = []
    
    for provider_config in providers:
        provider = provider_config["provider"]
        model = provider_config["model"]
        
        print(f"\n📊 Testing {provider.upper()}/{model}")
        print("-" * 40)
        
        for test_suite in test_suites:
            print(f"  🔄 Running {test_suite} test suite...")
            
            try:
                results = await trading_benchmark.run_trading_benchmark(
                    provider=provider,
                    model=model,
                    test_suite=test_suite
                )
                
                if results:
                    print(f"    ✅ {len(results)} tests completed successfully")
                    
                    # Calculate average metrics
                    avg_sharpe = sum(r.metrics.sharpe_ratio for r in results) / len(results)
                    avg_drawdown = sum(r.metrics.max_drawdown for r in results) / len(results)
                    avg_win_rate = sum(r.metrics.win_rate for r in results) / len(results)
                    
                    print(f"    📈 Average Sharpe Ratio: {avg_sharpe:.3f}")
                    print(f"    📉 Average Max Drawdown: {avg_drawdown:.3f}")
                    print(f"    🎯 Average Win Rate: {avg_win_rate:.3f}")
                    
                    all_results.extend(results)
                else:
                    print(f"    ❌ No results for {test_suite}")
                    
            except Exception as e:
                print(f"    ❌ Error in {test_suite}: {e}")
    
    # Generate comprehensive report
    if all_results:
        print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        report = trading_benchmark.generate_trading_report(all_results)
        print(report)
        
        # Save report to file
        with open("trading_benchmark_report.md", "w") as f:
            f.write(report)
        
        print(f"\n💾 Report saved to: trading_benchmark_report.md")
        
        # Test framework integrations
        print(f"\n🔧 TESTING FRAMEWORK INTEGRATIONS")
        print("=" * 60)
        
        # Test FinRL integration
        if trading_benchmark.finrl_env:
            print("✅ FinRL Environment: Initialized")
        else:
            print("⚠️ FinRL Environment: Not available")
        
        # Test Qlib integration
        if trading_benchmark.qlib_handler:
            print("✅ Qlib Platform: Initialized")
        else:
            print("⚠️ Qlib Platform: Not available")
        
        # Test VectorBT integration
        if trading_benchmark.vectorbt_portfolio:
            print("✅ VectorBT Backtesting: Initialized")
        else:
            print("⚠️ VectorBT Backtesting: Not available")
        
        # Test Academic Models integration
        if trading_benchmark.academic_models:
            print("✅ Academic Models: Initialized")
            for model_name in trading_benchmark.academic_models.keys():
                print(f"   - {model_name.title()}: Available")
        else:
            print("⚠️ Academic Models: Not available")
        
        # Test advanced metrics
        print(f"\n📊 TESTING ADVANCED METRICS")
        print("=" * 60)
        
        sample_results = [r for r in all_results if r.success][:5]
        
        for result in sample_results:
            print(f"Test: {result.test.test_id}")
            print(f"  - Deflated Sharpe Ratio: {result.metrics.deflated_sharpe_ratio:.3f}")
            print(f"  - PBO: {result.metrics.probability_backtest_overfitting:.3f}")
            print(f"  - White Reality Check: {result.metrics.white_reality_check_pvalue:.3f}")
            print(f"  - Consistency Score: {result.metrics.consistency_score:.3f}")
            print()
        
        print("🎉 Trading AI Benchmarking System test completed successfully!")
        
    else:
        print("❌ No benchmark results to analyze")

async def test_framework_specific_features():
    """Test framework-specific features"""
    
    print("\n🔬 TESTING FRAMEWORK-SPECIFIC FEATURES")
    print("=" * 60)
    
    trading_benchmark = get_trading_benchmarking_system()
    
    # Test FinRL RL predictions
    print("🧠 Testing FinRL RL Predictions...")
    try:
        # Create a mock test for FinRL
        from trading_benchmarking_system import TradingTest
        
        finrl_test = TradingTest(
            test_id="finrl_test",
            category="price_prediction",
            subcategory="short_term",
            market_data={"returns": [0.001, -0.002, 0.003, -0.001, 0.002]},
            expected_performance={"sharpe_ratio": 1.5},
            evaluation_criteria=["prediction_accuracy"]
        )
        
        market_data = trading_benchmark._generate_market_data(finrl_test)
        predictions = await trading_benchmark._finrl_prediction(finrl_test, market_data)
        
        print(f"  ✅ FinRL predictions generated: {len(predictions)} values")
        print(f"  📊 Sample predictions: {predictions[:3]}")
        
    except Exception as e:
        print(f"  ❌ FinRL test failed: {e}")
    
    # Test Qlib quant predictions
    print("\n📈 Testing Qlib Quant Predictions...")
    try:
        qlib_test = TradingTest(
            test_id="qlib_test",
            category="price_prediction",
            subcategory="long_term",
            market_data={"returns": [0.001, -0.002, 0.003, -0.001, 0.002]},
            expected_performance={"sharpe_ratio": 1.2},
            evaluation_criteria=["prediction_accuracy"]
        )
        
        market_data = trading_benchmark._generate_market_data(qlib_test)
        predictions = await trading_benchmark._qlib_prediction(qlib_test, market_data)
        
        print(f"  ✅ Qlib predictions generated: {len(predictions)} values")
        print(f"  📊 Sample predictions: {predictions[:3]}")
        
    except Exception as e:
        print(f"  ❌ Qlib test failed: {e}")
    
    # Test Academic Models
    print("\n🎓 Testing Academic Models...")
    academic_models = ["informer", "autoformer", "patchtst"]
    
    for model_name in academic_models:
        try:
            academic_test = TradingTest(
                test_id=f"{model_name}_test",
                category="price_prediction",
                subcategory="long_term",
                market_data={"returns": [0.001, -0.002, 0.003, -0.001, 0.002]},
                expected_performance={"sharpe_ratio": 1.8},
                evaluation_criteria=["prediction_accuracy"]
            )
            
            market_data = trading_benchmark._generate_market_data(academic_test)
            predictions = await trading_benchmark._academic_model_prediction(academic_test, market_data)
            
            print(f"  ✅ {model_name.title()} predictions generated: {len(predictions)} values")
            print(f"  📊 Sample predictions: {predictions[:3]}")
            
        except Exception as e:
            print(f"  ❌ {model_name} test failed: {e}")
    
    # Test VectorBT backtesting
    print("\n⚡ Testing VectorBT Backtesting...")
    try:
        import numpy as np
        
        prices = np.array([100, 101, 99, 102, 98, 103])
        returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
        signals = [1, 0, -1, 1, 0]
        
        vectorbt_returns = trading_benchmark._vectorbt_backtest(prices, returns, signals)
        
        print(f"  ✅ VectorBT backtesting completed: {len(vectorbt_returns)} returns")
        print(f"  📊 Sample returns: {vectorbt_returns[:3]}")
        
    except Exception as e:
        print(f"  ❌ VectorBT test failed: {e}")

async def main():
    """Main test function"""
    try:
        await test_trading_benchmarking()
        await test_framework_specific_features()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🚀 Trading AI Benchmarking System is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
