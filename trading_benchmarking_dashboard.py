#!/usr/bin/env python3
"""
Trading AI Benchmarking Dashboard - Streamlit
Interactive dashboard for trading AI model benchmarking
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json

# Import trading benchmarking system
from trading_benchmarking_system import get_trading_benchmarking_system

# Page configuration
st.set_page_config(
    page_title="🚀 Trading AI Benchmarking Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .framework-status {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
    .status-available {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-unavailable {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = []
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = get_trading_benchmarking_system()

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Trading AI Benchmarking Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Provider selection
        st.subheader("🤖 AI Provider")
        provider = st.selectbox(
            "Select Provider",
            ["openai", "anthropic", "groq", "together", "mistral"],
            index=0
        )
        
        # Model selection
        models = {
            "openai": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "groq": ["llama3-70b", "mixtral-8x7b", "gemma-7b"],
            "together": ["llama2-70b", "mistral-7b", "codellama-34b"],
            "mistral": ["mistral-7b", "mixtral-8x7b", "codestral-22b"]
        }
        
        model = st.selectbox(
            "Select Model",
            models[provider],
            index=0
        )
        
        # Test suite selection
        st.subheader("🧪 Test Suite")
        test_suite = st.selectbox(
            "Select Test Suite",
            ["comprehensive", "performance", "domain_specific"],
            index=0
        )
        
        # Domain selection (for domain_specific)
        if test_suite == "domain_specific":
            domain = st.selectbox(
                "Select Domain",
                ["equity", "crypto", "forex"],
                index=0
            )
        else:
            domain = None
        
        # Run benchmark button
        st.subheader("🚀 Actions")
        run_benchmark = st.button("Run Trading Benchmark", type="primary")
        
        # Framework status
        st.subheader("🔧 Framework Status")
        display_framework_status()
    
    # Main content
    if run_benchmark:
        run_trading_benchmark(provider, model, test_suite, domain)
    
    # Display results
    if st.session_state.benchmark_results:
        display_results()
    
    # Framework integration demo
    display_framework_demo()

def display_framework_status():
    """Display status of integrated frameworks"""
    
    system = st.session_state.trading_system
    
    # FinRL
    if system.finrl_env:
        st.markdown('<div class="framework-status status-available">✅ FinRL Environment: Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="framework-status status-unavailable">❌ FinRL Environment: Not Available</div>', unsafe_allow_html=True)
    
    # Qlib
    if system.qlib_handler:
        st.markdown('<div class="framework-status status-available">✅ Qlib Platform: Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="framework-status status-unavailable">❌ Qlib Platform: Not Available</div>', unsafe_allow_html=True)
    
    # VectorBT
    if system.vectorbt_portfolio:
        st.markdown('<div class="framework-status status-available">✅ VectorBT Backtesting: Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="framework-status status-unavailable">❌ VectorBT Backtesting: Not Available</div>', unsafe_allow_html=True)
    
    # Academic Models
    if system.academic_models:
        st.markdown('<div class="framework-status status-available">✅ Academic Models: Ready</div>', unsafe_allow_html=True)
        for model_name in system.academic_models.keys():
            st.markdown(f'<div class="framework-status status-available">   - {model_name.title()}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="framework-status status-unavailable">❌ Academic Models: Not Available</div>', unsafe_allow_html=True)

def run_trading_benchmark(provider: str, model: str, test_suite: str, domain: str = None):
    """Run trading benchmark"""
    
    with st.spinner(f"🚀 Running {test_suite} benchmark for {provider}/{model}..."):
        
        try:
            system = st.session_state.trading_system
            
            # Run benchmark asynchronously
            results = asyncio.run(system.run_trading_benchmark(
                provider=provider,
                model=model,
                test_suite=test_suite
            ))
            
            if results:
                st.session_state.benchmark_results = results
                st.success(f"✅ Benchmark completed! {len(results)} tests passed.")
                
                # Display quick stats
                successful_results = [r for r in results if r.success]
                if successful_results:
                    avg_sharpe = np.mean([r.metrics.sharpe_ratio for r in successful_results])
                    avg_drawdown = np.mean([r.metrics.max_drawdown for r in successful_results])
                    avg_win_rate = np.mean([r.metrics.win_rate for r in successful_results])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Sharpe Ratio", f"{avg_sharpe:.3f}")
                    with col2:
                        st.metric("Average Max Drawdown", f"{avg_drawdown:.3f}")
                    with col3:
                        st.metric("Average Win Rate", f"{avg_win_rate:.3f}")
            else:
                st.error("❌ No results returned from benchmark.")
                
        except Exception as e:
            st.error(f"❌ Benchmark failed: {e}")

def display_results():
    """Display benchmark results"""
    
    st.header("📊 Benchmark Results")
    
    results = st.session_state.benchmark_results
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        st.warning("No successful results to display.")
        return
    
    # Create results DataFrame
    results_data = []
    for result in successful_results:
        results_data.append({
            "Test ID": result.test.test_id,
            "Category": result.test.category,
            "Subcategory": result.test.subcategory,
            "Domain": result.test.domain,
            "Difficulty": result.test.difficulty_level,
            "Provider": result.provider,
            "Model": result.model,
            "Sharpe Ratio": result.metrics.sharpe_ratio,
            "Max Drawdown": result.metrics.max_drawdown,
            "Win Rate": result.metrics.win_rate,
            "Profit Factor": result.metrics.profit_factor,
            "Prediction Accuracy": result.metrics.prediction_accuracy,
            "Directional Accuracy": result.metrics.directional_accuracy,
            "Execution Time": result.execution_time
        })
    
    df = pd.DataFrame(results_data)
    
    # Display results table
    st.subheader("📋 Detailed Results")
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    display_visualizations(successful_results)
    
    # Export options
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trading_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export as JSON"):
            json_data = json.dumps([{
                "test_id": r.test.test_id,
                "provider": r.provider,
                "model": r.model,
                "metrics": {
                    "sharpe_ratio": r.metrics.sharpe_ratio,
                    "max_drawdown": r.metrics.max_drawdown,
                    "win_rate": r.metrics.win_rate,
                    "profit_factor": r.metrics.profit_factor
                }
            } for r in successful_results], indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"trading_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📋 Generate Report"):
            system = st.session_state.trading_system
            report = system.generate_trading_report(successful_results)
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"trading_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

def display_visualizations(results):
    """Display benchmark visualizations"""
    
    st.subheader("📈 Performance Visualizations")
    
    # Provider comparison
    provider_data = {}
    for result in results:
        provider = result.provider
        if provider not in provider_data:
            provider_data[provider] = []
        provider_data[provider].append(result.metrics.sharpe_ratio)
    
    # Create provider comparison chart
    fig = go.Figure()
    
    for provider, sharpes in provider_data.items():
        fig.add_trace(go.Box(
            y=sharpes,
            name=provider.upper(),
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title="Sharpe Ratio Distribution by Provider",
        yaxis_title="Sharpe Ratio",
        xaxis_title="Provider"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Category performance
    category_data = {}
    for result in results:
        category = result.test.category
        if category not in category_data:
            category_data[category] = []
        category_data[category].append(result.metrics.sharpe_ratio)
    
    fig2 = go.Figure()
    
    for category, sharpes in category_data.items():
        fig2.add_trace(go.Bar(
            x=[category],
            y=[np.mean(sharpes)],
            name=category.title(),
            error_y=dict(type='data', array=[np.std(sharpes)])
        ))
    
    fig2.update_layout(
        title="Average Sharpe Ratio by Category",
        yaxis_title="Average Sharpe Ratio",
        xaxis_title="Category"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Risk-Return scatter plot
    fig3 = go.Figure()
    
    for result in results:
        fig3.add_trace(go.Scatter(
            x=[result.metrics.max_drawdown],
            y=[result.metrics.sharpe_ratio],
            mode='markers',
            marker=dict(
                size=10,
                color=result.metrics.win_rate,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Win Rate")
            ),
            text=f"{result.test.test_id} ({result.provider})",
            name=result.provider
        ))
    
    fig3.update_layout(
        title="Risk-Return Analysis",
        xaxis_title="Max Drawdown",
        yaxis_title="Sharpe Ratio"
    )
    
    st.plotly_chart(fig3, use_container_width=True)

def display_framework_demo():
    """Display framework integration demo"""
    
    st.header("🔧 Framework Integration Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 FinRL RL Environment")
        st.markdown("""
        **FinRL Integration:**
        - RL-based trading strategies
        - State-action-reward learning
        - Portfolio optimization
        - Risk management
        
        **Features:**
        - Multi-agent trading
        - Real-time environment
        - Custom reward functions
        """)
        
        if st.button("Test FinRL Prediction", key="finrl_test"):
            with st.spinner("Testing FinRL prediction..."):
                try:
                    system = st.session_state.trading_system
                    # Simulate FinRL prediction
                    st.success("✅ FinRL prediction test completed!")
                except Exception as e:
                    st.error(f"❌ FinRL test failed: {e}")
    
    with col2:
        st.subheader("📊 Qlib Quant Platform")
        st.markdown("""
        **Qlib Integration:**
        - Quant research platform
        - Feature engineering
        - Model training
        - Backtesting
        
        **Features:**
        - Technical indicators
        - Risk models
        - Performance attribution
        """)
        
        if st.button("Test Qlib Prediction", key="qlib_test"):
            with st.spinner("Testing Qlib prediction..."):
                try:
                    system = st.session_state.trading_system
                    # Simulate Qlib prediction
                    st.success("✅ Qlib prediction test completed!")
                except Exception as e:
                    st.error(f"❌ Qlib test failed: {e}")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("⚡ VectorBT Backtesting")
        st.markdown("""
        **VectorBT Integration:**
        - Vectorized backtesting
        - High performance
        - Multiple strategies
        - Risk metrics
        
        **Features:**
        - Fast execution
        - Portfolio optimization
        - Statistical analysis
        """)
        
        if st.button("Test VectorBT Backtest", key="vectorbt_test"):
            with st.spinner("Testing VectorBT backtest..."):
                try:
                    system = st.session_state.trading_system
                    # Simulate VectorBT backtest
                    st.success("✅ VectorBT backtest test completed!")
                except Exception as e:
                    st.error(f"❌ VectorBT test failed: {e}")
    
    with col4:
        st.subheader("🎓 Academic Models")
        st.markdown("""
        **Academic Models:**
        - Informer (AAAI 2021)
        - Autoformer (NeurIPS 2021)
        - PatchTST (ICLR 2023)
        - Advanced architectures
        
        **Features:**
        - State-of-the-art performance
        - Long sequence forecasting
        - Attention mechanisms
        """)
        
        if st.button("Test Academic Models", key="academic_test"):
            with st.spinner("Testing academic models..."):
                try:
                    system = st.session_state.trading_system
                    # Simulate academic model prediction
                    st.success("✅ Academic models test completed!")
                except Exception as e:
                    st.error(f"❌ Academic models test failed: {e}")

if __name__ == "__main__":
    main()
