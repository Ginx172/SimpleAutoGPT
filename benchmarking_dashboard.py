"""
Advanced AI Benchmarking Dashboard
Interactive Streamlit interface for comprehensive AI model benchmarking
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import List, Dict, Any
import os

# Import our advanced benchmarking system
from advanced_benchmarking_system import (
    get_advanced_benchmarking_system,
    BenchmarkTest,
    BenchmarkResult,
    BenchmarkMetrics
)

# Initialize the benchmarking system
benchmarking_system = get_advanced_benchmarking_system()

st.set_page_config(
    page_title="AI Model Benchmarking Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🚀 Advanced AI Model Benchmarking Dashboard")
    st.markdown("**Comprehensive benchmarking system for AI models with enterprise-grade metrics**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Run Benchmark", "Results Analysis", "Test Management", "Performance Metrics", "Export Data"]
    )
    
    if page == "Run Benchmark":
        show_benchmark_runner()
    elif page == "Results Analysis":
        show_results_analysis()
    elif page == "Test Management":
        show_test_management()
    elif page == "Performance Metrics":
        show_performance_metrics()
    elif page == "Export Data":
        show_export_data()

def show_benchmark_runner():
    st.header("🏃‍♂️ Run AI Model Benchmark")
    st.markdown("Execute comprehensive benchmarks on AI models")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Benchmark Configuration")
        
        # Provider selection
        providers = ["openai", "anthropic", "groq", "together", "mistral"]
        selected_providers = st.multiselect(
            "Select AI Providers:",
            providers,
            default=["openai", "groq"]
        )
        
        # Model selection
        models = {
            "openai": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "groq": ["llama2-70b", "mixtral-8x7b", "gemma-7b"],
            "together": ["llama-2-70b", "codellama-34b", "wizardcoder-33b"],
            "mistral": ["mistral-7b", "mixtral-8x7b", "codestral-22b"]
        }
        
        selected_models = {}
        for provider in selected_providers:
            if provider in models:
                selected_models[provider] = st.multiselect(
                    f"Models for {provider}:",
                    models[provider],
                    default=[models[provider][0]] if models[provider] else []
                )
        
        # Test suite selection
        test_suites = list(benchmarking_system.test_suites.keys())
        selected_test_suite = st.selectbox(
            "Select Test Suite:",
            test_suites,
            index=0
        )
        
        # Advanced configuration
        with st.expander("Advanced Configuration"):
            concurrency_limit = st.slider("Concurrency Limit:", 1, 20, 10)
            timeout_seconds = st.slider("Timeout (seconds):", 10, 120, 30)
            evaluation_runs = st.slider("Evaluation Runs:", 1, 10, 3)
            warmup_requests = st.slider("Warmup Requests:", 0, 20, 5)
    
    with col2:
        st.subheader("Test Suite Preview")
        
        if selected_test_suite in benchmarking_system.test_suites:
            tests = benchmarking_system.test_suites[selected_test_suite]
            
            if isinstance(tests, list):
                st.write(f"**{len(tests)} tests in {selected_test_suite} suite:**")
                
                for i, test in enumerate(tests[:5]):  # Show first 5 tests
                    with st.expander(f"Test {i+1}: {test.test_id}"):
                        st.write(f"**Category:** {test.category}")
                        st.write(f"**Difficulty:** {test.difficulty_level}")
                        st.write(f"**Domain:** {test.domain}")
                        st.write(f"**Prompt:** {test.prompt[:100]}...")
                        if test.expected_keywords:
                            st.write(f"**Expected Keywords:** {', '.join(test.expected_keywords[:5])}")
                
                if len(tests) > 5:
                    st.info(f"... and {len(tests) - 5} more tests")
            else:
                st.write("Domain-specific test suite selected")
    
    # Run benchmark button
    if st.button("🚀 Start Benchmark", type="primary"):
        if not selected_providers:
            st.error("Please select at least one provider")
        elif not any(selected_models.values()):
            st.error("Please select at least one model")
        else:
            run_benchmark(selected_providers, selected_models, selected_test_suite, {
                "concurrency_limit": concurrency_limit,
                "timeout_seconds": timeout_seconds,
                "evaluation_runs": evaluation_runs,
                "warmup_requests": warmup_requests
            })

def run_benchmark(providers: List[str], models: Dict[str, List[str]], 
                 test_suite: str, config: Dict[str, Any]):
    """Run the benchmark with selected configuration"""
    
    st.subheader("🏃‍♂️ Running Benchmark...")
    
    # Update configuration
    benchmarking_system.benchmarking_config.update(config)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tests = 0
    completed_tests = 0
    
    # Calculate total tests
    for provider, provider_models in models.items():
        if provider in providers:
            total_tests += len(provider_models)
    
    if test_suite in benchmarking_system.test_suites:
        tests = benchmarking_system.test_suites[test_suite]
        if isinstance(tests, list):
            total_tests *= len(tests)
    
    # Run benchmarks
    all_results = []
    
    try:
        for provider in providers:
            if provider in models:
                for model in models[provider]:
                    status_text.text(f"Testing {provider}/{model}...")
                    
                    # Simulate benchmark execution
                    # In real implementation, this would call the actual benchmarking system
                    results = simulate_benchmark_run(provider, model, test_suite)
                    all_results.extend(results)
                    
                    completed_tests += len(results)
                    progress = completed_tests / total_tests if total_tests > 0 else 1
                    progress_bar.progress(progress)
        
        # Store results
        benchmarking_system.benchmark_results.extend(all_results)
        
        # Show completion
        progress_bar.progress(1.0)
        status_text.text("✅ Benchmark completed!")
        
        st.success(f"🎉 Benchmark completed! {len(all_results)} tests executed")
        
        # Show quick summary
        show_quick_summary(all_results)
        
    except Exception as e:
        st.error(f"❌ Benchmark failed: {str(e)}")

def simulate_benchmark_run(provider: str, model: str, test_suite: str) -> List[BenchmarkResult]:
    """Simulate benchmark run (placeholder implementation)"""
    
    results = []
    tests = benchmarking_system.test_suites.get(test_suite, [])
    
    if isinstance(tests, list):
        for test in tests[:3]:  # Limit to 3 tests for simulation
            # Create simulated metrics
            metrics = BenchmarkMetrics()
            metrics.response_time = 1.5 + (hash(f"{provider}{model}") % 100) / 100
            metrics.accuracy_score = 75 + (hash(f"{provider}{model}{test.test_id}") % 25)
            metrics.relevance_score = 80 + (hash(f"{provider}{model}{test.test_id}") % 20)
            metrics.total_cost = 0.001 + (hash(f"{provider}{model}") % 100) / 100000
            
            result = BenchmarkResult(
                provider=provider,
                model=model,
                test=test,
                response=f"Simulated response from {provider}/{model} for {test.test_id}",
                metrics=metrics,
                timestamp=datetime.now(),
                execution_time=2.0,
                tokens_used=150,
                success=True
            )
            results.append(result)
    
    return results

def show_quick_summary(results: List[BenchmarkResult]):
    """Show quick summary of benchmark results"""
    
    if not results:
        return
    
    st.subheader("📊 Quick Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", len(results))
    
    with col2:
        avg_accuracy = sum(r.metrics.accuracy_score for r in results) / len(results)
        st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
    
    with col3:
        avg_speed = sum(r.metrics.response_time for r in results) / len(results)
        st.metric("Avg Speed", f"{avg_speed:.2f}s")
    
    with col4:
        total_cost = sum(r.metrics.total_cost for r in results)
        st.metric("Total Cost", f"${total_cost:.4f}")

def show_results_analysis():
    st.header("📊 Results Analysis")
    st.markdown("Analyze and visualize benchmark results")
    
    if not benchmarking_system.benchmark_results:
        st.warning("No benchmark results available. Please run a benchmark first.")
        return
    
    results = benchmarking_system.benchmark_results
    
    # Overall statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        successful_results = [r for r in results if r.success]
        success_rate = (len(successful_results) / len(results)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col2:
        avg_accuracy = sum(r.metrics.accuracy_score for r in successful_results) / len(successful_results)
        st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
    
    with col3:
        total_cost = sum(r.metrics.total_cost for r in successful_results)
        st.metric("Total Cost", f"${total_cost:.4f}")
    
    # Provider comparison
    st.subheader("🏆 Provider Comparison")
    
    by_provider = {}
    for result in successful_results:
        if result.provider not in by_provider:
            by_provider[result.provider] = []
        by_provider[result.provider].append(result)
    
    provider_data = []
    for provider, provider_results in by_provider.items():
        avg_accuracy = sum(r.metrics.accuracy_score for r in provider_results) / len(provider_results)
        avg_speed = sum(r.metrics.response_time for r in provider_results) / len(provider_results)
        total_cost = sum(r.metrics.total_cost for r in provider_results)
        
        provider_data.append({
            'Provider': provider,
            'Tests': len(provider_results),
            'Avg Accuracy': avg_accuracy,
            'Avg Speed': avg_speed,
            'Total Cost': total_cost
        })
    
    if provider_data:
        df_providers = pd.DataFrame(provider_data)
        st.dataframe(df_providers, use_container_width=True)
        
        # Provider comparison chart
        fig = px.bar(df_providers, x='Provider', y='Avg Accuracy', 
                     title='Provider Accuracy Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    # Category performance
    st.subheader("📊 Category Performance")
    
    by_category = {}
    for result in successful_results:
        if result.test.category not in by_category:
            by_category[result.test.category] = []
        by_category[result.test.category].append(result)
    
    category_data = []
    for category, category_results in by_category.items():
        avg_accuracy = sum(r.metrics.accuracy_score for r in category_results) / len(category_results)
        avg_quality = sum(r.metrics.relevance_score for r in category_results) / len(category_results)
        
        category_data.append({
            'Category': category,
            'Tests': len(category_results),
            'Avg Accuracy': avg_accuracy,
            'Avg Quality': avg_quality
        })
    
    if category_data:
        df_categories = pd.DataFrame(category_data)
        st.dataframe(df_categories, use_container_width=True)
        
        # Category performance chart
        fig = px.bar(df_categories, x='Category', y='Avg Accuracy', 
                     title='Category Performance')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("🔍 Detailed Results")
    
    detailed_data = []
    for result in successful_results[:20]:  # Show first 20 results
        detailed_data.append({
            'Provider': result.provider,
            'Model': result.model,
            'Test ID': result.test.test_id,
            'Category': result.test.category,
            'Accuracy': f"{result.metrics.accuracy_score:.1f}%",
            'Speed': f"{result.metrics.response_time:.2f}s",
            'Cost': f"${result.metrics.total_cost:.4f}",
            'Success': result.success
        })
    
    if detailed_data:
        df_detailed = pd.DataFrame(detailed_data)
        st.dataframe(df_detailed, use_container_width=True)

def show_test_management():
    st.header("🧪 Test Management")
    st.markdown("Manage and customize benchmark test suites")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Available Test Suites")
        
        for suite_name, tests in benchmarking_system.test_suites.items():
            with st.expander(f"📚 {suite_name.title()} Suite"):
                if isinstance(tests, list):
                    st.write(f"**Tests:** {len(tests)}")
                    
                    # Show test categories
                    categories = list(set(test.category for test in tests))
                    st.write(f"**Categories:** {', '.join(categories)}")
                    
                    # Show difficulty levels
                    difficulties = list(set(test.difficulty_level for test in tests))
                    st.write(f"**Difficulty Levels:** {', '.join(difficulties)}")
                    
                    # Show sample tests
                    st.write("**Sample Tests:**")
                    for test in tests[:3]:
                        st.write(f"- {test.test_id}: {test.category} ({test.difficulty_level})")
                    
                    if len(tests) > 3:
                        st.write(f"... and {len(tests) - 3} more tests")
                else:
                    st.write("Domain-specific test suite")
                    for domain, domain_tests in tests.items():
                        st.write(f"- {domain}: {len(domain_tests)} tests")
    
    with col2:
        st.subheader("Create Custom Test")
        
        with st.form("create_test_form"):
            test_id = st.text_input("Test ID:", "custom_001")
            category = st.selectbox("Category:", ["coding", "reasoning", "creative", "factual", "problem_solving"])
            subcategory = st.text_input("Subcategory:", "custom")
            prompt = st.text_area("Prompt:", "Enter your test prompt here...")
            expected_keywords = st.text_input("Expected Keywords (comma-separated):", "keyword1, keyword2")
            difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard", "expert"])
            domain = st.text_input("Domain:", "general")
            weight = st.slider("Weight:", 0.5, 3.0, 1.0)
            
            submitted = st.form_submit_button("Create Test")
            
            if submitted:
                keywords_list = [k.strip() for k in expected_keywords.split(",") if k.strip()]
                
                new_test = BenchmarkTest(
                    test_id=test_id,
                    category=category,
                    subcategory=subcategory,
                    prompt=prompt,
                    expected_keywords=keywords_list,
                    difficulty_level=difficulty,
                    domain=domain,
                    weight=weight
                )
                
                # Add to test suite
                if "custom" not in benchmarking_system.test_suites:
                    benchmarking_system.test_suites["custom"] = []
                
                benchmarking_system.test_suites["custom"].append(new_test)
                
                st.success(f"✅ Test '{test_id}' created successfully!")

def show_performance_metrics():
    st.header("📈 Performance Metrics")
    st.markdown("Advanced performance analysis and visualization")
    
    if not benchmarking_system.benchmark_results:
        st.warning("No benchmark results available. Please run a benchmark first.")
        return
    
    results = benchmarking_system.benchmark_results
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        st.warning("No successful benchmark results available.")
        return
    
    # Performance metrics dashboard
    st.subheader("🚀 Performance Dashboard")
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy vs Speed', 'Cost Efficiency', 'Category Performance', 'Provider Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy vs Speed scatter plot
    providers = [r.provider for r in successful_results]
    accuracies = [r.metrics.accuracy_score for r in successful_results]
    speeds = [r.metrics.response_time for r in successful_results]
    
    fig.add_trace(
        go.Scatter(x=speeds, y=accuracies, mode='markers', 
                  text=providers, name='Models',
                  marker=dict(size=10, opacity=0.7)),
        row=1, col=1
    )
    
    # Cost Efficiency
    provider_costs = {}
    provider_quality = {}
    
    for result in successful_results:
        if result.provider not in provider_costs:
            provider_costs[result.provider] = []
            provider_quality[result.provider] = []
        
        provider_costs[result.provider].append(result.metrics.total_cost)
        provider_quality[result.provider].append(result.metrics.accuracy_score)
    
    providers_list = list(provider_costs.keys())
    avg_costs = [sum(provider_costs[p]) / len(provider_costs[p]) for p in providers_list]
    
    fig.add_trace(
        go.Bar(x=providers_list, y=avg_costs, name='Average Cost'),
        row=1, col=2
    )
    
    # Category Performance
    categories = list(set([r.test.category for r in successful_results]))
    category_accuracies = []
    
    for category in categories:
        category_results = [r for r in successful_results if r.test.category == category]
        avg_accuracy = sum(r.metrics.accuracy_score for r in category_results) / len(category_results)
        category_accuracies.append(avg_accuracy)
    
    fig.add_trace(
        go.Bar(x=categories, y=category_accuracies, name='Category Accuracy'),
        row=2, col=1
    )
    
    # Provider Comparison
    provider_accuracies = [sum([r.metrics.accuracy_score for r in successful_results if r.provider == p]) / len([r for r in successful_results if r.provider == p]) for p in providers_list]
    
    fig.add_trace(
        go.Bar(x=providers_list, y=provider_accuracies, name='Provider Accuracy'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="AI Model Performance Dashboard",
        showlegend=False,
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Advanced metrics analysis
    st.subheader("🔬 Advanced Metrics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Response Time Distribution**")
        
        response_times = [r.metrics.response_time for r in successful_results]
        
        fig_hist = px.histogram(
            x=response_times,
            title="Response Time Distribution",
            labels={'x': 'Response Time (s)', 'y': 'Count'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.write("**Accuracy Distribution**")
        
        accuracies = [r.metrics.accuracy_score for r in successful_results]
        
        fig_hist = px.histogram(
            x=accuracies,
            title="Accuracy Distribution",
            labels={'x': 'Accuracy (%)', 'y': 'Count'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def show_export_data():
    st.header("📤 Export Data")
    st.markdown("Export benchmark results and reports")
    
    if not benchmarking_system.benchmark_results:
        st.warning("No benchmark results available. Please run a benchmark first.")
        return
    
    results = benchmarking_system.benchmark_results
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Export Options")
        
        export_format = st.selectbox(
            "Export Format:",
            ["CSV", "JSON", "PDF Report", "Excel"]
        )
        
        if st.button("📊 Export Results"):
            if export_format == "CSV":
                # Export to CSV
                filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                success = benchmarking_system.export_results_to_csv(results, filename)
                
                if success:
                    st.success(f"✅ Results exported to {filename}")
                else:
                    st.error("❌ Export failed")
            
            elif export_format == "JSON":
                # Export to JSON
                filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                export_data = []
                for result in results:
                    export_data.append({
                        'provider': result.provider,
                        'model': result.model,
                        'test_id': result.test.test_id,
                        'category': result.test.category,
                        'accuracy_score': result.metrics.accuracy_score,
                        'response_time': result.metrics.response_time,
                        'total_cost': result.metrics.total_cost,
                        'timestamp': result.timestamp.isoformat()
                    })
                
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                st.success(f"✅ Results exported to {filename}")
            
            elif export_format == "PDF Report":
                # Generate comprehensive report
                report = benchmarking_system.generate_comprehensive_report(results)
                
                filename = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                with open(filename, 'w') as f:
                    f.write(report)
                
                st.success(f"✅ Report generated: {filename}")
                st.text_area("Report Preview:", report[:1000] + "...", height=300)
    
    with col2:
        st.subheader("Export Statistics")
        
        successful_results = [r for r in results if r.success]
        
        st.metric("Total Tests", len(results))
        st.metric("Successful Tests", len(successful_results))
        st.metric("Success Rate", f"{(len(successful_results) / len(results)) * 100:.1f}%")
        
        if successful_results:
            avg_accuracy = sum(r.metrics.accuracy_score for r in successful_results) / len(successful_results)
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
            
            total_cost = sum(r.metrics.total_cost for r in successful_results)
            st.metric("Total Cost", f"${total_cost:.4f}")
        
        # Quick export buttons
        st.subheader("Quick Export")
        
        if st.button("📋 Copy Results to Clipboard"):
            # Convert results to clipboard-friendly format
            clipboard_data = []
            for result in successful_results:
                clipboard_data.append(f"{result.provider}/{result.model}: {result.metrics.accuracy_score:.1f}% accuracy, {result.metrics.response_time:.2f}s")
            
            st.text_area("Results for Clipboard:", "\n".join(clipboard_data), height=200)

if __name__ == "__main__":
    main()
