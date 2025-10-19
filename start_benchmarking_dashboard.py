"""
Start Advanced AI Benchmarking Dashboard
Launch the comprehensive benchmarking system interface
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Advanced AI Benchmarking Dashboard...")
    print("=" * 60)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "benchmarking_dashboard.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"❌ Error: Dashboard not found at {dashboard_path}")
        print("Please ensure 'benchmarking_dashboard.py' exists in your project structure.")
        sys.exit(1)
    
    print(f"🎯 Launching Advanced AI Benchmarking Dashboard...")
    print(f"📁 Dashboard path: {dashboard_path}")
    print("=" * 60)
    print("🚀 Features available:")
    print("   • Comprehensive AI Model Benchmarking")
    print("   • Enterprise-grade Metrics & Analytics")
    print("   • Interactive Performance Visualization")
    print("   • Custom Test Suite Management")
    print("   • Export & Reporting Capabilities")
    print("=" * 60)
    
    try:
        # Use sys.executable to ensure the correct Python interpreter (from venv) is used
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=True)
    except FileNotFoundError:
        print("❌ Error: 'streamlit' command not found.")
        print("Please ensure Streamlit is installed in your virtual environment:")
        print("pip install streamlit plotly pandas")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit dashboard: {e}")
        print(f"Stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
