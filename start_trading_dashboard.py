#!/usr/bin/env python3
"""
Start Trading AI Benchmarking Dashboard
Launch script for the Streamlit trading benchmarking dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        "streamlit",
        "plotly",
        "pandas",
        "numpy",
        "asyncio"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def start_dashboard():
    """Start the Streamlit dashboard"""
    
    print("🚀 Starting Trading AI Benchmarking Dashboard...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Get current directory
    current_dir = Path(__file__).parent
    dashboard_file = current_dir / "trading_benchmarking_dashboard.py"
    
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    print(f"📁 Dashboard file: {dashboard_file}")
    print("🌐 Starting Streamlit server...")
    print("📱 Dashboard will open in your default browser")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting dashboard: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    
    print("🎯 Trading AI Benchmarking Dashboard Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"📂 Current directory: {current_dir}")
    
    # Check for required files
    required_files = [
        "trading_benchmarking_system.py",
        "trading_benchmarking_dashboard.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📁 Please run this script from the correct directory")
        return False
    
    print("✅ All required files found")
    
    # Start dashboard
    success = start_dashboard()
    
    if success:
        print("\n🎉 Dashboard session completed successfully!")
    else:
        print("\n❌ Dashboard failed to start")
        return False

if __name__ == "__main__":
    main()
