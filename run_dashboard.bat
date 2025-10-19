@echo off
echo 🚀 Trading AI Benchmarking Dashboard
echo ===================================
echo.
echo Starting Streamlit dashboard...
echo Dashboard will open in your default browser
echo.
echo Press Ctrl+C to stop the dashboard
echo.

cd /d "%~dp0"
python start_trading_dashboard.py

echo.
echo Dashboard stopped. Press any key to exit...
pause > nul
