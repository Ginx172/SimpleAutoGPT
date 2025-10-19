# Trading AI Benchmarking System - Installation Script
# PowerShell script for Windows installation

Write-Host "🚀 Trading AI Benchmarking System - Installation" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Check Python version
Write-Host "🐍 Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check pip
Write-Host "📦 Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip not found. Please install pip first." -ForegroundColor Red
    exit 1
}

# Install core dependencies
Write-Host "📥 Installing core dependencies..." -ForegroundColor Yellow
$corePackages = @(
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "plotly>=5.0.0",
    "streamlit>=1.20.0",
    "aiohttp>=3.8.0",
    "python-dotenv>=0.19.0"
)

foreach ($package in $corePackages) {
    Write-Host "  Installing $package..." -ForegroundColor Cyan
    try {
        pip install $package --quiet
        Write-Host "  ✅ $package installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Failed to install $package" -ForegroundColor Red
    }
}

# Install trading dependencies
Write-Host "📈 Installing trading dependencies..." -ForegroundColor Yellow
$tradingPackages = @(
    "yfinance>=0.1.70",
    "talib>=0.4.25"
)

foreach ($package in $tradingPackages) {
    Write-Host "  Installing $package..." -ForegroundColor Cyan
    try {
        pip install $package --quiet
        Write-Host "  ✅ $package installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️ $package installation failed (optional)" -ForegroundColor Yellow
    }
}

# Install AI dependencies
Write-Host "🤖 Installing AI dependencies..." -ForegroundColor Yellow
$aiPackages = @(
    "torch>=1.9.0",
    "transformers>=4.20.0"
)

foreach ($package in $aiPackages) {
    Write-Host "  Installing $package..." -ForegroundColor Cyan
    try {
        pip install $package --quiet
        Write-Host "  ✅ $package installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️ $package installation failed (optional)" -ForegroundColor Yellow
    }
}

# Create .env file if it doesn't exist
Write-Host "🔑 Creating .env file..." -ForegroundColor Yellow
$envFile = ".env"
if (-not (Test-Path $envFile)) {
    $envContent = @"
# Trading AI Benchmarking System - API Keys
# Add your API keys here

OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
TOGETHER_API_KEY=your_together_key_here
MISTRAL_API_KEY=your_mistral_key_here

# Optional: Additional configuration
RISK_FREE_RATE=0.02
BENCHMARK_RETURN=0.08
TRANSACTION_COST=0.001
"@
    
    $envContent | Out-File -FilePath $envFile -Encoding UTF8
    Write-Host "✅ .env file created. Please add your API keys." -ForegroundColor Green
} else {
    Write-Host "✅ .env file already exists." -ForegroundColor Green
}

# Test installation
Write-Host "🧪 Testing installation..." -ForegroundColor Yellow
try {
    python -c "from trading_benchmarking_system import get_trading_benchmarking_system; print('✅ Installation test passed!')"
    Write-Host "✅ Installation test passed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Installation test failed. Please check dependencies." -ForegroundColor Red
}

# Create desktop shortcuts
Write-Host "🖥️ Creating desktop shortcuts..." -ForegroundColor Yellow
$currentDir = Get-Location
$desktopPath = [Environment]::GetFolderPath("Desktop")

# Dashboard shortcut
$dashboardShortcut = "$desktopPath\Trading Benchmarking Dashboard.lnk"
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($dashboardShortcut)
$Shortcut.TargetPath = "python"
$Shortcut.Arguments = "start_trading_dashboard.py"
$Shortcut.WorkingDirectory = $currentDir
$Shortcut.Description = "Trading AI Benchmarking Dashboard"
$Shortcut.Save()

Write-Host "✅ Desktop shortcut created: Trading Benchmarking Dashboard" -ForegroundColor Green

# Demo shortcut
$demoShortcut = "$desktopPath\Trading Benchmarking Demo.lnk"
$Shortcut = $WshShell.CreateShortcut($demoShortcut)
$Shortcut.TargetPath = "python"
$Shortcut.Arguments = "demo_trading_benchmarking.py"
$Shortcut.WorkingDirectory = $currentDir
$Shortcut.Description = "Trading AI Benchmarking Demo"
$Shortcut.Save()

Write-Host "✅ Desktop shortcut created: Trading Benchmarking Demo" -ForegroundColor Green

Write-Host "`n🎉 INSTALLATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

Write-Host "📚 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Add your API keys to the .env file" -ForegroundColor White
Write-Host "2. Run demo: python demo_trading_benchmarking.py" -ForegroundColor White
Write-Host "3. Start dashboard: python start_trading_dashboard.py" -ForegroundColor White
Write-Host "4. Read documentation: TRADING_BENCHMARKING_README.md" -ForegroundColor White

Write-Host "`n🖥️ Desktop shortcuts created:" -ForegroundColor Yellow
Write-Host "- Trading Benchmarking Dashboard" -ForegroundColor White
Write-Host "- Trading Benchmarking Demo" -ForegroundColor White

Write-Host "`n🚀 Ready to benchmark AI trading models!" -ForegroundColor Green
