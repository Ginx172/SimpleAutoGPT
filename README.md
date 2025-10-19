# 🔍 Universal S3 Bucket AI Analyzer

A powerful web application that analyzes every file in your AWS S3 bucket using AI to provide comprehensive insights and complete inventory.

## ✨ Features

- **🔍 Real File Type Detection**: Uses python-magic to detect actual file types from content
- **🤖 AI-Powered Analysis**: Analyzes supported files with OpenAI GPT models
- **📊 Complete Inventory**: Catalogs every file in your bucket, including unsupported types
- **🌐 Web Interface**: Beautiful Streamlit-based user interface
- **📈 Real-time Progress**: Live progress tracking during analysis
- **💾 Export Results**: Download comprehensive CSV reports

## 🎯 Supported File Types

### Full AI Analysis:
- 📄 Text files (.txt)
- 📋 PDF documents (.pdf)
- 📊 CSV data files (.csv)
- 🗃️ Parquet files (.parquet)
- 🔧 JSON files (.json)
- 🥒 Pickle files (.pkl)

### Complete Inventory:
- 🖼️ Images (JPEG, PNG, GIF, etc.)
- 🎵 Audio files (MP3, WAV, etc.)
- 🎬 Video files (MP4, AVI, etc.)
- 📦 Archives (ZIP, TAR, etc.)
- 💻 Executables and binaries
- 📄 Any other file type

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project directory:

```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_DEFAULT_REGION=us-east-1

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### 4. Analyze Your Bucket

1. Enter your S3 bucket name in the input field
2. Click "Start Analysis"
3. Watch the real-time progress
4. View results and download the complete report

## 📊 Analysis Output

The application provides:

- **📈 Summary Statistics**: Total files, analyzed files, unsupported files, errors
- **📁 File Type Distribution**: Breakdown of all file types found
- **🏷️ Content Categories**: AI-analyzed categories for supported files
- **📋 Data Preview**: First 20 rows of results
- **💾 CSV Export**: Complete downloadable report with all analysis data

## 🔧 Technical Details

### AI Analysis Features:
- **Content Categorization**: Automatic classification of document types
- **Summary Generation**: Concise summaries of file contents
- **Utility Scoring**: 1-10 scale rating of content usefulness
- **RAG Suggestions**: Recommendations for Retrieval-Augmented Generation systems
- **Security Alerts**: Identification of potential security concerns

### Performance Optimizations:
- **Efficient Processing**: Only reads necessary portions of files
- **Cost Optimization**: Limits API calls to essential content
- **Memory Management**: Processes files without loading entire contents
- **Progress Tracking**: Real-time feedback during analysis

## 🛡️ Security Features

- **Pickle File Warnings**: Clear security warnings for potentially dangerous files
- **Error Handling**: Comprehensive error handling for all file types
- **Safe Processing**: Limits content exposure for security-sensitive files

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── s3_openai_analyzer.py  # Original command-line version
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .env                  # Environment variables (create this)
```

## 🤝 Usage Examples

### Trading Data Analysis
Perfect for analyzing buckets containing:
- Stock market data (CSV files)
- Financial reports (PDF files)
- Trading algorithms (Python/Pickle files)
- Configuration files (JSON files)

### Data Lake Cataloging
Ideal for:
- Complete inventory of mixed file types
- Content discovery and categorization
- Migration planning
- Compliance auditing

### Research Data Management
Great for:
- Academic paper collections (PDF files)
- Research datasets (CSV/Parquet files)
- Configuration and metadata (JSON files)
- Analysis results (various formats)

## 🎉 Ready to Analyze!

Your S3 bucket analysis tool is now ready to provide comprehensive insights into your data. Simply run `streamlit run app.py` and start exploring your files with AI-powered intelligence!

---

*Built with ❤️ using Streamlit, OpenAI, and AWS*