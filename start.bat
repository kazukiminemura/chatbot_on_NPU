@echo off
echo 🤖 Llama2-7B NPU Chatbot
echo ========================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements if needed
if not exist "venv\pyvenv.cfg" (
    echo 📋 Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
)

:: Check if requirements are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo 📋 Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install requirements
        pause
        exit /b 1
    )
)

:: Start the application
echo 🚀 Starting Llama2-7B NPU Chatbot...
python run.py

pause