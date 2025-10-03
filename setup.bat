@echo off
echo ============================================================
echo DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot Setup
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://python.org
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo Python version: %PYTHON_VERSION%

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Run setup
echo Running setup script...
python setup.py
if errorlevel 1 (
    echo Error: Setup failed
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup completed successfully!
echo ============================================================
echo.
echo To start the chatbot, run: start_chatbot.bat
echo Or manually run: python run.py
echo.
pause