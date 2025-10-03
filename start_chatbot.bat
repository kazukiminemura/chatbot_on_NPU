@echo off
echo Starting DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot...

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Start the application
python run.py

pause