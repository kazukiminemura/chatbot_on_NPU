#!/usr/bin/env python3
"""
Setup script for DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot
"""
import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    return True


def install_requirements():
    """Install Python requirements."""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def check_openvino():
    """Check if OpenVINO is properly installed."""
    print("Checking OpenVINO installation...")
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        
        print(f"✓ OpenVINO installed. Available devices: {devices}")
        
        if "NPU" in devices:
            print("✓ NPU device detected")
        else:
            print("⚠ NPU device not detected. Will use CPU fallback.")
        
        return True
    except ImportError:
        print("✗ OpenVINO not properly installed")
        return False


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = ["models", "logs"]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def validate_config():
    """Validate configuration file."""
    print("Validating configuration...")
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        required_keys = ["model", "inference", "server", "hardware", "openvino"]
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing configuration key: {key}")
                return False
        
        print("✓ Configuration file is valid")
        return True
    except FileNotFoundError:
        print("✗ config.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in config.json: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check OpenVINO
    if not check_openvino():
        return False
    
    # Create directories
    create_directories()
    
    # Validate configuration
    if not validate_config():
        return False
    
    print("\n" + "=" * 60)
    print("✓ Setup completed successfully!")
    print("=" * 60)
    print("\nTo start the chatbot:")
    print("  python run.py")
    print("\nThen open your browser to:")
    print("  http://localhost:8000")
    print("\n" + "=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)