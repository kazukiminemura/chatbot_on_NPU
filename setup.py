"""
Setup script for Llama2-7B NPU Chatbot
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llama2-7b-npu-chatbot",
    version="1.0.0",
    author="AI Assistant",
    description="Llama2-7B NPU Chatbot powered by OpenVINO and Intel NPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukiminemura/chatbot_on_NPU",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "llama2-chatbot=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["static/**/*", "config.json"],
    },
)