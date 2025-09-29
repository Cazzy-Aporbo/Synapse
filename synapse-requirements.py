#!/usr/bin/env python3
"""
SYNAPSE - Complete Requirements and Dependencies Documentation
==============================================================
Cognitive Augmentation Research System
Version: 1.0.0
Documented by Cazzy Aporbo 2025
License: MIT

This file contains all pip installations and import statements required for SYNAPSE,
with comprehensive documentation explaining what each package does and why it's needed.

USAGE:
------
1. Run the pip installations in order (or use requirements.txt)
2. Import statements are organized by category
3. Each package includes explanation of its role in SYNAPSE
"""

# ==============================================================================
# PART 1: PIP INSTALLATIONS
# ==============================================================================
"""
Run these commands in your terminal/command prompt to install all dependencies.
It's recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
"""

# ------------------------------------------------------------------------------
# CORE SCIENTIFIC COMPUTING
# ------------------------------------------------------------------------------
"""
pip install numpy==1.24.3
# NumPy: Fundamental package for scientific computing with Python
# Used for: Multi-dimensional arrays, mathematical functions, linear algebra
# SYNAPSE usage: Embedding vectors, matrix operations, statistical calculations

pip install pandas==2.0.3
# Pandas: Powerful data structures and data analysis tools
# Used for: DataFrames, time series, data manipulation
# SYNAPSE usage: Research data organization, CSV processing, results tabulation

pip install scipy==1.11.1
# SciPy: Scientific and technical computing
# Used for: Optimization, signal processing, statistics, linear algebra
# SYNAPSE usage: Statistical analysis, clustering algorithms, signal processing for chaos theory
"""

# ------------------------------------------------------------------------------
# MACHINE LEARNING & AI
# ------------------------------------------------------------------------------
"""
pip install scikit-learn==1.3.0
# Scikit-learn: Machine learning library
# Used for: Classification, regression, clustering, dimensionality reduction
# SYNAPSE usage: PCA for dimensionality reduction, KMeans clustering, data preprocessing

pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# PyTorch: Deep learning framework
# Used for: Neural networks, automatic differentiation, GPU acceleration
# SYNAPSE usage: Custom neural architectures, attention mechanisms, tensor operations
# Note: Use cu118 for CUDA 11.8, cpu for CPU-only version

pip install torchvision==0.16.0
# TorchVision: Computer vision library for PyTorch
# Used for: Image transformations, pre-trained models
# SYNAPSE usage: Optional image processing for multimodal research

pip install torchaudio==2.1.0
# TorchAudio: Audio processing for PyTorch
# Used for: Audio I/O, transformations, datasets
# SYNAPSE usage: Optional audio processing for research synthesis
"""

# ------------------------------------------------------------------------------
# NLP & TRANSFORMERS
# ------------------------------------------------------------------------------
"""
pip install transformers==4.35.0
# Transformers: State-of-the-art NLP models by Hugging Face
# Used for: BERT, GPT, T5 and other transformer models
# SYNAPSE usage: Text embeddings, language models, tokenization

pip install sentence-transformers==2.2.2
# Sentence Transformers: Sentence and paragraph embeddings
# Used for: Semantic similarity, semantic search
# SYNAPSE usage: Creating embeddings for knowledge graph nodes, semantic search

pip install tokenizers==0.14.1
# Tokenizers: Fast tokenization library
# Used for: Text tokenization for transformer models
# SYNAPSE usage: Preprocessing text for neural networks

pip install nltk==3.8.1
# NLTK: Natural Language Toolkit
# Used for: Tokenization, stemming, tagging, parsing
# SYNAPSE usage: Text preprocessing, linguistic analysis

pip install spacy==3.6.0
python -m spacy download en_core_web_sm
# spaCy: Industrial-strength NLP
# Used for: Named entity recognition, POS tagging, dependency parsing
# SYNAPSE usage: Entity extraction from research papers, text analysis
"""

# ------------------------------------------------------------------------------
# AI APIS (OPTIONAL - Only if using external services)
# ------------------------------------------------------------------------------
"""
pip install anthropic==0.7.0
# Anthropic: Official Claude API client
# Used for: Accessing Claude AI models
# SYNAPSE usage: Multi-AI orchestration with Claude

pip install openai==1.3.0
# OpenAI: Official OpenAI API client
# Used for: Accessing GPT-4 and other OpenAI models
# SYNAPSE usage: Multi-AI orchestration with GPT-4

pip install google-generativeai==0.3.0
# Google Generative AI: Official Google AI client
# Used for: Accessing Gemini and other Google AI models
# SYNAPSE usage: Multi-AI orchestration with Gemini (optional)
"""

# ------------------------------------------------------------------------------
# VECTOR SEARCH & DATABASES
# ------------------------------------------------------------------------------
"""
pip install faiss-cpu==1.7.4
# FAISS: Facebook AI Similarity Search
# Used for: Efficient similarity search and clustering of dense vectors
# SYNAPSE usage: Fast semantic search in knowledge graph, nearest neighbor queries
# Note: Use faiss-gpu for GPU support

pip install chromadb==0.4.18
# ChromaDB: AI-native embedding database
# Used for: Vector database with metadata filtering
# SYNAPSE usage: Alternative to FAISS for persistent vector storage (optional)

pip install redis==4.6.0
# Redis: In-memory data structure store
# Used for: Caching, message broker, real-time operations
# SYNAPSE usage: Caching AI responses, session management

pip install pymongo==4.4.1
# PyMongo: MongoDB driver for Python
# Used for: NoSQL database operations
# SYNAPSE usage: Storing unstructured research data, flexible schemas

pip install sqlalchemy==2.0.23
# SQLAlchemy: SQL toolkit and ORM
# Used for: Database abstraction layer, ORM
# SYNAPSE usage: Managing SQLite database for structured data
"""

# ------------------------------------------------------------------------------
# GRAPH PROCESSING
# ------------------------------------------------------------------------------
"""
pip install networkx==3.1
# NetworkX: Network analysis in Python
# Used for: Graph creation, manipulation, and analysis
# SYNAPSE usage: Knowledge graph structure, graph algorithms, centrality measures

pip install igraph==0.10.6
# iGraph: High-performance graph library
# Used for: Fast graph operations, community detection
# SYNAPSE usage: Alternative graph backend for large-scale operations (optional)

pip install pyvis==0.3.2
# PyVis: Interactive network visualizations
# Used for: Interactive graph visualization in browser
# SYNAPSE usage: Visualizing knowledge graph relationships (optional)
"""

# ------------------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------------------
"""
pip install matplotlib==3.7.2
# Matplotlib: Comprehensive plotting library
# Used for: Static plots, charts, figures
# SYNAPSE usage: Basic visualizations, debugging plots

pip install seaborn==0.12.2
# Seaborn: Statistical data visualization
# Used for: Beautiful statistical graphics
# SYNAPSE usage: Heatmaps, correlation matrices, distribution plots

pip install plotly==5.15.0
# Plotly: Interactive graphing library
# Used for: Interactive 3D visualizations, dashboards
# SYNAPSE usage: 3D knowledge graphs, interactive research visualizations

pip install dash==2.11.1
# Dash: Web application framework for Python
# Used for: Building analytical web applications
# SYNAPSE usage: Creating web interface for SYNAPSE (optional)

pip install bokeh==3.2.1
# Bokeh: Interactive visualization library
# Used for: Interactive plots and dashboards
# SYNAPSE usage: Alternative visualization backend (optional)

pip install altair==5.1.2
# Altair: Declarative statistical visualization
# Used for: Vega-Lite based visualizations
# SYNAPSE usage: Statistical charts and graphics (optional)
"""

# ------------------------------------------------------------------------------
# ACADEMIC RESEARCH TOOLS
# ------------------------------------------------------------------------------
"""
pip install arxiv==1.4.8
# arXiv: Python wrapper for arXiv API
# Used for: Searching and downloading papers from arXiv
# SYNAPSE usage: Retrieving research papers for synthesis

pip install scholarly==1.7.11
# Scholarly: Google Scholar scraper
# Used for: Retrieving author profiles and citations from Google Scholar
# SYNAPSE usage: Citation analysis, finding related research
# Note: Can be rate-limited by Google

pip install biopython==1.81
# Biopython: Biological computation tools
# Used for: Accessing PubMed, biological sequences
# SYNAPSE usage: Medical/biological research retrieval

pip install pymed==0.8.9
# PyMed: PubMed access library
# Used for: Searching PubMed database
# SYNAPSE usage: Medical research papers (alternative to Biopython)

pip install semanticscholar==0.3.2
# Semantic Scholar: API client for Semantic Scholar
# Used for: Accessing Semantic Scholar database
# SYNAPSE usage: Finding papers with AI-extracted features

pip install crossref-commons==0.0.7
# Crossref Commons: Crossref API client
# Used for: DOI resolution, citation metadata
# SYNAPSE usage: Resolving paper DOIs, getting citation data
"""

# ------------------------------------------------------------------------------
# WEB SCRAPING & REQUESTS
# ------------------------------------------------------------------------------
"""
pip install requests==2.31.0
# Requests: HTTP library for Python
# Used for: Making HTTP requests
# SYNAPSE usage: API calls, downloading papers

pip install beautifulsoup4==4.12.2
# Beautiful Soup: Web scraping library
# Used for: Parsing HTML and XML
# SYNAPSE usage: Extracting data from web pages

pip install selenium==4.11.2
# Selenium: Web browser automation
# Used for: Scraping JavaScript-rendered content
# SYNAPSE usage: Accessing dynamic research databases (optional)

pip install scrapy==2.11.0
# Scrapy: Web scraping framework
# Used for: Large-scale web scraping
# SYNAPSE usage: Bulk paper metadata extraction (optional)

pip install httpx==0.25.1
# HTTPX: Modern HTTP client
# Used for: Async HTTP requests
# SYNAPSE usage: Asynchronous API calls
"""

# ------------------------------------------------------------------------------
# ASYNC & PARALLEL PROCESSING
# ------------------------------------------------------------------------------
"""
pip install asyncio==3.4.3
# Asyncio: Asynchronous I/O (usually included with Python)
# Used for: Concurrent code using async/await
# SYNAPSE usage: Parallel AI queries, async research retrieval

pip install aiohttp==3.8.5
# AIOHTTP: Async HTTP client/server
# Used for: Asynchronous HTTP requests
# SYNAPSE usage: Parallel API calls to multiple AI services

pip install multiprocess==0.70.15
# Multiprocess: Better multiprocessing
# Used for: Parallel processing with better serialization
# SYNAPSE usage: CPU-bound parallel tasks

pip install ray==2.7.0
# Ray: Distributed computing framework
# Used for: Distributed machine learning
# SYNAPSE usage: Scaling to multiple machines (optional)

pip install dask==2023.9.2
# Dask: Parallel computing library
# Used for: Parallel arrays, dataframes, and lists
# SYNAPSE usage: Large-scale data processing (optional)
"""

# ------------------------------------------------------------------------------
# CLI & TERMINAL OUTPUT
# ------------------------------------------------------------------------------
"""
pip install rich==13.5.2
# Rich: Beautiful terminal formatting
# Used for: Colored output, tables, progress bars
# SYNAPSE usage: Beautiful CLI interface, progress tracking

pip install tqdm==4.65.0
# TQDM: Progress bars for loops
# Used for: Simple progress indicators
# SYNAPSE usage: Showing progress during long operations

pip install click==8.1.7
# Click: Command-line interface creation
# Used for: Building CLI commands
# SYNAPSE usage: Command-line interface for SYNAPSE

pip install typer==0.9.0
# Typer: Modern CLI apps with type hints
# Used for: Type-safe CLI creation
# SYNAPSE usage: Alternative CLI framework (optional)

pip install colorama==0.4.6
# Colorama: Cross-platform colored terminal text
# Used for: Terminal colors on Windows
# SYNAPSE usage: Windows compatibility for colored output
"""

# ------------------------------------------------------------------------------
# DATA FORMATS & SERIALIZATION
# ------------------------------------------------------------------------------
"""
pip install pyyaml==6.0.1
# PyYAML: YAML parser and emitter
# Used for: Reading/writing YAML files
# SYNAPSE usage: Configuration files

pip install jsonschema==4.19.0
# JSON Schema: JSON validation
# Used for: Validating JSON data structure
# SYNAPSE usage: API request/response validation

pip install msgpack==1.0.5
# MessagePack: Efficient binary serialization
# Used for: Fast serialization
# SYNAPSE usage: Caching serialized data

pip install pickle5==0.0.12
# Pickle5: Backport of pickle protocol 5
# Used for: Python object serialization
# SYNAPSE usage: Saving/loading Python objects

pip install toml==0.10.2
# TOML: Tom's Obvious Minimal Language parser
# Used for: Configuration files
# SYNAPSE usage: Project configuration (optional)
"""

# ------------------------------------------------------------------------------
# SECURITY & AUTHENTICATION
# ------------------------------------------------------------------------------
"""
pip install cryptography==41.0.3
# Cryptography: Cryptographic recipes and primitives
# Used for: Encryption, decryption, digital signatures
# SYNAPSE usage: Encrypting sensitive data, API key storage

pip install pyjwt==2.8.0
# PyJWT: JSON Web Token implementation
# Used for: JWT token creation and verification
# SYNAPSE usage: API authentication

pip install python-dotenv==1.0.0
# Python-dotenv: Load environment variables from .env
# Used for: Managing environment variables
# SYNAPSE usage: Secure API key management

pip install bcrypt==4.0.1
# Bcrypt: Password hashing
# Used for: Secure password storage
# SYNAPSE usage: User authentication (if implementing accounts)

pip install passlib==1.7.4
# Passlib: Password hashing library
# Used for: Multiple hashing algorithms
# SYNAPSE usage: Alternative to bcrypt (optional)
"""

# ------------------------------------------------------------------------------
# IMAGE & MULTIMEDIA (OPTIONAL)
# ------------------------------------------------------------------------------
"""
pip install pillow==10.0.0
# Pillow: Python Imaging Library
# Used for: Image processing
# SYNAPSE usage: Processing images in research papers

pip install opencv-python==4.8.0.74
# OpenCV: Computer vision library
# Used for: Advanced image processing
# SYNAPSE usage: Extracting figures from papers (optional)

pip install librosa==0.10.0
# Librosa: Audio analysis
# Used for: Audio and music analysis
# SYNAPSE usage: Audio research analysis (optional)

pip install soundfile==0.11.0
# SoundFile: Audio file I/O
# Used for: Reading/writing audio files
# SYNAPSE usage: Audio data handling (optional)

pip install moviepy==1.0.3
# MoviePy: Video editing
# Used for: Video processing
# SYNAPSE usage: Research presentation videos (optional)
"""

# ------------------------------------------------------------------------------
# TESTING & QUALITY ASSURANCE
# ------------------------------------------------------------------------------
"""
pip install pytest==7.4.2
# Pytest: Testing framework
# Used for: Unit and integration testing
# SYNAPSE usage: Testing all components

pip install pytest-asyncio==0.21.1
# Pytest-asyncio: Async test support
# Used for: Testing async functions
# SYNAPSE usage: Testing async AI queries

pip install pytest-cov==4.1.0
# Pytest-cov: Coverage plugin for pytest
# Used for: Code coverage reporting
# SYNAPSE usage: Ensuring test coverage

pip install black==23.7.0
# Black: Code formatter
# Used for: Automatic code formatting
# SYNAPSE usage: Maintaining code style

pip install flake8==6.1.0
# Flake8: Style guide enforcement
# Used for: Linting Python code
# SYNAPSE usage: Code quality checks

pip install mypy==1.5.1
# Mypy: Static type checker
# Used for: Type checking
# SYNAPSE usage: Ensuring type safety

pip install bandit==1.7.5
# Bandit: Security linter
# Used for: Finding security issues
# SYNAPSE usage: Security auditing
"""

# ------------------------------------------------------------------------------
# DOCUMENTATION
# ------------------------------------------------------------------------------
"""
pip install sphinx==7.2.6
# Sphinx: Documentation generator
# Used for: Creating documentation
# SYNAPSE usage: API documentation

pip install mkdocs==1.5.2
# MkDocs: Project documentation with Markdown
# Used for: Creating documentation websites
# SYNAPSE usage: User guide documentation

pip install pydoc-markdown==4.8.2
# Pydoc-Markdown: Generate markdown from docstrings
# Used for: Auto-generating API docs
# SYNAPSE usage: Automatic documentation generation
"""

# ==============================================================================
# PART 2: IMPORT STATEMENTS
# ==============================================================================
"""
Below are all the import statements used in SYNAPSE, organized by category.
Each import includes explanation of its specific use in the system.
"""

# ------------------------------------------------------------------------------
# STANDARD LIBRARY IMPORTS
# ------------------------------------------------------------------------------
import asyncio          # Asynchronous I/O - Used for parallel AI queries
import hashlib          # Secure hashing - Used for creating unique IDs
import json             # JSON parsing - Used for API responses and data storage
import logging          # Logging system - Used for debugging and monitoring
import math             # Mathematical functions - Used for calculations
import multiprocessing  # Parallel processing - Used for CPU-bound tasks
import os               # Operating system interface - Used for file operations
import pickle           # Object serialization - Used for saving/loading models
import random           # Random number generation - Used for sampling
import re               # Regular expressions - Used for text parsing
import sqlite3          # SQLite database - Used for local data storage
import sys              # System parameters - Used for system operations
import time             # Time functions - Used for timestamps and delays
import warnings         # Warning control - Used to suppress unnecessary warnings
from collections import defaultdict, deque  # Data structures for efficient operations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor  # Parallel execution
from dataclasses import dataclass, field    # Data classes for clean data structures
from datetime import datetime, timedelta    # Date and time operations
from enum import Enum, auto                 # Enumerations for states and types
from functools import lru_cache, wraps      # Function tools for optimization
from pathlib import Path                    # Object-oriented filesystem paths
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable  # Type hints
from uuid import uuid4                      # Unique identifier generation

# ------------------------------------------------------------------------------
# SCIENTIFIC COMPUTING IMPORTS
# ------------------------------------------------------------------------------
import numpy as np                          # Array operations and linear algebra
import pandas as pd                         # DataFrames for structured data

from scipy import stats                     # Statistical functions
from scipy import signal                    # Signal processing for chaos theory
from scipy import optimize                  # Optimization algorithms
from scipy.spatial import distance          # Distance metrics for similarity

# ------------------------------------------------------------------------------
# MACHINE LEARNING IMPORTS
# ------------------------------------------------------------------------------
from sklearn.decomposition import PCA, NMF  # Dimensionality reduction
from sklearn.manifold import TSNE           # t-SNE for visualization
from sklearn.cluster import DBSCAN, KMeans  # Clustering algorithms
from sklearn.preprocessing import StandardScaler  # Data normalization

# ------------------------------------------------------------------------------
# DEEP LEARNING IMPORTS (PyTorch)
# ------------------------------------------------------------------------------
import torch                                # PyTorch core
import torch.nn as nn                       # Neural network modules
import torch.nn.functional as F             # Functional operations
from torch.optim import Adam, AdamW         # Optimizers

# ------------------------------------------------------------------------------
# NLP & TRANSFORMERS IMPORTS
# ------------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModel  # Pre-trained models
from transformers import pipeline           # High-level NLP pipelines
from sentence_transformers import SentenceTransformer  # Sentence embeddings
import nltk                                 # Natural language toolkit
import spacy                               # Industrial NLP

# ------------------------------------------------------------------------------
# GRAPH PROCESSING IMPORTS
# ------------------------------------------------------------------------------
import networkx as nx                       # Graph creation and analysis
# from igraph import Graph                  # Alternative graph library (optional)

# ------------------------------------------------------------------------------
# VISUALIZATION IMPORTS
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt             # Basic plotting
from matplotlib.animation import FuncAnimation  # Animated plots
import seaborn as sns                       # Statistical visualizations
import plotly.graph_objects as go          # Interactive 3D plots
import plotly.express as px                # Quick interactive plots
from plotly.subplots import make_subplots  # Multiple plot layouts

# ------------------------------------------------------------------------------
# DATABASE & SEARCH IMPORTS
# ------------------------------------------------------------------------------
import faiss                                # Vector similarity search
import redis                                # Redis cache
from pymongo import MongoClient             # MongoDB connection

# ------------------------------------------------------------------------------
# AI API IMPORTS (Optional - only if using external services)
# ------------------------------------------------------------------------------
try:
    import anthropic                        # Claude API
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai                           # OpenAI GPT API
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ------------------------------------------------------------------------------
# ACADEMIC RESEARCH IMPORTS
# ------------------------------------------------------------------------------
import arxiv                                # arXiv paper search
try:
    from scholarly import scholarly         # Google Scholar (may be rate-limited)
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False

# ------------------------------------------------------------------------------
# WEB & ASYNC IMPORTS
# ------------------------------------------------------------------------------
import requests                             # HTTP requests
from bs4 import BeautifulSoup              # HTML parsing
import aiohttp                             # Async HTTP

# ------------------------------------------------------------------------------
# UTILITY IMPORTS
# ------------------------------------------------------------------------------
from tqdm.auto import tqdm                  # Progress bars
import rich                                 # Rich terminal output
from rich.console import Console            # Console for formatted output
from rich.table import Table               # Formatted tables
from rich.progress import Progress         # Progress tracking
from rich.panel import Panel               # Formatted panels

# ------------------------------------------------------------------------------
# SECURITY IMPORTS
# ------------------------------------------------------------------------------
from cryptography.fernet import Fernet     # Encryption
import jwt                                 # JSON Web Tokens
from dotenv import load_dotenv            # Environment variables

# ------------------------------------------------------------------------------
# IMAGE PROCESSING IMPORTS (Optional)
# ------------------------------------------------------------------------------
try:
    from PIL import Image                  # Image processing
    import cv2                             # Computer vision
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

# ------------------------------------------------------------------------------
# AUDIO PROCESSING IMPORTS (Optional)
# ------------------------------------------------------------------------------
try:
    import librosa                         # Audio analysis
    import soundfile as sf                 # Audio file I/O
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False


# ==============================================================================
# PART 3: REQUIREMENTS.TXT GENERATION
# ==============================================================================
def generate_requirements_txt():
    """
    Generate a requirements.txt file with all dependencies.
    This function creates a properly formatted requirements file.
    """
    requirements = """# SYNAPSE Requirements File
# Generated for Cognitive Augmentation Research System
# Version: 1.0.0

# Core Scientific Computing
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1

# Machine Learning
scikit-learn==1.3.0
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# NLP & Transformers
transformers==4.35.0
sentence-transformers==2.2.2
tokenizers==0.14.1
nltk==3.8.1
spacy==3.6.0

# AI APIs (optional)
anthropic==0.7.0
openai==1.3.0

# Vector Search & Databases
faiss-cpu==1.7.4
redis==4.6.0
pymongo==4.4.1
sqlalchemy==2.0.23

# Graph Processing
networkx==3.1
igraph==0.10.6

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
dash==2.11.1

# Academic Research
arxiv==1.4.8
scholarly==1.7.11
biopython==1.81

# Web & Requests
requests==2.31.0
beautifulsoup4==4.12.2
aiohttp==3.8.5

# CLI & Output
rich==13.5.2
tqdm==4.65.0
click==8.1.7

# Data Formats
pyyaml==6.0.1
jsonschema==4.19.0

# Security
cryptography==41.0.3
pyjwt==2.8.0
python-dotenv==1.0.0

# Image Processing (optional)
pillow==10.0.0
opencv-python==4.8.0.74

# Audio Processing (optional)
librosa==0.10.0
soundfile==0.11.0

# Testing
pytest==7.4.2
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Code Quality
black==23.7.0
flake8==6.1.0
mypy==1.5.1
bandit==1.7.5

# Documentation
sphinx==7.2.6
mkdocs==1.5.2
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ requirements.txt generated successfully!")


# ==============================================================================
# PART 4: INSTALLATION SCRIPT
# ==============================================================================
def install_all_packages():
    """
    Automated installation script for all SYNAPSE dependencies.
    Run this function to install everything at once.
    """
    import subprocess
    import sys
    
    print("🚀 Starting SYNAPSE dependency installation...")
    print("=" * 60)
    
    # Core packages that should be installed first
    core_packages = [
        'numpy==1.24.3',
        'pip',
        'setuptools',
        'wheel'
    ]
    
    # All other packages
    packages = [
        'pandas==2.0.3',
        'scipy==1.11.1',
        'scikit-learn==1.3.0',
        'torch==2.1.0',
        'transformers==4.35.0',
        'sentence-transformers==2.2.2',
        'networkx==3.1',
        'faiss-cpu==1.7.4',
        'plotly==5.15.0',
        'rich==13.5.2',
        'arxiv==1.4.8',
        'requests==2.31.0',
        'beautifulsoup4==4.12.2',
        'tqdm==4.65.0',
        'python-dotenv==1.0.0',
        'cryptography==41.0.3',
        'pyjwt==2.8.0'
    ]
    
    def install(package):
        """Install a single package"""
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Successfully installed: {package}")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {package}")
            return False
    
    # Install core packages first
    print("\n📦 Installing core packages...")
    for package in core_packages:
        install(package)
    
    # Install all other packages
    print("\n📦 Installing SYNAPSE dependencies...")
    failed = []
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{len(packages)}] Installing {package}...")
        if not install(package):
            failed.append(package)
    
    # Report results
    print("\n" + "=" * 60)
    if failed:
        print(f"⚠️  Installation completed with {len(failed)} failures:")
        for package in failed:
            print(f"  - {package}")
        print("\nTry installing failed packages manually:")
        print(f"pip install {' '.join(failed)}")
    else:
        print("✨ All packages installed successfully!")
        print("🎉 SYNAPSE is ready to use!")
    
    # Additional setup steps
    print("\n📝 Additional setup steps:")
    print("1. Download spaCy language model: python -m spacy download en_core_web_sm")
    print("2. Create .env file with your API keys")
    print("3. Run verification: python verify_installation.py")


# ==============================================================================
# PART 5: DEPENDENCY VERIFICATION
# ==============================================================================
def verify_installation():
    """
    Verify that all required packages are installed and functional.
    This helps diagnose installation issues.
    """
    import importlib
    
    print("🔍 Verifying SYNAPSE installation...")
    print("=" * 60)
    
    # Essential packages that must be installed
    essential = {
        'numpy': 'Scientific computing',
        'pandas': 'Data manipulation',
        'torch': 'Deep learning',
        'transformers': 'NLP models',
        'networkx': 'Graph processing',
        'plotly': 'Visualization',
        'requests': 'HTTP requests'
    }
    
    # Optional packages
    optional = {
        'anthropic': 'Claude API',
        'openai': 'OpenAI API',
        'faiss': 'Vector search',
        'redis': 'Caching',
        'scholarly': 'Google Scholar'
    }
    
    def check_package(name, description):
        """Check if a package is installed"""
        try:
            importlib.import_module(name)
            print(f"✅ {name:20} - {description}")
            return True
        except ImportError:
            print(f"❌ {name:20} - {description}")
            return False
    
    # Check essential packages
    print("\n📦 Essential Packages:")
    essential_ok = all(check_package(name, desc) for name, desc in essential.items())
    
    # Check optional packages
    print("\n📦 Optional Packages:")
    for name, desc in optional.items():
        check_package(name, desc)
    
    # Check Python version
    print(f"\n🐍 Python Version: {sys.version}")
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 9:
        print("✅ Python version is compatible")
    else:
        print("⚠️  Python 3.9+ is recommended")
    
    # Check GPU availability (for PyTorch)
    print("\n🎮 Hardware Acceleration:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  No CUDA GPU detected (CPU mode)")
    except:
        print("⚠️  Could not check GPU availability")
    
    # Final verdict
    print("\n" + "=" * 60)
    if essential_ok:
        print("✨ SYNAPSE core dependencies are installed!")
        print("🚀 You can start using the system.")
    else:
        print("⚠️  Some essential packages are missing.")
        print("Run: python -c 'from requirements import install_all_packages; install_all_packages()'")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    """
    When this file is run directly, it provides installation options.
    """
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           SYNAPSE - Requirements & Dependencies            ║
    ║                  Cognitive Augmentation System             ║
    ╚════════════════════════════════════════════════════════════╝
    
    Options:
    1. Generate requirements.txt file
    2. Install all packages automatically
    3. Verify installation
    4. Show package information
    5. Exit
    """)
    
    while True:
        choice = input("\nSelect option (1-5): ")
        
        if choice == '1':
            generate_requirements_txt()
        elif choice == '2':
            install_all_packages()
        elif choice == '3':
            verify_installation()
        elif choice == '4':
            print("\nThis file contains documentation for all SYNAPSE dependencies.")
            print("View the source code for detailed explanations of each package.")
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1-5.")
