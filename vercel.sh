#!/bin/bash

# Install system dependencies
bash setup.sh

# Install Python dependencies
python3 -m pip install --no-cache-dir -r requirements.txt

# Run your application
python3 index.py