#!/bin/bash

# Install system dependencies
apk add --no-cache libpango1.0-dev pkg-config python3-dev

# Install Python dependencies
python3 -m pip install --no-cache-dir -r requirements.txt