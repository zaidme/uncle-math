#!/bin/bash

# Run the setup script
bash setup.sh

# Build the Next.js application
npm run build

# Start the FastAPI application
python3 api/index.py