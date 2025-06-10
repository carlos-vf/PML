#!/bin/bash

# Exit immediately if any command fails
set -e

# Define paths
# VENV_PATH="venv/bin/activate"
CONFIG_DIR="configs"
CONFIGS=("iris1.yaml" "iris2.yaml" "iris3.yaml")

# Check if virtual environment exists
# if [ ! -f "$VENV_PATH" ]; then
#     echo "Error: Virtual environment not found at $VENV_PATH"
#     exit 1
# fi

# Activate virtual environment
# source "$VENV_PATH"

# Loop through each config file
for config in "${CONFIGS[@]}"; do
    CONFIG_PATH="$CONFIG_DIR/$config"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Config file not found at $CONFIG_PATH"
        continue  # Skip to next config instead of exiting
    fi

    echo -e "\nStarting experiment with $config..."
    python3 -m scripts.run_experiment --config "$CONFIG_PATH"
    echo "Completed experiment with $config"
done

# Deactivate virtual environment
# deactivate

echo -e "\nAll experiments completed successfully!"