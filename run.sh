#!/bin/bash

# This script runs a series of experiments defined by YAML configuration files.
# It accepts one or more config filenames as command-line arguments.

# Exit immediately if any command fails
set -e

# Define paths
CONFIG_DIR="configs"

# Check if at least one config file was provided
if [ "$#" -eq 0 ]; then
    echo "Usage: ./run_experiments.sh <config1.yaml> [<config2.yaml> ...]"
    exit 1
fi

# Optional: Activate a virtual environment if you use one
# source venv/bin/activate

# Loop through each config file passed as an argument
for config in "$@"
do
    CONFIG_PATH="$CONFIG_DIR/$config"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Config file not found at $CONFIG_PATH"
        continue # Skip to the next config
    fi

    echo ""
    echo "--- Starting experiment with $config ---"
    
    # Execute the Python experiment pipeline
    # The 'set -e' at the top will cause the script to exit if this command fails
    python -m scripts.run_experiment --config "$CONFIG_PATH"
    
    echo "--- Completed experiment with $config ---"
done

# Optional: Deactivate the virtual environment
# deactivate

echo ""
echo "All experiments completed successfully!"
