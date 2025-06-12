# This script runs a series of experiments defined by YAML configuration files.
# It now accepts a list of config files as a command-line argument.

# Define the script parameters
param (
    # The -Configs parameter is a mandatory list of strings.
    [Parameter(Mandatory=$true)]
    [string[]]$Configs
)

# Exit immediately if any command fails
$ErrorActionPreference = "Stop"

# Define paths
$CONFIG_DIR = "configs"

# Loop through each config file passed as an argument
foreach ($config in $Configs) {
    $CONFIG_PATH = Join-Path $CONFIG_DIR $config
    
    # Check if config file exists
    if (-not (Test-Path $CONFIG_PATH)) {
        Write-Host "Error: Config file not found at $CONFIG_PATH"
        continue # Skip to next config instead of exiting
    }

    Write-Host "`nStarting experiment with $config..."
    
    # Execute the Python experiment pipeline
    python -m scripts.run_experiment --config "$CONFIG_PATH"
    
    # Check if the last command was successful
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running experiment with $config. Exiting."
        exit 1
    }
    Write-Host "Completed experiment with $config"
}

Write-Host "`nAll experiments completed successfully!"