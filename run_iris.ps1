# Exit immediately if any command fails
$ErrorActionPreference = "Stop"

# Define paths
$CONFIG_DIR = "configs"
$CONFIGS = @("mnist1.yaml", "mnist2.yaml", "mnist3.yaml")

# Check if virtual environment exists (optional - uncomment if needed)
# $VENV_ACTIVATE_SCRIPT = "venv/Scripts/Activate.ps1"
# if (-not (Test-Path $VENV_ACTIVATE_SCRIPT)) {
#     Write-Host "Error: Virtual environment not found at $VENV_ACTIVATE_SCRIPT"
#     exit 1
# }

# Activate virtual environment (optional - uncomment if needed)
# . $VENV_ACTIVATE_SCRIPT # The '.' or 'source' command in PowerShell

# Loop through each config file
foreach ($config in $CONFIGS) {
    $CONFIG_PATH = Join-Path $CONFIG_DIR $config
    
    # Check if config file exists
    if (-not (Test-Path $CONFIG_PATH)) {
        Write-Host "Error: Config file not found at $CONFIG_PATH"
        continue # Skip to next config instead of exiting
    }

    Write-Host "`nStarting experiment with $config..."
    # Using 'python' instead of 'python3' as 'python' is the standard executable name on Windows
    python -m scripts.run_experiment --config "$CONFIG_PATH"
    
    # Check if the last command was successful (ErrorLevel 0)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running experiment with $config. Exiting."
        exit 1
    }
    Write-Host "Completed experiment with $config"
}

# Deactivate virtual environment (optional - uncomment if needed)
# if ($env:VIRTUAL_ENV) {
#     deactivate
# }

Write-Host "`nAll experiments completed successfully!"