# Stop script execution on any error
$ErrorActionPreference = "Stop"

# Define paths
# Note: In PowerShell, the venv activation script is typically a .ps1 file in the 'Scripts' folder.
# $VenvPath = "venv\Scripts\activate.ps1"

$ConfigDir = "configs_grid"
$Configs = @(
    "wdbc1.yaml", 
    "wdbc2.yaml", 
    "wdbc3.yaml",
    "iris1.yaml",
    "iris2.yaml",
    "iris3.yaml",
    "phoneme1.yaml",
    "phoneme2.yaml",
    "phoneme3.yaml"
)

# Check if virtual environment exists
# if (-not (Test-Path -Path $VenvPath -PathType Leaf)) {
#   Write-Host "Error: Virtual environment not found at $VenvPath" -ForegroundColor Red
#   exit 1
# }

# Activate virtual environment
# . $VenvPath

# Loop through each config file
foreach ($config in $Configs) {
    # Use Join-Path for robustly creating file paths
    $ConfigPath = Join-Path -Path $ConfigDir -ChildPath $config
    
    # Check if config file exists
    if (-not (Test-Path -Path $ConfigPath -PathType Leaf)) {
        Write-Host "Error: Config file not found at $ConfigPath" -ForegroundColor Red
        continue  # Skip to next config instead of exiting
    }

    Write-Host "`nStarting experiment with $config..." -ForegroundColor Cyan
    # Assuming 'python3' is in the system's PATH. You can also use 'python'.
    py -m scripts.run_grid --config "$ConfigPath"
    Write-Host "Completed experiment with $config" -ForegroundColor Green
}

# Deactivate virtual environment
# The 'deactivate' function is available after sourcing the activation script.
# deactivate

Write-Host "`nAll experiments completed successfully!" -ForegroundColor Green