# sMeetSec first-start setup (needs internet ONCE). Run by sMeetSec.exe or setup_windows.bat.
# Installs a private Python (embeddable, inside runtime\python - nothing is installed system-wide),
# the Python packages, NVIDIA libraries when an NVIDIA GPU exists, and the speech / speaker models.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$rt = Join-Path $root 'runtime'
$py = Join-Path $rt 'python'
$pyexe = Join-Path $py 'python.exe'
$ver = '3.12.10'

function Step($n, $text) { Write-Output "STEP $n/5: $text"; [Console]::Out.Flush() }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

New-Item -ItemType Directory -Force -Path $rt | Out-Null
Set-Location $root

if (-not (Test-Path $pyexe)) {
    Step 1 "Downloading Python $ver (private copy, about 11 MB)"
    $zip = Join-Path $rt 'python-embed.zip'
    Invoke-WebRequest "https://www.python.org/ftp/python/$ver/python-$ver-embed-amd64.zip" -OutFile $zip -UseBasicParsing
    Expand-Archive $zip -DestinationPath $py -Force
    Remove-Item $zip
    # enable site-packages and make the sMeetSec folder importable
    $pth = Get-ChildItem $py -Filter 'python*._pth' | Select-Object -First 1
    $zipname = (Get-ChildItem $py -Filter 'python3*.zip' | Select-Object -First 1).Name
    Set-Content -Path $pth.FullName -Encoding ASCII -Value @($zipname, '.', 'Lib\site-packages', '..\..', 'import site')
    Step 1 'Installing pip'
    Invoke-WebRequest 'https://bootstrap.pypa.io/get-pip.py' -OutFile (Join-Path $rt 'get-pip.py') -UseBasicParsing
    & $pyexe (Join-Path $rt 'get-pip.py') --no-warn-script-location
    Check 'pip install'
    Remove-Item (Join-Path $rt 'get-pip.py')
} else {
    Step 1 'Python already installed'
}

Step 2 'Installing packages (about 300 MB) - this takes a few minutes'
& $pyexe -m pip install --upgrade --no-warn-script-location -r (Join-Path $root 'requirements.txt')
Check 'package install'

Step 3 'Checking for an NVIDIA GPU'
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    Write-Output 'NVIDIA GPU found - installing CUDA libraries (about 1 GB)'
    & $pyexe -m pip install --no-warn-script-location nvidia-cublas-cu12 'nvidia-cudnn-cu12==9.*'
    Check 'CUDA libraries'
} else {
    Write-Output 'No NVIDIA GPU - CPU mode'
}

Step 4 'Downloading speech and speaker models (about 1-2 GB)'
& $pyexe (Join-Path $root 'tools\download_models.py')
Check 'model download'

Step 5 'Done'
Set-Content -Path (Join-Path $rt 'ready.txt') -Value (Get-Date -Format s)
Write-Output 'SETUP OK'
