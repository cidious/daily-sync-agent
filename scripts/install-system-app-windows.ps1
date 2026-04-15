# PowerShell installation script for daily-sync-agent on Windows

param(
    [switch]$WithDeps = $false,
    [switch]$Uninstall = $false,
    [switch]$Help = $false
)

$AppName = "daily-sync-agent"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$InstallPath = "C:\Program Files\$AppName"
$VenvPath = "$InstallPath\.venv"
$DesktopPath = "$env:ProgramData\Microsoft\Windows\Start Menu\Programs"
$LauncherPath = "$env:LOCALAPPDATA\$AppName\launcher.bat"

function Show-Help {
    Write-Host @"
Usage: .\install-system-app-windows.ps1 [options]

Options:
  -WithDeps       Install system dependencies via winget/choco (ffmpeg, etc.)
  -Uninstall      Remove the installed app
  -Help           Show this help message

Examples:
  .\install-system-app-windows.ps1 -WithDeps
  .\install-system-app-windows.ps1 -Uninstall
"@
}

function Check-Admin {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Host "ERROR: This script requires administrator privileges." -ForegroundColor Red
        Write-Host "Please run PowerShell as Administrator and try again."
        exit 1
    }
}

function Install-SystemDeps {
    Write-Host "Checking system dependencies..." -ForegroundColor Cyan

    # Check for ffmpeg
    $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $ffmpeg) {
        Write-Host "ffmpeg not found. Installing via winget..." -ForegroundColor Yellow
        winget install -e --id Gyan.FFmpeg 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: Could not install ffmpeg. Please install manually from https://ffmpeg.org/"
        }
    } else {
        Write-Host "ffmpeg already installed: $($ffmpeg.Path)" -ForegroundColor Green
    }
}

function Create-Venv {
    param([string]$Path)

    Write-Host "Creating virtual environment at $Path..." -ForegroundColor Cyan
    python -m venv $Path
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

function Install-Package {
    param([string]$VenvPath, [string]$SourcePath)

    Write-Host "Installing daily-sync-agent into venv..." -ForegroundColor Cyan
    $PipExe = "$VenvPath\Scripts\pip.exe"
    & $PipExe install -e $SourcePath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install package" -ForegroundColor Red
        exit 1
    }
    & $PipExe install -e "$SourcePath[windows]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install package" -ForegroundColor Red
        exit 1
    }
}

function Create-Launcher {
    param([string]$VenvPath, [string]$LauncherPath)

    Write-Host "Creating launcher..." -ForegroundColor Cyan
    $LauncherDir = Split-Path -Parent $LauncherPath
    New-Item -ItemType Directory -Force -Path $LauncherDir | Out-Null

    $LauncherContent = @"
@echo off
REM Launcher for daily-sync-agent
"$VenvPath\Scripts\daily-sync-agent.exe" %*
"@

    $LauncherContent | Out-File -FilePath $LauncherPath -Encoding ASCII
    Write-Host "Launcher created at $LauncherPath" -ForegroundColor Green
}

function Create-StartMenuShortcut {
    param([string]$LauncherPath, [string]$DesktopPath)

    Write-Host "Creating Start Menu shortcut..." -ForegroundColor Cyan
    $DesktopDir = "$DesktopPath\daily-sync-agent"
    New-Item -ItemType Directory -Force -Path $DesktopDir | Out-Null

    # Create a .lnk shortcut using COM
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("$DesktopDir\daily-sync-agent.lnk")
    $Shortcut.TargetPath = $LauncherPath
    $Shortcut.WorkingDirectory = $env:USERPROFILE
    $Shortcut.Description = "Daily Sync Agent - Record & Transcribe"
    $Shortcut.Save()

    Write-Host "Shortcut created at $DesktopDir\daily-sync-agent.lnk" -ForegroundColor Green
}

function Uninstall-App {
    Write-Host "Uninstalling $AppName..." -ForegroundColor Yellow

    if (Test-Path $InstallPath) {
        Remove-Item -Recurse -Force $InstallPath
        Write-Host "Removed $InstallPath" -ForegroundColor Green
    }

    if (Test-Path $LauncherPath) {
        Remove-Item -Force $LauncherPath
        Write-Host "Removed $LauncherPath" -ForegroundColor Green
    }

    $ShortcutPath = "$DesktopPath\daily-sync-agent\daily-sync-agent.lnk"
    if (Test-Path $ShortcutPath) {
        Remove-Item -Recurse -Force (Split-Path -Parent $ShortcutPath)
        Write-Host "Removed Start Menu shortcuts" -ForegroundColor Green
    }

    Write-Host "Uninstallation complete." -ForegroundColor Green
}

# Main
if ($Help) {
    Show-Help
    exit 0
}

if ($Uninstall) {
    Check-Admin
    Uninstall-App
    exit 0
}

Check-Admin

if ($WithDeps) {
    Install-SystemDeps
}

Write-Host "Installing $AppName to $InstallPath" -ForegroundColor Cyan

# Copy repository
Write-Host "Copying repository..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $InstallPath | Out-Null
Copy-Item -Path "$RepoRoot\src" -Destination "$InstallPath" -Recurse -Force
Copy-Item -Path "$RepoRoot\pyproject.toml" -Destination "$InstallPath" -Force
Copy-Item -Path "$RepoRoot\README.md" -Destination "$InstallPath" -Force -ErrorAction SilentlyContinue

# Create venv and install
Create-Venv $VenvPath
Install-Package $VenvPath $InstallPath

# Create launcher and shortcuts
Create-Launcher $VenvPath $LauncherPath
Create-StartMenuShortcut $LauncherPath $DesktopPath

# Add to PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$LauncherDir = Split-Path -Parent $LauncherPath
if ($UserPath -notlike "*$LauncherDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$LauncherDir", "User")
    Write-Host "Added launcher directory to PATH" -ForegroundColor Green
}

Write-Host "`nInstallation complete!`n" -ForegroundColor Green
Write-Host "You can now run: daily-sync-agent" -ForegroundColor Cyan
Write-Host "Or use the Start Menu shortcut." -ForegroundColor Cyan

