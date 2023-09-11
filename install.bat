@echo off
rem Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Administrator privileges are required to run this script.
    goto :eof
)

rem Install Chocolatey
@powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

rem Install Python3
choco install -y python

rem Install Node.js
choco install -y nodejs

rem Install edge-impulse-cli tools
npm install -g edge-impulse-cli

rem Install Arduino CLI
@powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/arduino/arduino-cli/master/install.ps1'))"

rem Optional: Notify the user about completion
echo Installation completed successfully.

exit /b 0
