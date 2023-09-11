@echo off
rem Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Administrator privileges are required to run this script.
    pause
    goto :eof
)

rem Create a log file
set LOGFILE=install_log.txt
echo. > %LOGFILE%

rem Check if Chocolatey is installed
choco --version >nul 2>&1
if %errorLevel% neq 0 (
    echo Chocolatey is not installed. Installing Chocolatey...
    
    rem Install Chocolatey
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" 2>> %LOGFILE%
    
    rem Restart the script after Chocolatey installation
    start cmd /k %0
    exit /b 0
)

rem Check if Node.js is installed
node -v >nul 2>&1
if %errorLevel% neq 0 (
    echo Node.js is not installed. Installing Node.js...
    
    rem Install Node.js
    choco install -y nodejs 2>> %LOGFILE%
)
rem Install Python3
choco install -y python 2>> %LOGFILE%

rem Install edge-impulse-cli tools
npm install -g edge-impulse-cli 2>> %LOGFILE%

rem Install Arduino CLI
@powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/arduino/arduino-cli/master/install.ps1'))" 2>> %LOGFILE%

rem Optional: Notify the user about completion
echo Installation completed successfully.
echo Check %LOGFILE% for any installation errors.
pause
exit /b 0
