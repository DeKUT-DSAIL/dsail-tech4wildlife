@echo off
rem Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Administrator privileges are required to run this script.
    pause
    goto :eof
)

rem Define the log file path
set LOGFILE=install_log.txt

rem Create or clear the log file
type nul > %LOGFILE%
echo Log file created at %LOGFILE%

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

rem Check if Arduino CLI is installed
arduino-cli version >nul 2>&1
if %errorLevel% neq 0 (
    echo Arduino CLI is not installed. Installing Arduino CLI...
    
    rem Install Arduino CLI
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/arduino/arduino-cli/master/install.ps1'))" 2>> %LOGFILE%
    
    rem Add Arduino CLI to PATH
    setx PATH "%PATH%;%USERPROFILE%\arduino-cli\bin" 2>> %LOGFILE%
)

rem Install Python3
choco install -y python 2>> %LOGFILE%

rem Install edge-impulse-cli tools
npm install -g edge-impulse-cli --force 2>> %LOGFILE%

rem Optional: Notify the user about completion
echo Installation completed successfully.
echo Check %LOGFILE% for any installation errors.
pause
exit /b 0
