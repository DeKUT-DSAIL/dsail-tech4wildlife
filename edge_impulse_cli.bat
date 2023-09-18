@echo off
rem Check for admin privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Administrator privileges are required to run this script.
    pause
    goto :eof
)




rem Define the log file path with a full path
set LOGFILE=%~dp0install_log.txt


rem Create or clear the log file
type nul > %LOGFILE%
echo Log file created at %LOGFILE%


rem Check if Chocolatey is installed
choco --version >nul 2>&1
if %errorLevel% neq 0 (
    echo Chocolatey is not installed. Installing Chocolatey...
   
    rem Install Chocolatey
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" 2>> %LOGFILE%
)


rem Already installed
echo Chocolatey is already installed.

rem Check if Node.js is installed
node -v >nul 2>&1
if %errorLevel% neq 0 (
    echo Node.js is not installed. Installing Node.js...
   
    rem Install Node.js version 18.16.0
    choco install -y nodejs --version=18.16.0 2>> %LOGFILE%
) else (
    rem Node.js is installed; check the version
    for /f "tokens=1,2,3" %%a in ('node -v') do (
        set node_version=%%a
    )


    rem Check if node_version is 18 or higher
    if "%node_version%" geq "v18" (
        echo Node.js version is 18 or higher: %node_version%
    ) else (
        choco install -y nodejs --version=18.16.0  2>> %LOGFILE%
    )
)

rem check if python3.10 is installed
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo python3 is not installed. Installing python3.10...
   
    rem Install Python3
    choco install -y python --version=3.10.11 2>> %LOGFILE%
)
rem python3.10 Already installed
echo python3.10 is already installed.

rem install visualstudio2022-workload-vctools if raises an error
choco install -y visualstudio2022-workload-vctools  >nul 2>&1
if %errorLevel% neq 0 (
    echo visualstudio2022-workload-vctools is not installed. Installing visualstudio2022-workload-vctools...
   
    rem Install visualstudio2022-workload-vctools
    choco install -y visualstudio2022-workload-vctools 2>> %LOGFILE%
)
rem visualstudio2022-workload-vctools Already installed
echo visualstudio2022-workload-vctools is already installed.

rem Check if edge-impulse-cli is installed
edge-impulse-daemon --version >nul 2>&1
if %errorLevel% neq 0 (
    echo edge-impulse-cli is not installed. Installing edge-impulse-cli...
   
    rem Install edge-impulse-cli tools
    npm install -g edge-impulse-cli  2>> %LOGFILE%
)


rem edge-impulse-cli Already installed
echo edge-impulse-cli is already installed.


