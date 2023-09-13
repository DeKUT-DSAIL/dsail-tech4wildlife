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
else (
    rem Already installed
    echo Chocolatey is already installed.
)

rem Check if Node.js is installed
node -v >nul 2>&1
if %errorLevel% neq 0 (
    echo Node.js is not installed. Installing Node.js...
    
    rem Install Node.js version 18.x
    choco install -y nodejs --version=18.16.0 2>> %LOGFILE%
)
else (
    rem Already installed
    echo Node.js is already installed.
)


rem Check if Arduino CLI is installed
arduino-cli version >nul 2>&1
if %errorLevel% neq 0 (
    echo Arduino CLI is not installed. Installing Arduino CLI...
    
    rem Install Arduino CLI
    choco install arduino-cli -y 2>> %LOGFILE%
)
else (
    rem Already installed
    echo Arduino CLI is already installed.
)

rem Install Python3
choco install -y python --version==3.10.11 2>> %LOGFILE%

rem Check if edge-impulse-cli is installed
edge-impulse-daemon --version >nul 2>&1
if %errorLevel% neq 0 (
    echo edge-impulse-cli is not installed. Installing edge-impulse-cli...
    
    rem Install edge-impulse-cli tools
    npm install -g edge-impulse-cli  2>> %LOGFILE%
)
else (
    rem Already installed
    echo edge-impulse-cli is already installed.
)

rem check if pip is installed 
pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo pip is not installed. Installing pip...
    
    rem Install pip
    py -m ensurepip --upgrade 2>> %LOGFILE%
)
else (
    rem Already installed
    echo pip is already installed.
)

rem install virtualenvironment
echo Installing virtual environment...
pip install virtualenv 2>> %LOGFILE%

rem create virtual environment
echo Creating virtual environment...
virtualenv dsail-tech4wildlife 2>> %LOGFILE%

rem activate virtual environment
echo Activating virtual environment...
.\dsail-tech4wildlife\Scripts\activate

rem install requirements
echo Installing requirements...
pip install -r requirements.txt 2>> %LOGFILE%


rem Optional: Notify the user about completion
echo Installation completed successfully.
echo Check %LOGFILE% for any installation errors.
pause
exit /b 0
