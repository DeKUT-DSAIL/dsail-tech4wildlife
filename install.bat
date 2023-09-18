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
   
    rem Install Node.js version 18.x
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
        choco install -y nodejs --version=18.16.0 --force 2>> %LOGFILE%
    )
)


rem Check if Arduino CLI is installed
arduino-cli version >nul 2>&1
if %errorLevel% neq 0 (
    echo Arduino CLI is not installed. Installing Arduino CLI...
    
    rem Install Arduino CLI
    choco install arduino-cli -y 2>> %LOGFILE%
)

rem Already installed
echo Arduino CLI is already installed.


rem check if python3.10 is installed
choco install -y python --version=3.10.11 --force >nul 2>&1
if %errorLevel% neq 0 (
    echo python3.10 is not installed. Installing python3.10...
    
    rem Install Python3
    choco install -y python --version=3.10.11 --force 2>> %LOGFILE%
)
rem python3.10 Already installed
echo python3.10 is already installed.


rem install visualstudio2019buildtools if raises an error
choco install -y visualstudio2019buildtools >nul 2>&1
if %errorLevel% neq 0 (
    echo visualstudio2019buildtools is not installed. Installing visualstudio2019buildtools...
    
    rem Install visualstudio2019buildtools
    choco install -y visualstudio2019buildtools 2>> %LOGFILE%
)
rem visualstudio2019buildtools Already installed
echo visualstudio2019buildtools is already installed.

rem install visualstudio2022buildtools if raises an error
choco install -y visualstudio2022buildtools >nul 2>&1
if %errorLevel% neq 0 (
    echo visualstudio2022buildtools is not installed. Installing visualstudio2022buildtools...
    
    rem Install visualstudio2022buildtools
    choco install -y visualstudio2022buildtools 2>> %LOGFILE%
)
rem visualstudio2022buildtools Already installed
echo visualstudio2022buildtools is already installed.

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


rem check if pip is installed 
pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo pip is not installed. Installing pip...
    
    rem Install pip
    py -m ensurepip --upgrade 2>> %LOGFILE%
)

rem pip is Already installed
echo pip is already installed.


rem install virtualenvironment
echo Installing virtual environment...
pip install virtualenv 2>> %LOGFILE%

rem create virtual environment
echo Creating virtual environment...
:: Navigate to the current working directory
cd /d %CD%
rem Create a virtual environment named dsail-tech4wildlife at the current working directory
virtualenv %~dp0dsail-tech4wildlife

rem activate virtual environment
echo Activating virtual environment...
%~dp0dsail-tech4wildlife\Scripts\activate

rem install requirements
echo Installing requirements...
pip install -r requirements.txt 2>> %LOGFILE%


rem Optional: Notify the user about completion
echo Installation completed successfully.
echo Check %LOGFILE% for any installation errors.
pause
exit /b 0
