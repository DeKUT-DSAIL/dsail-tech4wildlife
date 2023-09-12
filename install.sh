#!/bin/bash

# Check for sudo privileges
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo."
    exit
fi

# Install python3
apt-get install -y python3


# check if node is installed
if ! [ -x "$(command -v node)" ]; then
    echo "node is not installed. Installing node..."
    # Install nodejs
    curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    apt-get install -y nodejs
    node -v
else
    echo "node is already installed."
fi

# Verify the installation and if it returns /usr/local/ export to PATH
npm_config_prefix=$(npm config get prefix)
if [ "$npm_config_prefix" != "/usr/local" ]; then
    echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# check if edge-impulse-cli is installed
if ! [ -x "$(command -v edge-impulse-daemon)" ]; then
    echo "edge-impulse-cli is not installed. Installing edge-impulse-cli..."
    # Install edge-impulse-cli tools
    npm install -g edge-impulse-cli 
else
    echo "edge-impulse-cli is already installed."
fi


# check if snap is installed
if ! [ -x "$(command -v snap)" ]; then
    echo "snap is not installed. Installing snap..."
    apt-get install -y snapd
fi


# Check if arduino-cli is installed
if ! [ -x "$(command -v arduino-cli)" ]; then
    echo "arduino-cli is not installed. Installing arduino-cli..."
    # Install arduino-cli
    snap install arduino-cli
else
    echo "arduino-cli is already installed."
fi

# install a virtual environment
echo "Installing virtual environment..."
apt-get install -y python3-venv

# create a virtual environment
echo "Creating virtual environment..."
python3 -m venv dsail-tech4wildlife

# activate the virtual environment
echo "Activating virtual environment..."
source dsail-tech4wildlife/bin/activate

# install the requirements
pip install -r requirements.txt


# Optional: Notify user about completion
echo "Installation completed successfully."
