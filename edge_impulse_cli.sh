#!/bin/bash


# Check for sudo privileges
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo."
    exit 1
fi


# Check if Python 3.10 is already installed
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Installing it..."
    apt update
    apt install -y python3.10
else
    echo "Python3 is already installed."
fi


# Check if Node.js is installed
if ! [ -x "$(command -v node)" ]; then
    echo "Node.js is not installed. Installing Node.js..."
    # Install Node.js
    curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    apt-get install -y nodejs
else
    echo "Node.js is already installed."
fi

# Check the installed Node.js version
node_version=$(node -v | cut -d "v" -f 2)
desired_version="18"

# Compare the Node.js version
if [ "$(printf '%s\n' "$desired_version" "$node_version" | sort -V | head -n 1)" != "$desired_version" ]; then
    echo "Node.js version is below $desired_version. Installing the desired version..."
    
    # Reinstall Node.js version 18.x
    curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    apt-get install -y nodejs
else
    echo "Node.js version is $desired_version or higher."
fi

# Verify the installation and update PATH
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

