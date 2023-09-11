#!/bin/bash

# Check for sudo privileges
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo."
    exit
fi

# Install python3
apt-get install -y python3

# Install nodejs
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
apt-get install -y nodejs
node -v

# Verify the installation and if it returns /usr/local/ export to PATH
npm_config_prefix=$(npm config get prefix)
if [ "$npm_config_prefix" != "/usr/local" ]; then
    echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

# Install edge-impulse-cli tools
npm install -g edge-impulse-cli

# Install arduino-cli
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh -s 0.11.0

# Export arduino-cli to PATH
arduino_cli_path="$HOME/.arduino15/packages/arduino/tools/arduino-cli/0.11.0/"
if [[ ":$PATH:" != *":$arduino_cli_path:"* ]]; then
    echo 'export PATH="$PATH:'"$arduino_cli_path"'"' >> ~/.bashrc
    source ~/.bashrc
fi

# Optional: Notify user about completion
echo "Installation completed successfully."
