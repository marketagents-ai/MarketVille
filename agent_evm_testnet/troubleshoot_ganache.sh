#!/bin/bash

fix_node_version() {
    echo "=== Installing Compatible Node.js Version ==="
    
    # Remove existing ganache installation
    echo "1. Removing existing Ganache installation..."
    sudo npm uninstall -g ganache
    
    # Install Node Version Manager (nvm)
    echo "2. Installing nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    
    # Load nvm
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    # Install and use Node.js LTS version (known to work with Ganache)
    echo "3. Installing Node.js LTS version..."
    nvm install 18.18.0
    nvm use 18.18.0
    
    # Verify Node.js version
    echo "4. Verifying Node.js version..."
    node -v
    
    # Install Ganache globally
    echo "5. Installing Ganache..."
    npm install -g ganache
    
    # Verify Ganache installation
    echo "6. Verifying Ganache installation..."
    ganache --version
    
    # Add nvm setup to shell configuration
    SHELL_CONFIG="$HOME/.bashrc"
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi
    
    echo "7. Adding nvm configuration to $SHELL_CONFIG..."
    cat << 'EOF' >> "$SHELL_CONFIG"

# NVM Configuration
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
EOF
}

fix_node_version