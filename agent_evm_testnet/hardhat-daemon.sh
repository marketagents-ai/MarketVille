#!/bin/bash
# Hardhat Testnet Manager Script
# Usage: ./hardhat-daemon.sh [start|stop|status|reset]

# Set script variables
PROJECT_DIR="./hardhat-testnet"
PID_FILE="./hardhat-node.pid"
LOG_FILE="./hardhat-node.log"
PORT=8545  # Default Hardhat port
MNEMONIC="candy maple cake sugar pudding cream honey rich smooth crumble sweet treat"

# save mnemonic to file
echo $MNEMONIC > ../.mnemonic

# Embedded Hardhat configuration
HARDHAT_CONFIG='require("@nomicfoundation/hardhat-toolbox");
require("@openzeppelin/hardhat-upgrades");
module.exports = {
 solidity: {
 version: "0.8.20",
 settings: {
 optimizer: {
 enabled: true,
 runs: 200
 }
 }
 },
 networks: {
 hardhat: {
 allowUnlimitedContractSize: true,
 accounts: {
   mnemonic: "'$MNEMONIC'"
 }
 }
 },
 paths: {
 sources: "./contracts",
 tests: "./test",
 cache: "./cache",
 artifacts: "./artifacts"
 }
};'

# Function to check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1 ;
}

# Function to check if a port is in use
check_port() {
    netstat -tuln | grep ":$PORT " >/dev/null 2>&1
}

# Function to verify if hardhat is actually running
verify_hardhat_running() {
    local pid=$1
    # Check if process exists
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        # Check if it's actually a Hardhat process
        if ps -p "$pid" -o command= | grep -q "hardhat"; then
            # Check if the port is active
            if check_port; then
                return 0  # All checks passed
            fi
        fi
    fi
    return 1  # Process not running properly
}

# Function to find hardhat process
find_hardhat_process() {
    pgrep -f "node.*hardhat.*node" | grep -v "$$" | head -n 1
}

# Function to find and kill orphaned Hardhat processes
kill_orphaned_processes () {
    echo "Checking for orphaned Hardhat processes..."
    # Find all node processes running hardhat
    orphaned_pids=$(pgrep -f "node.*hardhat.*node" | grep -v "$$")
    
    if [ -n "$orphaned_pids" ]; then
        echo "Found orphaned Hardhat processes with PIDs: $orphaned_pids"
        for pid in $orphaned_pids; do
            echo "Killing process $pid..."
            kill -9 "$pid" 2>/dev/null
        done
        echo "Orphaned processes have been terminated."
        # Wait a moment to ensure processes are fully terminated
        sleep 2
    else
        echo "No orphaned Hardhat processes found."
    fi
    
    # Clean up stale PID file if it exists
    if [ -f "$PID_FILE" ]; then
        if ! verify_hardhat_running "$(cat "$PID_FILE")"; then
            echo "Removing stale PID file..."
            rm "$PID_FILE"
        fi
    fi
}

# Function to increase file watch limit
increase_watch_limit () {
    echo "Checking and adjusting file watch limit..."
    current_limit=$(cat /proc/sys/fs/inotify/max_user_watches)
    desired_limit=524288
    
    if [ "$current_limit" -lt "$desired_limit" ]; then
        echo "Current watch limit ($current_limit) is too low. Increasing to $desired_limit..."
        sudo sysctl -w fs.inotify.max_user_watches=$desired_limit
        if ! grep -q "fs.inotify.max_user_watches" /etc/sysctl.conf; then
            echo "fs.inotify.max_user_watches=$desired_limit" | sudo tee -a /etc/sysctl.conf
        else
            sudo sed -i "s/fs.inotify.max_user_watches=.*/fs.inotify.max_user_watches=$desired_limit/" /etc/sysctl.conf
        fi
        sudo sysctl -p
        echo "Watch limit increased successfully."
    else
        echo "Watch limit is already sufficient ($current_limit)."
    fi
}

# Function to wait for Hardhat to be fully running
wait_for_hardhat() {
    local pid=$1
    local timeout=30
    local counter=0
    
    echo "Waiting for Hardhat to start..."
    while [ $counter -lt $timeout ]; do
        if verify_hardhat_running "$pid" && curl -s http://localhost:$PORT >/dev/null 2>&1; then
            echo "Hardhat is now running and responding."
            return 0
        fi
        sleep 1
        counter=$((counter + 1))
    done
    return 1
}

# Function to install Node.js if not installed
install_node () {
    if ! command_exists node ; then
        echo "Node.js not found. Installing Node.js..."
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    else
        echo "Node.js is already installed."
    fi
}

# Function to install dependencies
install_dependencies () {
    echo "Installing necessary packages..."
    sudo apt-get update
    sudo apt-get install -y curl build-essential
    #npm install @nomiclabs/hardhat-ethers ethers
    echo "Installing additional Python packages..."
    # TODO: move these
    pip install web3 eth-account requests py-solc-x


}

# Function to set up Hardhat project
setup_hardhat_project () {
    if [ ! -d "$PROJECT_DIR" ]; then
        echo "Setting up Hardhat project..."
        mkdir -p "$PROJECT_DIR"

        # copy ./contracts to the project directory
        cp -r ./contracts "$PROJECT_DIR"

        cd "$PROJECT_DIR"
        
        # Initialize fresh package.json
        echo "Initializing package.json..."
        npm init -y
        
        # Clear npm cache first to ensure clean installs
        #echo "Clearing npm cache..."
        #npm cache clean --force
        
        # Install core dependencies first
        echo "Installing core dependencies..."
        npm install --save-dev hardhat@latest
        npm install --save-dev @nomicfoundation/hardhat-toolbox@latest
        
        # Install OpenZeppelin contracts and wait for completion
        echo "Installing OpenZeppelin contracts..."
        npm install --save-dev @openzeppelin/contracts
        
        # Verify OpenZeppelin installation
        if [ ! -d "node_modules/@openzeppelin/contracts" ]; then
            echo "Error: OpenZeppelin contracts not installed properly"
            exit 1
        fi
        
        # Install OpenZeppelin upgrades
        echo "Installing OpenZeppelin upgrades..."
        npm install --save-dev @openzeppelin/hardhat-upgrades@latest
        npm install --save-dev @openzeppelin/contracts-upgradeable
        
        # Create Hardhat configuration file
        echo "$HARDHAT_CONFIG" > hardhat.config.js
        
        echo "Compiling contracts..."
        # Clean cache first
        rm -rf cache artifacts
        npx hardhat clean
        npx hardhat compile
        
        echo "Setup complete."
        cd ..
    else
        echo "Hardhat project already exists."
    fi
}

# Function to start Hardhat testnet
start_hardhat_node () {
    # Kill any orphaned processes before starting
    kill_orphaned_processes
    
    # Increase file watch limit before starting
    increase_watch_limit
    
    if [ -f "$PID_FILE" ] && verify_hardhat_running "$(cat "$PID_FILE")"; then
        echo "Hardhat testnet is already running."
    else
        echo "Starting Hardhat testnet..."
        cd "$PROJECT_DIR"
        nohup npx hardhat node --hostname 0.0.0.0 > "$LOG_FILE" 2>&1 &
        new_pid=$!
        echo $new_pid > "$PID_FILE"
        
        if wait_for_hardhat "$new_pid"; then
            echo "Hardhat testnet started successfully with PID $new_pid"
        else
            echo "Failed to start Hardhat testnet. Check the log file at $LOG_FILE for details."
            kill -9 "$new_pid" 2>/dev/null
            rm "$PID_FILE"
            exit 1
        fi
    fi
}

# Function to stop Hardhat testnet
stop_hardhat_node () {
    # First kill any orphaned processes
    kill_orphaned_processes
    
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if verify_hardhat_running "$pid"; then
            echo "Stopping Hardhat testnet..."
            kill -TERM "$pid"
            # Wait for process to actually stop
            for i in {1..10}; do
                if ! verify_hardhat_running "$pid"; then
                    break
                fi
                sleep 1
            done
            # Force kill if still running
            if verify_hardhat_running "$pid"; then
                kill -9 "$pid"
            fi
            rm "$PID_FILE"
            echo "Hardhat testnet stopped."
        else
            echo "Hardhat testnet was not running properly."
            rm "$PID_FILE"
        fi
    else
        echo "No PID file found."
    fi
}

# Function to check the status of Hardhat testnet
status_hardhat_node () {
    local running=false
    local pid=""
    
    # First check PID file
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if verify_hardhat_running "$pid"; then
            running=true
        fi
    fi
    
    # If not found in PID file, try to find running process
    if [ "$running" = false ]; then
        pid=$(find_hardhat_process)
        if [ -n "$pid" ] && verify_hardhat_running "$pid"; then
            running=true
            echo "$pid" > "$PID_FILE"
        fi
    fi
    
    if [ "$running" = true ]; then
        echo "Hardhat testnet is running with PID $pid"
        echo "Port $PORT is active"
        curl -s http://localhost:$PORT >/dev/null 2>&1 && echo "HTTP endpoint is responding"
    else
        echo "Hardhat testnet is not running."
    fi
}

# Function to reset Hardhat testnet
reset_hardhat_node () {
    echo "Resetting Hardhat testnet..."
    stop_hardhat_node
    kill_orphaned_processes
    rm -rf "$PROJECT_DIR"
    setup_hardhat_project
    echo "Hardhat testnet has been reset."
}

# Main script logic
case "$1" in
    start)
        install_dependencies
        install_node
        setup_hardhat_project
        start_hardhat_node
        ;;
    stop)
        stop_hardhat_node
        ;;
    status)
        status_hardhat_node
        ;;
    reset)
        reset_hardhat_node
        ;;
    clean)
        kill_orphaned_processes
        ;;
    *)
        echo "Usage: $0 [start|stop|status|reset|clean]"
        exit 1
        ;;
esac