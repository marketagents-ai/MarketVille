#!/bin/bash

# Default config file location
CONFIG_FILE="${1:-./ganache_config.json}"
PID_FILE="/tmp/ganache-daemon.pid"
LOG_FILE="/tmp/ganache.log"

# Parse command line arguments
parse_args() {
    FORK_URL=""
    FORK_BLOCK=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --fork)
                if [ -z "$2" ]; then
                    echo "Error: --fork requires a URL"
                    exit 1
                fi
                FORK_URL="$2"
                shift 2
                ;;
            --fork-block)
                if [ -z "$2" ]; then
                    echo "Error: --fork-block requires a block number"
                    exit 1
                fi
                FORK_BLOCK="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
}

# Function to ensure correct PATH setup
setup_path() {
    if [[ ":$PATH:" != *":/usr/local/bin:"* ]]; then
        echo "Adding /usr/local/bin to PATH..."
        export PATH="$PATH:/usr/local/bin"
        
        SHELL_CONFIG="$HOME/.bashrc"
        if [ -f "$HOME/.zshrc" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        fi
        
        if ! grep -q 'export PATH="$PATH:/usr/local/bin"' "$SHELL_CONFIG"; then
            echo 'export PATH="$PATH:/usr/local/bin"' >> "$SHELL_CONFIG"
            echo "Added PATH update to $SHELL_CONFIG"
        fi
    fi
}

# Check for required dependencies and install if missing
check_dependencies() {
    setup_path
    
    if ! command -v node >/dev/null 2>&1; then
        echo "Node.js is not installed. Please install Node.js first."
        echo "Visit https://nodejs.org/ for installation instructions."
        exit 1
    fi

    if ! command -v npm >/dev/null 2>&1; then
        echo "npm is not installed. Please install npm first."
        echo "It usually comes with Node.js installation."
        exit 1
    fi

    if ! command -v ganache >/dev/null 2>&1; then
        echo "Ganache is not installed. Attempting to install..."
        if npm install -g ganache >/dev/null 2>&1; then
            echo "Ganache installed successfully!"
        else
            echo "Failed to install Ganache without sudo. Trying with sudo..."
            if sudo npm install -g ganache; then
                echo "Ganache installed successfully with sudo!"
            else
                echo "Failed to install Ganache. Please install manually:"
                echo "sudo npm install -g ganache"
                exit 1
            fi
        fi
    fi

    if ! command -v ganache >/dev/null 2>&1; then
        echo "Ganache installation verification failed."
        exit 1
    fi

    echo "All dependencies are satisfied."
}

# Enhanced process check function
is_ganache_process() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    
    if ps -p "$pid" -o comm= 2>/dev/null | grep -q "node"; then
        if ps -p "$pid" -o args= | grep -q "ganache"; then
            return 0
        fi
    fi
    
    return 1
}

# Function to find ganache process by port
find_ganache_by_port() {
    local port=$(jq -r '.port' "$CONFIG_FILE")
    pgrep -f "ganache.*--port $port"
}

# Enhanced running check
is_running() {
    local pid
    
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if is_ganache_process "$pid"; then
            return 0
        fi
        rm "$PID_FILE"
    fi
    
    pid=$(find_ganache_by_port)
    if [ ! -z "$pid" ]; then
        echo "$pid" > "$PID_FILE"
        return 0
    fi
    
    return 1
}

# Enhanced stop function
stop_ganache() {
    local pid
    local port=$(jq -r '.port' "$CONFIG_FILE")
    
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if is_ganache_process "$pid"; then
            echo "Stopping Ganache process with PID: $pid..."
            kill -15 "$pid" 2>/dev/null
            
            for i in {1..10}; do
                if ! is_ganache_process "$pid"; then
                    break
                fi
                sleep 1
            done
            
            if is_ganache_process "$pid"; then
                echo "Process didn't stop gracefully, forcing..."
                kill -9 "$pid" 2>/dev/null
            fi
        fi
        rm "$PID_FILE"
    fi
    
    for pid in $(pgrep -f "ganache.*--port $port"); do
        if [ ! -z "$pid" ]; then
            echo "Found additional Ganache process on port $port (PID: $pid)"
            kill -15 "$pid" 2>/dev/null
            sleep 2
            kill -9 "$pid" 2>/dev/null 2>&1
        fi
    done
    
    echo "Ganache stopped"
}

# Enhanced start function with fork support
start_ganache() {
    if is_running; then
        echo "Ganache is already running"
        return 1
    fi

    echo "Starting Ganache..."
    
    # Read configuration
    network_id=$(jq -r '.network_id' "$CONFIG_FILE")
    port=$(jq -r '.port' "$CONFIG_FILE")
    default_balance=$(jq -r '.default_balance' "$CONFIG_FILE")
    block_time=$(jq -r '.block_time' "$CONFIG_FILE")
    gas_limit=$(jq -r '.gas_limit' "$CONFIG_FILE")
    gas_price=$(jq -r '.gas_price' "$CONFIG_FILE")
    
    # Handle fork configuration with CLI override
    fork_params=""
    if [ ! -z "$FORK_URL" ]; then
        echo "Using fork URL from CLI: $FORK_URL"
        fork_params="--fork.url \"$FORK_URL\""
        if [ ! -z "$FORK_BLOCK" ]; then
            echo "Using fork block from CLI: $FORK_BLOCK"
            fork_params="$fork_params --fork.blockNumber $FORK_BLOCK"
        fi
    else
        fork_enabled=$(jq -r '.fork' "$CONFIG_FILE")
        if [ "$fork_enabled" = "true" ]; then
            fork_url=$(jq -r '.fork_url' "$CONFIG_FILE")
            if [ ! -z "$fork_url" ] && [ "$fork_url" != "null" ]; then
                fork_params="--fork.url \"$fork_url\""
                fork_block_number=$(jq -r '.fork_block_number // empty' "$CONFIG_FILE")
                if [ ! -z "$fork_block_number" ]; then
                    fork_params="$fork_params --fork.blockNumber $fork_block_number"
                fi
            else
                echo "Warning: Fork is enabled but fork_url is not set in config"
            fi
        fi
    fi
    
    # Build accounts string with 0x prefix
    accounts_param=""
    while IFS= read -r key; do
        # Add 0x prefix if not present
        if [[ ! "$key" =~ ^0x ]]; then
            key="0x$key"
        fi
        accounts_param="$accounts_param --account=\"$key,$default_balance\""
    done < <(jq -r '.private_keys[]' "$CONFIG_FILE")
    
    # Start ganache with nohup
    cmd="nohup ganache \
        --host 0.0.0.0 \
        --networkId $network_id \
        --port $port \
        --blockTime $block_time \
        --gasLimit $gas_limit \
        --gasPrice $gas_price \
        $fork_params \
        $accounts_param \
        >> $LOG_FILE 2>&1 &"
    
    echo "Running command: $cmd"
    eval "$cmd"
    
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # Wait for process to start and verify
    sleep 3
    
    if is_running; then
        echo "Ganache started successfully with PID: $pid"
        echo "Log file: $LOG_FILE"
        
        # Verify port is listening
        if timeout 10 bash -c "until netstat -tuln | grep -q ':$port '; do sleep 1; done"; then
            echo "Ganache is listening on port $port"
            return 0
        else
            echo "Warning: Ganache is not listening on port $port after 10 seconds"
            stop_ganache
            return 1
        fi
    else
        echo "Failed to start Ganache"
        [ -f "$PID_FILE" ] && rm "$PID_FILE"
        return 1
    fi
}

# Function to generate private keys
generate_keys() {
    # Create config file if it doesn't exist
    if [ ! -f "$CONFIG_FILE" ]; then
        create_default_config
    fi

    keys=()
    addresses=()
    
    echo "Generating 10 private keys..."
    for i in {1..10}; do
        # Generate private key without 0x prefix
        private_key=$(openssl rand -hex 32)
        
        address=$(NODE_NO_WARNINGS=1 node -e "
            const { Web3 } = require('web3');
            const web3 = new Web3();
            
            async function getAddress() {
                try {
                    const privKey = '0x${private_key}';
                    const account = web3.eth.accounts.privateKeyToAccount(privKey);
                    console.log(account.address);
                } catch (error) {
                    console.error('Error generating address:', error);
                    process.exit(1);
                }
            }
            
            getAddress();
        ")
        
        if [ $? -eq 0 ] && [ ! -z "$address" ]; then
            # Store private key without 0x prefix in config
            keys+=("\"$private_key\"")
            addresses+=("\"$address\"")
        else
            echo "Error generating address for private key. Skipping..."
            continue
        fi
    done
    
    if [ ${#keys[@]} -gt 0 ] && [ ${#addresses[@]} -gt 0 ]; then
        private_keys=$(IFS=,; echo "[${keys[*]}]")
        account_addresses=$(IFS=,; echo "[${addresses[*]}]")
        
        local temp_config=$(mktemp)
        if jq --arg keys "$private_keys" --arg addrs "$account_addresses" \
            '.private_keys = ($keys | fromjson) | .addresses = ($addrs | fromjson)' \
            "$CONFIG_FILE" > "$temp_config"; then
            mv "$temp_config" "$CONFIG_FILE"
            echo "Generated and stored ${#keys[@]} private keys and addresses in config file"
        else
            echo "Error updating config file"
            rm "$temp_config"
            return 1
        fi
    else
        echo "No valid keys were generated"
        return 1
    fi
}

# Enhanced default config with fork settings
create_default_config() {
    echo "Creating default config file at $CONFIG_FILE..."
    cat > "$CONFIG_FILE" << EOF
{
    "network": "mainnet",
    "fork": false,
    "fork_url": "",
    "fork_block_number": null,
    "network_id": 1337,
    "port": 8556,
    "default_balance": "10000000000000000000",
    "block_time": 0,
    "gas_limit": "12000000",
    "gas_price": "20000000000",
    "private_keys": [],
    "addresses": []
}
EOF
    generate_keys
    echo "Default config file created"
}

# Enhanced status function with fork information
status_ganache() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo "Ganache is running with PID: $pid"
        local port=$(jq -r '.port' "$CONFIG_FILE")
        echo "Checking port $port..."
        if netstat -tuln | grep -q ":$port "; then
            echo "Ganache is listening on port $port"
        else
            echo "Warning: Process is running but not listening on port $port"
            echo "This might indicate a problem with the service"
        fi
        
        # Display fork status
        if [ ! -z "$FORK_URL" ]; then
            echo "Fork enabled (via CLI):"
            echo "  URL: $FORK_URL"
            [ ! -z "$FORK_BLOCK" ] && echo "  Block: $FORK_BLOCK"
        else
            local fork_enabled=$(jq -r '.fork' "$CONFIG_FILE")
            if [ "$fork_enabled" = "true" ]; then
                local fork_url=$(jq -r '.fork_url' "$CONFIG_FILE")
                local fork_block=$(jq -r '.fork_block_number // "latest"' "$CONFIG_FILE")
                echo "Fork enabled (via config):"
                echo "  URL: $fork_url"
                echo "  Block: $fork_block"
            else
                echo "Fork disabled"
            fi
        fi
        
        echo "Log file: $LOG_FILE"
        echo "Funded accounts:"
        jq -r '.addresses[]' "$CONFIG_FILE"
        
        echo -e "\nLast few lines of log file:"
        tail -n 5 "$LOG_FILE"
    else
        echo "Ganache is not running"
    fi
}

# Command line interface
case "$2" in
    start)
        check_dependencies
        if [ ! -f "$CONFIG_FILE" ]; then
            create_default_config
        fi
        parse_args "${@:3}"  # Parse remaining arguments after 'start'
        start_ganache
        ;;
    stop)
        stop_ganache
        ;;
    restart)
        check_dependencies
        stop_ganache
        sleep 2
        parse_args "${@:3}"  # Parse remaining arguments after 'restart'
        start_ganache
        ;;
    status)
        parse_args "${@:3}"  # Parse remaining arguments after 'status'
        status_ganache
        ;;
    regen-keys)
        generate_keys
        ;;
    *)
        echo "Usage: $0 [config_file] {start|stop|restart|status|regen-keys} [options]"
        echo ""
        echo "Options:"
        echo "  --fork <URL>        Fork from a running Ethereum node"
        echo "  --fork-block <num>  Fork from a specific block number"
        exit 1
        ;;
esac

exit 0