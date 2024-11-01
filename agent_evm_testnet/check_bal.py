from web3 import Web3
import json
import sys
from typing import Dict
from decimal import Decimal
import requests
from requests.exceptions import ConnectionError

def load_config(config_path: str) -> Dict:
    """Load the ganache configuration file."""
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in config file {config_path}")
        sys.exit(1)

def test_connection(url: str) -> bool:
    """Test if we can reach the Ganache endpoint."""
    try:
        response = requests.post(url, 
                               json={"jsonrpc": "2.0", "method": "web3_clientVersion", "params": [], "id": 1},
                               timeout=5)
        return response.status_code == 200
    except ConnectionError:
        return False
    except Exception as e:
        print(f"Connection test error: {str(e)}")
        return False

def check_balances(config: Dict) -> None:
    """Check and display balances for all accounts in the config."""
    # Get port from config or use default
    port = config.get('port', 8545)
    endpoint = f"http://0.0.0.0:{port}"
    
    # Test basic connection first
    print(f"\nTesting connection to {endpoint}...")
    if not test_connection(endpoint):
        print(f"Error: Cannot reach Ganache at {endpoint}")
        print("\nPossible issues:")
        print("1. Ganache is not running")
        print("2. Wrong port number")
        print("3. Firewall blocking connection")
        print("\nTroubleshooting steps:")
        print("1. Run 'ps aux | grep ganache' to check if Ganache is running")
        print("2. Check the port in ganache_config.json")
        print("3. Try 'curl -X POST -H \"Content-Type: application/json\" --data \'{\"jsonrpc\":\"2.0\",\"method\":\"web3_clientVersion\",\"params\":[],\"id\":1}\' http://localhost:8545'")
        sys.exit(1)

    # Connect to local ganache instance
    w3 = Web3(Web3.HTTPProvider(endpoint))

    # Verify Web3 connection
    if not w3.is_connected():
        print("Error: Web3 cannot connect to Ganache instance")
        print("Basic HTTP connection successful but Web3 connection failed")
        print("Possible issue: Ganache RPC interface not responding correctly")
        sys.exit(1)

    # Get network info
    try:
        network_id = w3.net.version
        latest_block = w3.eth.block_number
        print(f"\nConnected to network:")
        print(f"Network ID: {network_id}")
        print(f"Latest block: {latest_block}")
    except Exception as e:
        print(f"Error getting network info: {str(e)}")
        sys.exit(1)

    print("\nAccount Balances:")
    print("-" * 80)
    print(f"{'Address':<42} | {'ETH Balance':>20} | {'Private Key':<64}")
    print("-" * 80)

    # Get addresses and private keys
    addresses = config.get('addresses', [])
    private_keys = config.get('private_keys', [])
    total_eth = Decimal('0')

    # Check each address
    for addr, priv_key in zip(addresses, private_keys):
        try:
            balance_wei = w3.eth.get_balance(addr)
            balance_eth = Decimal(w3.from_wei(balance_wei, 'ether'))
            total_eth += balance_eth
            
            # Format private key display (show first 6 and last 4 chars)
            masked_key = f"{priv_key[:6]}...{priv_key[-4:]}"
            print(f"{addr} | {balance_eth:>20.4f} | {masked_key}")
        except Exception as e:
            print(f"Error checking balance for {addr}: {str(e)}")

    print("-" * 80)
    print(f"Total ETH: {total_eth:,.2f}")
    print("-" * 80)

def main():
    # Get config file path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./ganache_config.json"
    
    try:
        # Load config
        config = load_config(config_path)
        
        # Check balances
        check_balances(config)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()