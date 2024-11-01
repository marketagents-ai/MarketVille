import json
import requests
from web3 import Web3
from eth_account import Account
import eth_utils
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import tempfile
import os
from pathlib import Path


"""
pip install web3
pip install eth-account
pip install requests
"""

ERC20_ABI = []
# load ./erc20_abi.json
with open(Path(__file__).parent / 'erc20_abi.json', 'r') as f:
    ERC20_ABI = json.load(f)

def external(func):
    return func

class EthereumInterface:
    def __init__(self, ganache_config_path: str, etherscan_api_key: str = "", whitelist_erc20_addresses: List[Dict[str, str]] = []):
        self.whitelist_erc20_addresses = whitelist_erc20_addresses
        # Previous initialization code remains the same...
        with open(ganache_config_path, 'r') as f:
            self.config = json.load(f)
        
        self.w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{self.config['port']}"))
        
        if not self.config['private_keys']:
            self._generate_default_accounts(10)
                

    def _generate_default_accounts(self, num_accounts: int):
        """Generate default accounts with private keys"""
        self.config['private_keys'] = []
        self.config['addresses'] = []
        
        for _ in range(num_accounts):
            acct = Account.create()
            self.config['private_keys'].append(acct.key.hex())
            self.config['addresses'].append(acct.address)
    
    @external
    def get_address_from_index(self, index: int) -> str:
        r"""
        {
        "description": "Retrieves the Ethereum address for a given account index.",
        "args": [
            {"name": "index", "type": "int", "description": "Index of the account to retrieve"}
        ],
        "returns": {
            "type": "str",
            "description": "The Ethereum address corresponding to the given index"
        },
        "example": ">>> address = get_address_from_index(0)\n'0x...'"
        }
        """
        if 0 <= index < len(self.config['addresses']):
            return self.config['addresses'][index]
        raise IndexError("Account index out of range")
    
    @external
    def get_private_key_from_index(self, index: int) -> str:
        r"""
        {
        "description": "Retrieves the private key for a given account index.",
        "args": [
            {"name": "index", "type": "int", "description": "Index of the account to retrieve"}
        ],
        "returns": {
            "type": "str",
            "description": "The private key corresponding to the given index"
        },
        "example": ">>> private_key = get_private_key_from_index(0)\n'0x...'"
        }
        """
        if 0 <= index < len(self.config['private_keys']):
            return self.config['private_keys'][index]
        raise IndexError("Account index out of range")

    @external
    def get_eth_balance(self, address: str) -> float:
        r"""
        {
        "description": "Gets the ETH balance of an address in ETH units.",
        "args": [
            {"name": "address", "type": "str", "description": "The Ethereum address to check"}
        ],
        "returns": {
            "type": "float",
            "description": "The balance in ETH"
        },
        "example": ">>> balance = get_eth_balance('0x...')\n1.5"
        }
        """
        balance_wei = self.w3.eth.get_balance(address)
        return Web3.from_wei(balance_wei, 'ether')
    
    @external
    def send_eth_from_index(self, address_index: int, to_address: str, value_eth: float) -> str:
        r"""
        {
        "description": "Sends ETH from an account (specified by index) to a destination address.",
        "args": [
            {"name": "address_index", "type": "int", "description": "Index of the sending account"},
            {"name": "to_address", "type": "str", "description": "Destination Ethereum address"},
            {"name": "value_eth", "type": "float", "description": "Amount of ETH to send"}
        ],
        "returns": {
            "type": "str",
            "description": "The transaction hash of the transfer"
        },
        "example": ">>> tx_hash = send_eth_from_index(0, '0x...', 1.5)\n'0x...'"
        }
        """
        from_address = self.get_address_from_index(address_index)
        private_key = self.config['private_keys'][address_index]
        
        # Convert ETH to Wei
        value_wei = Web3.to_wei(value_eth, 'ether')
        
        # Get accurate gas estimate
        gas_estimate = self.w3.eth.estimate_gas({
            'from': from_address,
            'to': to_address,
            'value': value_wei
        })
        
        # Add 10% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.1)
        
        # Get current gas price
        gas_price = self.w3.eth.gas_price
        
        # Calculate total transaction cost
        total_cost_wei = value_wei + (gas_limit * gas_price)
        
        # Check if sender has sufficient balance
        sender_balance = self.w3.eth.get_balance(from_address)
        if sender_balance < total_cost_wei:
            raise ValueError(
                f"Insufficient balance for transaction including gas. "
                f"Required: {Web3.from_wei(total_cost_wei, 'ether')} ETH, "
                f"Available: {Web3.from_wei(sender_balance, 'ether')} ETH"
            )
        
        # Build transaction
        transaction = {
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'to': to_address,
            'value': value_wei,
            'gas': gas_limit,
            'gasPrice': gas_price,
            'chainId': self.config['network_id']
        }
        
        # Sign and send transaction
        signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        return self.w3.to_hex(tx_hash)
    
    @external
    def get_erc20_balance(self, contract_address: str, address: str) -> int:
        r"""
        {
        "description": "Gets the ERC20 token balance of an address.",
        "args": [
            {"name": "contract_address", "type": "str", "description": "Address of the ERC20 token contract"},
            {"name": "address", "type": "str", "description": "The Ethereum address to check"}
        ],
        "returns": {
            "type": "int",
            "description": "The token balance"
        },
        "example": ">>> balance = get_erc20_balance('0x...', '0x...')\n1000000"
        }
        """
        contract = self.w3.eth.contract(address=contract_address, abi=ERC20_ABI)
        
        return contract.functions.balanceOf(address).call()
    
    @external
    def send_erc20(self, contract_address: str, from_index: int, to_address: str, amount: int) -> str:
        r"""
        {
        "description": "Sends ERC20 tokens from an account to a destination address.",
        "args": [
            {"name": "contract_address", "type": "str", "description": "Address of the ERC20 token contract"},
            {"name": "from_index", "type": "int", "description": "Index of the sending account"},
            {"name": "to_address", "type": "str", "description": "Destination Ethereum address"},
            {"name": "amount", "type": "int", "description": "Amount of tokens to send"}
        ],
        "returns": {
            "type": "str",
            "description": "The transaction hash of the transfer"
        },
        "example": ">>> tx_hash = send_erc20('0x...', 0, '0x...', 1000000)\n'0x...'"
        }
        """
        # require than contract address is in whitelist
        if contract_address not in [x['address'] for x in self.whitelist_erc20_addresses]:
            raise ValueError(f"Contract address {contract_address} is not in the whitelist")
        
        from_address = self.get_address_from_index(from_index)
        private_key = self.config['private_keys'][from_index]
        
        contract = self.w3.eth.contract(address=contract_address, abi=ERC20_ABI)
        
        data = contract.functions.transfer(to_address, amount).build_transaction({
            'chainId': self.config['network_id'],
            'gas': 100000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(from_address)
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(data, private_key)
        
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        return self.w3.to_hex(tx_hash)

    @external
    def get_transaction_receipt(self, tx_hash: str) -> Dict:
        r"""
        {
        "description": "Retrieves the receipt for a transaction.",
        "args": [
            {"name": "tx_hash", "type": "str", "description": "Hash of the transaction"}
        ],
        "returns": {
            "type": "Dict",
            "description": "Transaction receipt containing status, gas used, and other details"
        },
        "example": ">>> receipt = get_transaction_receipt('0x...')\n{'status': 1, 'gasUsed': 21000, ...}"
        }
        """
        return self.w3.eth.get_transaction_receipt(tx_hash)
    
    @external
    def estimate_gas(self, from_address: str, to_address: str, value: int = 0, data: str = '') -> int:
        r"""
        {
        "description": "Estimates the gas required for a transaction.",
        "args": [
            {"name": "from_address", "type": "str", "description": "Address sending the transaction"},
            {"name": "to_address", "type": "str", "description": "Destination address"},
            {"name": "value", "type": "int", "description": "Amount of wei to send"},
            {"name": "data", "type": "str", "description": "Transaction data in hex format"}
        ],
        "returns": {
            "type": "int",
            "description": "Estimated gas required for the transaction"
        },
        "example": ">>> gas = estimate_gas('0x...', '0x...', 1000000000)\n21000"
        }
        """
        return self.w3.eth.estimate_gas({
            'from': from_address,
            'to': to_address,
            'value': value,
            'data': data
        })
    
    @external
    def get_block(self, block_identifier: Union[str, int]) -> Dict:
        r"""
        {
        "description": "Retrieves information about a specific block.",
        "args": [
            {"name": "block_identifier", "type": "Union[str, int]", "description": "Block number or hash"}
        ],
        "returns": {
            "type": "Dict",
            "description": "Block information including transactions, timestamp, and other details"
        },
        "example": ">>> block = get_block(12345)\n{'number': 12345, 'hash': '0x...', 'transactions': [...]}"
        }
        """
        return self.w3.eth.get_block(block_identifier)
    
    @external
    def get_logs(
        self,
        from_block: int,
        to_block: int,
        address: Optional[str] = None,
        topics: Optional[List[str]] = None
    ) -> List[Dict]:
        r"""
        {
        "description": "Retrieves event logs from the blockchain within the specified block range.",
        "args": [
            {"name": "from_block", "type": "int", "description": "Starting block number"},
            {"name": "to_block", "type": "int", "description": "Ending block number"},
            {"name": "address", "type": "Optional[str]", "description": "Contract address to filter logs"},
            {"name": "topics", "type": "Optional[List[str]]", "description": "List of topics to filter logs"}
        ],
        "returns": {
            "type": "List[Dict]",
            "description": "List of event logs matching the specified criteria"
        },
        "example": ">>> logs = get_logs(1000000, 1000100, '0x...')\n[{'address': '0x...', 'topics': [...], 'data': '0x...'}]"
        }
        """

        filter_params = {
            'fromBlock': from_block,
            'toBlock': to_block
        }
        
        if address:
            filter_params['address'] = address
        if topics:
            filter_params['topics'] = topics
            
        return self.w3.eth.get_logs(filter_params)
    
    @external
    def extract_revert_reason(self, tx_hash: str) -> str:
        r"""
        {
        "description": "Extracts the reason for a transaction revert from a failed transaction.",
        "args": [
            {"name": "tx_hash", "type": "str", "description": "Hash of the failed transaction"}
        ],
        "returns": {
            "type": "str",
            "description": "Human readable explanation of why the transaction reverted"
        },
        "example": ">>> reason = extract_revert_reason('0x...')\n'Insufficient balance'"
        }
        """
        try:
            # Get the transaction
            tx = self.w3.eth.get_transaction(tx_hash)
            
            # Get the transaction receipt
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            
            # Check if transaction failed
            if receipt['status'] == 1:
                raise Exception("Transaction did not fail")
                
            # Replay the transaction to get the revert reason
            replay_tx = {
                'from': tx['from'],
                'to': tx['to'],
                'data': tx['input'],
                'value': tx['value'],
                'gas': tx['gas'],
                'gasPrice': tx['gasPrice'],
                'nonce': tx['nonce']
            }
            
            try:
                # Try to replay the transaction
                self.w3.eth.call(
                    replay_tx,
                    receipt.blockNumber
                )
            except Exception as e:
                # Extract revert reason from the error message
                error_message = str(e)
                
                # Handle different error formats
                
                # Format 1: Standard revert message
                if 'revert' in error_message.lower():
                    # Look for message between single quotes
                    import re
                    matches = re.findall(r"\'(.+?)\'", error_message)
                    if matches:
                        return matches[0]
                        
                # Format 2: Custom error selector
                if 'custom error' in error_message.lower():
                    # Extract the selector and decode if possible
                    selector = error_message[-10:-2] if error_message[-10:-2].startswith('0x') else None
                    if selector and self.w3.eth.contract(address=tx['to'], abi=self.get_contract(tx['to'])['abi']):
                        contract = self.w3.eth.contract(address=tx['to'], abi=self.get_contract(tx['to'])['abi'])
                        try:
                            # Try to decode custom error
                            decoded = contract.decode_function_input(tx['input'])
                            return f"Custom error: {decoded[0].fn_name} with args {decoded[1]}"
                        except:
                            return f"Custom error with selector: {selector}"
                            
                # Format 3: Panic code
                if 'panic code' in error_message.lower():
                    panic_codes = {
                        0x01: "Assertion failed",
                        0x11: "Arithmetic overflow/underflow",
                        0x12: "Division or modulo by zero",
                        0x21: "Invalid enum value",
                        0x22: "Storage data is incorrectly encoded",
                        0x31: "Array index out of bounds",
                        0x32: "Memory access out of bounds",
                        0x41: "Zero initialization of uninitialized variable",
                        0x51: "Call to an invalid/uninitialized function pointer"
                    }
                    
                    try:
                        # Extract panic code and lookup meaning
                        import re
                        panic_code = int(re.findall(r"Panic\(uint256 (0x[0-9a-fA-F]+)\)", error_message)[0], 16)
                        return f"Panic: {panic_codes.get(panic_code, 'Unknown panic code')}"
                    except:
                        pass
                
                # Format 4: Raw revert data
                if error_message.startswith('0x'):
                    try:
                        # Try to decode raw revert data
                        hex_data = error_message[2:] # remove '0x' prefix
                        # Check if it's a string (first 4 bytes are the selector)
                        if len(hex_data) > 8:  # 4 bytes selector + at least some data
                            decoded = bytes.fromhex(hex_data[8:]).decode('utf-8').strip('\x00')
                            if decoded:
                                return decoded
                    except:
                        pass
                        
                # If no specific format is recognized, return the raw error
                return f"Raw error: {error_message}"
                
            return "Unknown revert reason"
            
        except Exception as e:
            raise Exception(f"Error extracting revert reason: {str(e)}")

    @external     
    def get_transaction_error_details(self, tx_hash: str) -> Dict[str, Any]:
        r"""
        {
        "description": "Gets detailed error information for a failed transaction including gas usage and error location.",
        "args": [
            {"name": "tx_hash", "type": "str", "description": "Hash of the failed transaction"}
        ],
        "returns": {
            "type": "Dict[str, Any]",
            "description": "Dictionary containing error details, gas usage, and debug information"
        },
        "example": ">>> details = get_transaction_error_details('0x...')\n{'revert_reason': 'Insufficient balance', 'gas_used': 21000, 'gas_limit': 21000, 'block_number': 12345, 'error_location': {...}}"
        }
        """
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            tx = self.w3.eth.get_transaction(tx_hash)
            
            error_details = {
                'revert_reason': self.extract_revert_reason(tx_hash),
                'gas_used': receipt['gasUsed'],
                'gas_limit': tx['gas'],
                'block_number': receipt['blockNumber'],
                'error_location': None  # Will be populated if debug trace is available
            }
            
            # Try to get debug trace if available
            try:
                trace = self.w3.provider.make_request("debug_traceTransaction", [tx_hash])
                if trace.get('result', {}).get('structLogs'):
                    # Find the last executed operation before revert
                    logs = trace['result']['structLogs']
                    for i in range(len(logs) - 1, -1, -1):
                        if logs[i].get('op') in ['REVERT', 'INVALID']:
                            error_details['error_location'] = {
                                'pc': logs[i].get('pc'),
                                'op': logs[i].get('op'),
                                'gas_remaining': logs[i].get('gas'),
                                'depth': logs[i].get('depth')
                            }
                            break
            except:
                pass
                
            return error_details
            
        except Exception as e:
            raise Exception(f"Error getting transaction error details: {str(e)}")
        


    @external
    def get_whitelist_erc20_addresses(self):
        r"""
        {
        "description": "Retrieves a list of whitelisted ERC20 token addresses.",
        "returns": {
            "type": "List[Dict]",
            "description": "List of dictionaries containing token address, name, symbol, and decimals"
        },
        "example": ">>> whitelist = get_whitelist_erc20_addresses()\n[{'address': '0x...', 'name': 'Token Name', 'symbol': 'TKN', 'decimals': 18}]"
        }
        """
        return self.whitelist_erc20_addresses
        