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
    func._external_tagged = True
    return func

def init(func):
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
    def get_address_list(self) -> List[str]:
        r"""
        {
        "description": "Retrieves a list of Ethereum addresses that are available for use.",
        "returns": {
            "type": "List[str]",
            "description": "List of Ethereum addresses"
        },
        "example": ">>> addresses = get_address_list()\n['0x...', '0x...', ...]"
        }
        """
        return self.config['addresses']

    
    def get_address_from_index(self, index: int) -> str:
        r"""
        {
        "description": "Retrieves the ethereum address at a gven index (max 9). These are your accounts.",
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
        "description": "Gets the ETH balance of an address in ETH units. You MUST have a valid ethereum address. If you are wondering about accounts you own, get the address list first.",
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
        return str(Web3.from_wei(balance_wei, 'ether')) + " Ether (ETH)"
    
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
        "description": "Gets the ERC20 token balance of an address. The contract address must be in the whitelist. If you are wondering about tokens you own, get the whitelist first.",
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
        
@init
def init_ethereum_interface():
    ganache_config_path = Path(__file__).parent / '../ganache_config.json'
    etherscan_api_key = ""
    whitelist_erc20_addresses = []
    instance =  EthereumInterface(ganache_config_path, etherscan_api_key, whitelist_erc20_addresses)
    print("Ethereum interface initialized")
    return instance