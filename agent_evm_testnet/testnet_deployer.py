from web3 import Web3
from eth_account import Account
import json
import requests
from pathlib import Path
import solcx
import os
import subprocess

is_compiled = False

def compile_contract(contract_path, abi_name):
    global is_compiled
    """Compile the ERC20 contract using Hardhat"""
    # Run hardhat compile from hardhat-testnet directory
    os.chdir('hardhat-testnet')

    if not is_compiled:    
        subprocess.run(['npx', 'hardhat', 'compile'], check=True)
        is_compiled = True
    
    # Get the compiled contract artifact
    artifact_path = os.path.join(
        'artifacts/contracts',
        os.path.basename(contract_path),
        abi_name
    )
    
    with open(artifact_path, 'r') as file:
        contract_data = json.load(file)
    
    contract_interface = {
        'abi': contract_data['abi'],
        'bin': contract_data['bytecode']
    }
    os.chdir('..')
    return contract_interface

class ERC20TestDeployer:
    def __init__(self, node_url="http://127.0.0.1:8545"):
        self.w3 = Web3(Web3.HTTPProvider(node_url))
        self.account_address = self.w3.eth.accounts[0]

    def compile_contract(self, contract_path):
        """Compile the ERC20 contract using Hardhat"""
        return compile_contract(contract_path, 'MinimalERC20.json')
    
    def deploy_contract(self, contract_interface, name="TestToken", symbol="TST"):
        """Deploy the ERC20 contract to Hardhat network"""
        contract = self.w3.eth.contract(
            abi=contract_interface['abi'],
            bytecode=contract_interface['bin']
        )
        
        # Build transaction
        construct_txn = contract.constructor(name, symbol).build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': 2000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Send transaction using account[0]
        tx_hash = self.w3.eth.send_transaction(construct_txn)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.contractAddress
    
    def get_contract(self, contract_address, contract_interface):
        """Get contract instance at deployed address"""
        contract = self.w3.eth.contract(
            address=contract_address,
            abi=contract_interface['abi']
        )
        return contract
    
    def mint_tokens(self, contract, recipient, amount):
        """Mint new tokens to a recipient address"""
        tx = contract.functions.mint(recipient, amount).build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def get_balance(self, contract, address):
        """Get token balance of an address"""
        return contract.functions.balanceOf(address).call()


class OrderBookTestDeployer:
    def __init__(self, node_url="http://127.0.0.1:8545"):
        self.w3 = Web3(Web3.HTTPProvider(node_url))
        self.account_address = self.w3.eth.accounts[0]
    
    def compile_contract(self, contract_path):
        """Compile the OrderBook contract using Hardhat"""
        return compile_contract(contract_path, 'OrderBook.json')
    
    def deploy_contract(self, contract_interface):
        """Deploy the OrderBook contract to Hardhat network"""
        contract = self.w3.eth.contract(
            abi=contract_interface['abi'],
            bytecode=contract_interface['bin']
        )
        
        # Build transaction
        construct_txn = contract.constructor().build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Send transaction using account[0]
        tx_hash = self.w3.eth.send_transaction(construct_txn)
        
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_receipt.contractAddress
    
    def get_contract(self, contract_address, contract_interface):
        """Get contract instance at deployed address"""
        contract = self.w3.eth.contract(
            address=contract_address,
            abi=contract_interface['abi']
        )
        return contract
    
    def set_fee(self, contract, new_fee):
        """Set new fee for the OrderBook"""
        tx = contract.functions.set_fee(new_fee).build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def set_price_token(self, contract, token_address):
        """Set the price token for the OrderBook"""
        tx = contract.functions.set_price_token(token_address).build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def approve_token(self, token_contract, spender, amount, from_address=None):
        """Approve tokens for spending"""
        if from_address is None:
            from_address = self.account_address
            
        tx = token_contract.functions.approve(spender, amount).build_transaction({
            'from': from_address,
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'gas': 100000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def place_limit_buy_order(self, contract, source_token, source_amount, limit_price, from_address=None):
        """Place a limit buy order on the OrderBook"""
        if from_address is None:
            from_address = self.account_address
            
        tx = contract.functions.place_limit_buy_order(source_token, source_amount, limit_price).build_transaction({
            'from': from_address,
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def place_limit_sell_order(self, contract, source_token, source_amount, limit_price, from_address=None):
        """Place a limit sell order on the OrderBook"""
        if from_address is None:
            from_address = self.account_address
            
        tx = contract.functions.place_limit_sell_order(source_token, source_amount, limit_price).build_transaction({
            'from': from_address,
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def get_price(self, contract, token_address):
        """Get the current price between two tokens"""
        return contract.functions.get_price(token_address).call()
    
    def get_current_fee(self, contract):
        """Get current fee setting"""
        return contract.functions.fee().call()
    
    def get_price_token_address(self, contract):
        """Get the token address of the price token"""
        return contract.functions.price_token().call()
    
    def get_token_balance(self, contract, token_address):
        """Get the total balance of a token in the OrderBook"""
        return contract.functions.get_token_balance(token_address).call()

    def set_price(self, contract, token_address, new_price):
        """Set a new price for a token"""
        tx = contract.functions.set_price(token_address, new_price).build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def set_price_batch(self, contract, token_addresses, new_prices):
        """Set new prices for multiple tokens"""
        tx = contract.functions.set_price_batch(token_addresses, new_prices).build_transaction({
            'from': self.account_address,
            'nonce': self.w3.eth.get_transaction_count(self.account_address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

def test_limit_orders(erc20_deployer, orderbook_deployer, orderbook_address, tokens, token_addresses, token_symbols):
    """Test placing limit orders on the OrderBook"""
    orderbook = orderbook_deployer.get_contract(orderbook_address, orderbook_deployer.compile_contract("contracts/OrderBook.sol"))
    
    print(f"orderbook address: {orderbook_address}")
    
    # Get price token (token[0])
    price_token = tokens[0]
    
    # get current price of each token
    token_prices = []
    print("\nCurrent prices:")
    for i, address in enumerate(token_addresses):
        price = orderbook_deployer.get_price(orderbook, address)
        token_prices.append(price)
        print(f"{token_symbols[i]}: {price} {token_symbols[0]} per token")

    # For buy orders: Approve price token with enough allowance
    # Calculate needed allowance: amount * price * (1 + fee)
    buy_amount = 10  # 10 tokens
    buy_price = token_prices[1]  # price of BETA token
    sell_price = token_prices[2]  # price of GAMMA token
    fee = orderbook_deployer.get_current_fee(orderbook)  # fee in thousandths (e.g., 1 = 0.1%)
    buy_allowance_needed = (buy_amount * buy_price) + 1
    
    print("\nApproving price token for buy orders...")
    tx = orderbook_deployer.approve_token(
        price_token, 
        orderbook_address, 
        buy_allowance_needed
    )
    print(f"Approved {buy_allowance_needed // 10**18} {token_symbols[0]} for OrderBook")
    
    # For sell orders: Approve source token
    sell_amount = 1  # 1 token
    print("\nApproving source token for sell orders...")
    tx = orderbook_deployer.approve_token(
        tokens[2],  # GAMMA token for sell order
        orderbook_address, 
        sell_amount
    )
    print(f"Approved {sell_amount // 10**18} {token_symbols[2]} for OrderBook")

    # Place limit buy order
    print("\nPlacing limit buy order...")
    print(f"order book price token balance: {erc20_deployer.get_balance(price_token, orderbook_address) // 10**18}")
    print(f"order book source token balance: {erc20_deployer.get_balance(tokens[1], orderbook_address) // 10**18}")
    print(f"deployer source token balance: {erc20_deployer.get_balance(tokens[1], erc20_deployer.account_address) // 10**18}")
    print(f"deployer price token balance: {erc20_deployer.get_balance(price_token, erc20_deployer.account_address) // 10**18}")
    source_token = token_addresses[1]  # BETA token
    orderbook_deployer.place_limit_buy_order(orderbook, source_token, buy_amount, buy_price)
    print(f"Placed limit buy order for {buy_amount} {token_symbols[1]} at {buy_price / 10**18} {token_symbols[0]} per token")
    print(f"order book price token balance: {erc20_deployer.get_balance(price_token, orderbook_address) // 10**18}")
    print(f"order book source token balance: {erc20_deployer.get_balance(tokens[1], orderbook_address) // 10**18}")
    print(f"deployer source token balance: {erc20_deployer.get_balance(tokens[1], erc20_deployer.account_address) // 10**18}")
    print(f"deployer price token balance: {erc20_deployer.get_balance(price_token, erc20_deployer.account_address) // 10**18}")

    # Place limit sell order
    print("\nPlacing limit sell order...")
    print(f"order book price token balance: {erc20_deployer.get_balance(price_token, orderbook_address) // 10**18}")
    print(f"order book source token balance: {erc20_deployer.get_balance(tokens[2], orderbook_address) // 10**18}")
    print(f"deployer source token balance: {erc20_deployer.get_balance(tokens[2], erc20_deployer.account_address) // 10**18}")
    print(f"deployer price token balance: {erc20_deployer.get_balance(price_token, erc20_deployer.account_address) // 10**18}")
    source_token = token_addresses[2]  # GAMMA token
    orderbook_deployer.place_limit_sell_order(orderbook, source_token, sell_amount, sell_price)
    print(f"Placed limit sell order for {sell_amount} {token_symbols[2]} at {sell_price / 10**18} {token_symbols[0]} per token")
    print(f"order book price token balance: {erc20_deployer.get_balance(price_token, orderbook_address) // 10**18}")
    print(f"order book source token balance: {erc20_deployer.get_balance(tokens[2], orderbook_address) // 10**18}")
    print(f"deployer source token balance: {erc20_deployer.get_balance(tokens[2], erc20_deployer.account_address) // 10**18}")
    print(f"deployer price token balance: {erc20_deployer.get_balance(price_token, erc20_deployer.account_address) // 10**18}")

def main():
    # Initialize deployers
    print("Initializing deployers...")
    erc20_deployer = ERC20TestDeployer()
    orderbook_deployer = OrderBookTestDeployer()
    
    try:
        # Deploy OrderBook contract
        print("\nDeploying OrderBook contract...")
        orderbook_interface = orderbook_deployer.compile_contract("contracts/OrderBook.sol")
        orderbook_address = orderbook_deployer.deploy_contract(orderbook_interface)
        orderbook = orderbook_deployer.get_contract(orderbook_address, orderbook_interface)
        print(f"OrderBook deployed at: {orderbook_address}")

        # Deploy 10 test tokens
        print("\nDeploying test tokens...")
        token_symbols = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", 
                        "ZETA", "ETA", "THETA", "IOTA", "KAPPA"]
        tokens = []
        token_addresses = []

        # get hardhat accounts
        accounts = erc20_deployer.w3.eth.accounts

        
        # Compile ERC20 contract once
        erc20_interface = erc20_deployer.compile_contract("contracts/MinimalERC20.sol")
        
        # Deploy each token
        for symbol in token_symbols:
            name = f"Test {symbol}"
            address = erc20_deployer.deploy_contract(erc20_interface, name, symbol)
            contract = erc20_deployer.get_contract(address, erc20_interface)
            tokens.append(contract)
            token_addresses.append(address)
            print(f"Deployed {symbol} at: {address}")
            
            # Mint initial supply (1M tokens) to orderbook_address
            initial_supply = 1_000_000_000_000 * 10**18  # 1M tokens with 18 decimals
            erc20_deployer.mint_tokens(contract, orderbook_address, initial_supply)

            print(f"Minted {initial_supply // 10**18} {symbol} to OrderBook")

            # mint 10k tokens to every account
            for account in accounts:
                erc20_deployer.mint_tokens(contract, account, 10_000 * 10**18)
                print(f"Minted 10k {symbol} to {account}")            

            #print balances
            print(f"OrderBook {symbol} balance: {erc20_deployer.get_balance(contract, orderbook_address) // 10**18}")
            print(f"Deployer {symbol} balance: {erc20_deployer.get_balance(contract, erc20_deployer.account_address) // 10**18}")

        # set price_token to initial token
        orderbook_deployer.set_price_token(orderbook, token_addresses[0])

        # set batch price to be random from 1 to 10 price_token per token
        prices = [i * 10**18 for i in range(1, 11)]
        orderbook_deployer.set_price_batch(orderbook, token_addresses[1:], prices[1:])
       
        print("\nOrderbook prices:")
        for i, address in enumerate(token_addresses):
            print(f"{token_symbols[i]}: {orderbook_deployer.get_price(orderbook, address)} {token_symbols[0]} per token")

        print("\nSetup complete! OrderBook is ready for testing.")
        print(f"OrderBook contract address: {orderbook_address}")
        print("\nDeployed tokens:")
        for i, address in enumerate(token_addresses):
            print(f"{token_symbols[i]}: {address}")
            print(f"Balance: {erc20_deployer.get_balance(tokens[i], orderbook_address) // 10**18} {token_symbols[i]}")
            print(f"account {erc20_deployer.account_address} balance: {erc20_deployer.get_balance(tokens[i], erc20_deployer.account_address) // 10**18} {token_symbols[i]}")

        # Test placing limit orders
        test_limit_orders(erc20_deployer, orderbook_deployer, orderbook_address, tokens, token_addresses, token_symbols)

        # save all addresses and ABIs to a json file
        data = {
            "orderbook_address": orderbook_address,
            "orderbook_abi": orderbook_interface['abi'],
            "token_addresses": token_addresses,
            "token_symbols": token_symbols,
            "token_abi": erc20_interface['abi']
        }
        
        with open('../testnet_data.json', 'w') as file:
            json.dump(data, file)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()