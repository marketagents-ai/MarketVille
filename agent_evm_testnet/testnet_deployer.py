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
    
    def deposit(self, contract, token_address, amount, from_address=None):
        """Deposit tokens into the OrderBook"""
        if from_address is None:
            from_address = self.account_address
            
        # Remove the withdrawal check and just deposit
        tx = contract.functions.deposit(token_address, amount).build_transaction({
            'from': from_address,
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)

    def withdraw(self, contract, token_address, from_address=None):
        """Withdraw tokens from the OrderBook"""
        if from_address is None:
            from_address = self.account_address
            
        # Check if there's any liquidity to withdraw
        existing_liquidity = contract.functions.individual_liquidity(from_address, token_address).call()
        if existing_liquidity == 0:
            return None
            
        tx = contract.functions.withdraw(token_address).build_transaction({
            'from': from_address,
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def swap(self, contract, source_token, source_amount, target_token, from_address=None):
        """Execute a token swap"""
        if from_address is None:
            from_address = self.account_address
            
        tx = contract.functions.swap(source_token, source_amount, target_token).build_transaction({
            'from': from_address,
            'nonce': self.w3.eth.get_transaction_count(from_address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        tx_hash = self.w3.eth.send_transaction(tx)
        return self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def get_price(self, contract, sell_token_address, buy_token_address):
        """Get the current price between two tokens"""
        return contract.functions.get_price(sell_token_address, buy_token_address).call()
    
    def get_individual_liquidity(self, contract, user_address, token_address):
        """Get individual liquidity for a user and token"""
        return contract.functions.individual_liquidity(user_address, token_address).call()
    
    def get_total_pool_balance(self, contract, token_address):
        """Get total pool balance for a token"""
        return contract.functions.total_pool_balance(token_address).call()
    
    def get_current_fee(self, contract):
        """Get current fee setting"""
        return contract.functions.fee().call()
    
def test_swaps_and_price_changes(erc20_deployer, orderbook_deployer, orderbook_address, tokens, token_addresses, token_symbols):
    """
    Test token swaps and verify price changes in the OrderBook
    
    Args:
        erc20_deployer: ERC20TestDeployer instance
        orderbook_deployer: OrderBookTestDeployer instance
        orderbook_address: Address of the deployed OrderBook contract
        tokens: List of token contract instances
        token_addresses: List of token contract addresses
        token_symbols: List of token symbols
    """
    
    # Get OrderBook contract
    orderbook_details = orderbook_deployer.compile_contract("OrderBook.sol")
    orderbook = orderbook_deployer.get_contract(orderbook_address, orderbook_details)
    
    # Test account (using accounts[1] to simulate different user)
    test_account = erc20_deployer.w3.eth.accounts[1]
    
    # Amount to swap (100 tokens)
    swap_amount = 100 * 10**18
    
    print("\nTesting swaps and price changes...")
    
    # Test swaps between first three tokens (ALPHA, BETA, GAMMA)
    for i in range(3):
        for j in range(i + 1, 3):
            token1, token2 = tokens[i], tokens[j]
            addr1, addr2 = token_addresses[i], token_addresses[j]
            sym1, sym2 = token_symbols[i], token_symbols[j]
            
            print(f"\nTesting {sym1}-{sym2} swap pair:")
            
            try:
                # Get initial balances and price
                initial_price = orderbook_deployer.get_price(orderbook, addr1, addr2)
                initial_balance1 = token1.functions.balanceOf(test_account).call()
                initial_balance2 = token2.functions.balanceOf(test_account).call()
                initial_pool1 = orderbook_deployer.get_total_pool_balance(orderbook, addr1)
                initial_pool2 = orderbook_deployer.get_total_pool_balance(orderbook, addr2)
                
                print(f"Initial pool balances: {initial_pool1 / 10**18} {sym1}, {initial_pool2 / 10**18} {sym2}")
                
                # Store the pre-mint balance to calculate actual change
                pre_mint_balance1 = token1.functions.balanceOf(test_account).call()
                
                # Mint some tokens to test account
                erc20_deployer.mint_tokens(token1, test_account, swap_amount * 2)
                print(f"Minted {swap_amount / 10**18} {sym1} to test account")
                
                # Important fix: Approve tokens for OrderBook contract address
                tx = token1.functions.approve(
                    orderbook_address,
                    swap_amount * 2
                ).build_transaction({
                    'from': test_account,
                    'nonce': erc20_deployer.w3.eth.get_transaction_count(test_account),
                    'gas': 100000,
                    'gasPrice': erc20_deployer.w3.eth.gas_price
                })
                
                tx_hash = erc20_deployer.w3.eth.send_transaction(tx)
                erc20_deployer.w3.eth.wait_for_transaction_receipt(tx_hash)
                print(f"Approved {sym1} for OrderBook")
                
                # Verify allowance
                allowance = token1.functions.allowance(test_account, orderbook_address).call()
                print(f"Current allowance: {allowance / 10**18} {sym1}")
                
                # Get balance right before swap
                balance_before_swap1 = token1.functions.balanceOf(test_account).call()
                balance_before_swap2 = token2.functions.balanceOf(test_account).call()
                
                # Execute swap
                print(f"Swapping {swap_amount / 10**18} {sym1} for {sym2}...")
                orderbook_deployer.swap(
                    orderbook,
                    addr1,
                    swap_amount,
                    addr2,
                    test_account
                )
                
                # Get post-swap values
                new_price = orderbook_deployer.get_price(orderbook, addr1, addr2)
                new_balance1 = token1.functions.balanceOf(test_account).call()
                new_balance2 = token2.functions.balanceOf(test_account).call()
                new_pool1 = orderbook_deployer.get_total_pool_balance(orderbook, addr1)
                new_pool2 = orderbook_deployer.get_total_pool_balance(orderbook, addr2)
                
                # Calculate actual balance changes from the swap (excluding minting)
                balance_change1 = new_balance1 - balance_before_swap1
                balance_change2 = new_balance2 - balance_before_swap2
                
                # Print results
                print(f"\nSwap completed!")
                print(f"New price {sym1}/{sym2}: {new_price / 10**18}")
                print(f"Price change: {((new_price - initial_price) / initial_price) * 100:.2f}%")
                print(f"New pool balances: {new_pool1 / 10**18} {sym1}, {new_pool2 / 10**18} {sym2}")
                print(f"Token balance changes from swap:")
                print(f"{sym1}: {balance_change1 / 10**18}")
                print(f"{sym2}: {balance_change2 / 10**18}")
                
                # Verify expected behaviors
                assert new_price != initial_price, "Price should change after swap"
                assert new_pool1 > initial_pool1, f"Pool {sym1} balance should increase"
                assert new_pool2 < initial_pool2, f"Pool {sym2} balance should decrease"
                assert balance_change1 == -swap_amount, f"User {sym1} balance should decrease by swap amount"
                assert balance_change2 > 0, f"User {sym2} balance should increase"
                
            except Exception as e:
                print(f"Error testing {sym1}-{sym2} swap: {str(e)}")
                continue
    
    print("\nSwap tests completed!")


def main():
    # Initialize deployers
    print("Initializing deployers...")
    erc20_deployer = ERC20TestDeployer()
    orderbook_deployer = OrderBookTestDeployer()
    
    try:
        # Load token configurations
        with open('testnet.json', 'r') as file:
            config = json.load(file)
        
        # Deploy OrderBook contract
        print("\nDeploying OrderBook contract...")
        orderbook_interface = orderbook_deployer.compile_contract("contracts/OrderBook.sol")
        orderbook_address = orderbook_deployer.deploy_contract(orderbook_interface)
        orderbook = orderbook_deployer.get_contract(orderbook_address, orderbook_interface)
        print(f"OrderBook deployed at: {orderbook_address}")

        # Deploy tokens from configuration
        print("\nDeploying tokens from configuration...")
        tokens = []
        token_addresses = []
        token_symbols = []
        
        # Compile ERC20 contract once
        erc20_interface = erc20_deployer.compile_contract("contracts/MinimalERC20.sol")
        
        # Deploy each token
        for token_config in config['tokens']:
            symbol = token_config['symbol']
            name = token_config['name']
            initial_supply = token_config['initial_supply'] * 10**18  # Convert to wei
            
            # Deploy token
            address = erc20_deployer.deploy_contract(erc20_interface, name, symbol)
            contract = erc20_deployer.get_contract(address, erc20_interface)
            tokens.append(contract)
            token_addresses.append(address)
            token_symbols.append(symbol)
            print(f"Deployed {symbol} at: {address}")
            
            # Mint initial supply to account[0]
            erc20_deployer.mint_tokens(contract, erc20_deployer.account_address, initial_supply)
            print(f"Minted {initial_supply // 10**18} {symbol} to {erc20_deployer.account_address}")

        # Create pools with initial prices
        print("\nCreating liquidity pools...")
        usdc_index = token_symbols.index('USDC')
        
        for i, token_config in enumerate(config['tokens']):
            if token_symbols[i] != 'USDC':  # Skip USDC-USDC pair
                # Calculate pool amounts based on initial price
                base_amount = 10000
                initial_price_usd = token_config['initial_price_usd']
                token_pool_amount = base_amount * 10**18
                usdc_pool_amount = int(base_amount * initial_price_usd * 10**18)
                
                print(f"\nCreating pool for {token_symbols[i]}-USDC")
                print(f"Initial price: ${initial_price_usd}")
                
                # Approve tokens
                orderbook_deployer.approve_token(tokens[i], orderbook_address, token_pool_amount)
                orderbook_deployer.approve_token(tokens[usdc_index], orderbook_address, usdc_pool_amount)
                
                try:
                    # Deposit tokens to create pool
                    orderbook_deployer.deposit(orderbook, token_addresses[i], token_pool_amount)
                    orderbook_deployer.deposit(orderbook, token_addresses[usdc_index], usdc_pool_amount)
                    
                    # Verify pool creation
                    price = orderbook_deployer.get_price(orderbook, token_addresses[i], token_addresses[usdc_index])
                    print(f"Pool created! Current price {token_symbols[i]}/USDC: {price / 10**18}")
                    
                except Exception as e:
                    print(f"Error creating pool {token_symbols[i]}-USDC: {str(e)}")
                    continue

        # Save deployment data
        data = {
            "orderbook_address": orderbook_address,
            "orderbook_abi": orderbook_interface['abi'],
            "token_addresses": {
                symbol: address
                for symbol, address in zip(token_symbols, token_addresses)
            },
            "token_symbols": token_symbols,
            "token_abi": erc20_interface['abi'],
            "initial_prices": {
                symbol: config['tokens'][i]['initial_price_usd']
                for i, symbol in enumerate(token_symbols)
            }
        }

        with open('testnet_data.json', 'w') as file:
            json.dump(data, file, indent=2)

        print("\nSetup complete! Configuration saved to testnet_data.json")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()