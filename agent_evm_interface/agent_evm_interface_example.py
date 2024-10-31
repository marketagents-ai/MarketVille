from evm_interface import EthereumInterface
import json
import time
import os
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version
"""
py-solc-x
"""


def load_ganache_config():
    try:
        with open('../ganache_config.json', 'r') as f:
            config = json.load(f)
            print("Successfully loaded existing Ganache configuration")
            return config
    except json.JSONDecodeError:
        print("Error: ganache_config.json is not valid JSON")
        raise
    except Exception as e:
        print(f"Error loading ganache_config.json: {str(e)}")
        raise

def load_solidity_source(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading Solidity file: {str(e)}")
        raise

def deploy_dummy_erc20_contract(address_index=0, initial_amount=10000):
    """
    Deploy a dummy ERC20 token contract with initial supply minted to deployer
    
    Args:
        address_index (int): Index of the deployer account
        
    Returns:
        dict: Contract deployment details including address and instance
    """
    # Load the Solidity source code
    source_code = load_solidity_source('dummy_erc20.sol')
    
    # Initialize Ethereum interface
    eth = EthereumInterface('../ganache_config.json')
    address = eth.get_address_from_index(address_index)
    private_key = eth.get_private_key_from_index(address_index)
    
    # Compile the contract using solcx
    compiled_sol = compile_source(
        source_code,
        output_values=['abi', 'bin'],
        solc_version='0.8.0'
    )
    contract_id, contract_interface = compiled_sol.popitem()
    
    # Get bytecode and abi
    bytecode = contract_interface['bin']
    abi = contract_interface['abi']
    
    # Create contract instance
    w3 = eth.w3
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Build constructor transaction with all required parameters
    nonce = w3.eth.get_transaction_count(address)
    initial_supply = w3.to_wei(initial_amount, 'ether')  # 1 million tokens initial supply
    
    # Create constructor transaction
    construct_txn = Contract.constructor(
        "Dummy Token",  # name
        "DUMMY",       # symbol
        18,           # decimals
        0 # initial supply
    ).build_transaction({
        'from': address,
        'gas': 2000000,
        'maxFeePerGas': w3.eth.max_priority_fee + (2 * w3.eth.get_block('latest').baseFeePerGas),
        'maxPriorityFeePerGas': w3.eth.max_priority_fee,
        'nonce': nonce,
    })
    
    # Sign transaction
    signed_txn = w3.eth.account.sign_transaction(construct_txn, private_key)
    
    # Send raw transaction
    # print the keys in signed_txn
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    
    # Wait for transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt.contractAddress
    
    # Create contract instance at deployed address
    contract = w3.eth.contract(address=contract_address, abi=abi)
    
    return {
        'contract_address': contract_address,
        'contract_instance': contract,
        'deployer_address': address,
        'initial_supply': initial_supply,
        'deployment_receipt': tx_receipt,
        'eth': eth
    }

def mint_erc20_tokens(contract_instance, address_index, amount):
    """
    Mint ERC20 tokens to an address
    
    Args:
        contract_address (str): Address of the ERC20 contract
        address_index (int): Index of the account to mint tokens to
        amount (int): Amount of tokens to mint
    """
    # Initialize Ethereum interface
    eth = EthereumInterface('../ganache_config.json')
    address = eth.get_address_from_index(address_index)
    private_key = eth.get_private_key_from_index(address_index)
    
    # Mint tokens
    tx_hash = contract_instance.functions.mint(address, amount).transact({
        'from': address
    })
    
    # Wait for transaction receipt
    receipt = eth.w3.eth.wait_for_transaction_receipt(tx_hash)


def main():
        install_solc('0.8.0')
        set_solc_version('0.8.0')

        # Load existing configuration
        config = load_ganache_config()
        
        # Initialize Ethereum interface
        eth = EthereumInterface('../ganache_config.json')
        
        try:
            # Deploy a dummy ERC20 token contract
            contract_address = deploy_dummy_erc20_contract()
            # mint to address 0
            mint_erc20_tokens(contract_address['contract_instance'], 0, 1000000)

            eth.whitelist_erc20_addresses = [
                {
                        "address": contract_address['contract_address'],
                        "name": "Dummy Token",
                        "symbol": "DUMMY",
                        "decimals": 18
                }
            ]

            # Get addresses for demonstration
            address0 = eth.get_address_from_index(0)  # First account
            address1 = eth.get_address_from_index(1)  # Second account
            
            print(f"Account 0 address: {address0}")
            print(f"Account 1 address: {address1}")
            
            # Check initial balances
            balance0 = eth.get_eth_balance(address0)
            balance1 = eth.get_eth_balance(address1)
            
            print(f"\nInitial balances:")
            print(f"Account 0: {balance0} ETH")
            print(f"Account 1: {balance1} ETH")
            
            # Send some ETH from account 0 to account 1
            amount_to_send = 1.0  # 1 ETH
            print(f"\nSending {amount_to_send} ETH from Account 0 to Account 1...")
            
            tx_hash = eth.send_eth_from_index(0, address1, amount_to_send)
            print(f"Transaction hash: {tx_hash}")
            
            # Get transaction receipt
            receipt = eth.get_transaction_receipt(tx_hash)
            print("\nTransaction receipt:")
            print(f"Status: {'Success' if receipt['status'] == 1 else 'Failed'}")
            print(f"Gas used: {receipt['gasUsed']}")
            
            # Check updated balances
            new_balance0 = eth.get_eth_balance(address0)
            new_balance1 = eth.get_eth_balance(address1)
            
            print(f"\nUpdated balances:")
            print(f"Account 0: {new_balance0} ETH")
            print(f"Account 1: {new_balance1} ETH")
            
            # Get some block information
            latest_block = eth.get_block('latest')
            print(f"\nLatest block info:")
            print(f"Block number: {latest_block['number']}")
            print(f"Block timestamp: {latest_block['timestamp']}")
            print(f"Number of transactions: {len(latest_block['transactions'])}")
            
            # Demonstrate gas estimation
            estimated_gas = eth.estimate_gas(address0, address1, Web3.to_wei(0.1, 'ether'))
            print(f"\nEstimated gas for 0.1 ETH transfer: {estimated_gas}")
            
            # Example of error handling with an invalid transaction
            print("\nTrying an invalid transaction (sending more ETH than available)...")
            try:
                # Try to send more ETH than available
                tx_hash = eth.send_eth_from_index(1, address0, 10000.0)
            except Exception as e:
                print(f"Transaction failed as expected")

            

            #get contract token balance
            token_balance = contract_address['contract_instance'].functions.balanceOf(contract_address['deployer_address']).call()

            token_balance0 = eth.get_erc20_balance(contract_address['contract_instance'].address, address0)
            print(f"Account 0 token balance: {token_balance0}")

            # Transfer tokens from account 0 to account 1
            print("\nTransferring tokens from Account 0 to Account 1...")
            # print balance before transfer
            token_balance0 = eth.get_erc20_balance(contract_address['contract_instance'].address, address0)
            token_balance1 = eth.get_erc20_balance(contract_address['contract_instance'].address, address1)
            print(f"Account 0 token balance: {token_balance0}")
            print(f"Account 1 token balance: {token_balance1}")
            
            amount_to_transfer = 1000  # 1000 tokens
            
            tx_hash = eth.send_erc20(contract_address['contract_instance'].address, 0, address1, amount_to_transfer)
            print(f"Transfer transaction hash: {tx_hash}")

            # Get transaction receipt
            receipt = eth.get_transaction_receipt(tx_hash)
            print("\nTransfer transaction receipt:")
            print(f"Status: {'Success' if receipt['status'] == 1 else 'Failed'}")
            print(f"Gas used: {receipt['gasUsed']}")

            #print balance after transfer
            token_balance0 = eth.get_erc20_balance(contract_address['contract_instance'].address, address0)
            token_balance1 = eth.get_erc20_balance(contract_address['contract_instance'].address, address1)
            print(f"Account 0 token balance: {token_balance0}")
            print(f"Account 1 token balance: {token_balance1}")

            print(f"Whitelist ERC20 addresses: {eth.whitelist_erc20_addresses}")

            # deploy new dummy erc20 contract and do not whitelist it
            contract_address2 = deploy_dummy_erc20_contract()
            #mint to new contract address
            mint_erc20_tokens(contract_address2['contract_instance'], 0, 1000000)

            # attempt to transfer tokens from account 0 to account 1 using the new contract
            print("\nTransferring tokens from Account 0 to Account 1 using new contract...")
            # print balance before transfer
            token_balance0 = eth.get_erc20_balance(contract_address2['contract_instance'].address, address0)
            token_balance1 = eth.get_erc20_balance(contract_address2['contract_instance'].address, address1)
            print(f"Account 0 token balance: {token_balance0}")
            print(f"Account 1 token balance: {token_balance1}")

            amount_to_transfer = 1000  # 1000 tokens

            try:
                tx_hash = eth.send_erc20(contract_address2['contract_instance'].address, 0, address1, amount_to_transfer)
            except Exception as e:
                print(f"Transaction failed as expected: {str(e)}")

                
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()