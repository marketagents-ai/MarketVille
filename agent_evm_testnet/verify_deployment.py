# verify_token_relationships.py

import json
from web3 import Web3
from eth_account import Account
import os

def main():
    # Load testnet data
    with open('testnet_data.json', 'r') as f:
        testnet_data = json.load(f)

    # Set up Web3 connection
    rpc_url = "http://localhost:8545"
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    # Get data from testnet_data.json
    token_addresses = testnet_data['token_addresses']
    token_abi = testnet_data['token_abi']
    orderbook_address = testnet_data['orderbook_address']
    orderbook_abi = testnet_data['orderbook_abi']
    initial_prices = testnet_data['initial_prices']

    # Initialize OrderBook contract
    orderbook_contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)

    print("Verifying token symbols, addresses, decimals, and prices:\n")

    # Dictionary to store token info
    tokens_info = {}

    # Verify token symbols and addresses
    for symbol, address in token_addresses.items():
        # Initialize token contract
        token_contract = w3.eth.contract(address=address, abi=token_abi)
        # Get symbol from contract
        contract_symbol = token_contract.functions.symbol().call()
        # Get decimals
        decimals = token_contract.functions.decimals().call()
        # Store info
        tokens_info[symbol] = {
            'address': address,
            'contract_symbol': contract_symbol,
            'decimals': decimals,
            'contract': token_contract
        }
        # Print verification
        print(f"Token Symbol: {symbol}")
        print(f"  Expected Address: {address}")
        print(f"  Contract Symbol: {contract_symbol}")
        print(f"  Decimals: {decimals}")
        if symbol != contract_symbol:
            print(f"  Mismatch detected! Expected symbol {symbol}, but contract symbol is {contract_symbol}")
        else:
            print(f"  Symbol verified.")
        print()

    # Verify initial prices by calculating expected prices from liquidity pools
    print("\nVerifying token prices from OrderBook contract:\n")

    # Assume USDC is the quote currency
    usdc_symbol = 'USDC'
    usdc_address = token_addresses[usdc_symbol]

    for symbol, info in tokens_info.items():
        if symbol == usdc_symbol:
            continue  # Skip USDC-USDC pair
        token_address = info['address']
        token_symbol = symbol

        # Get price from OrderBook
        price = orderbook_contract.functions.get_price(token_address, usdc_address).call()

        # Adjust price according to decimals
        adjusted_price = price / (10 ** 18)

        # Get initial price from testnet_data.json
        initial_price = initial_prices[symbol]

        print(f"Price of {token_symbol} in {usdc_symbol}:")
        print(f"  Expected Initial Price: {initial_price}")
        print(f"  Price from OrderBook: {adjusted_price}")
        if abs(adjusted_price - initial_price) / initial_price > 0.01:  # Allow 1% difference
            print(f"  Warning: Price discrepancy detected!")
        else:
            print(f"  Price verified.")
        print()

    # Verify pool balances
    print("\nVerifying liquidity pool balances:\n")
    for symbol, info in tokens_info.items():
        token_address = info['address']
        token_contract = info['contract']
        # Get pool balance
        pool_balance = token_contract.functions.balanceOf(orderbook_address).call()
        # Adjust for decimals
        pool_balance_readable = pool_balance / (10 ** info['decimals'])
        print(f"Liquidity Pool Balance for {symbol}: {pool_balance_readable} {symbol}")
    print()

if __name__ == "__main__":
    main()
