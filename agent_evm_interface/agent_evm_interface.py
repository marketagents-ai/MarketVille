import json
from web3 import Web3
from eth_account import Account
import random


def external(func):
    func._external_tagged = True
    return func

def init(func):
    return func

class EthereumInterface:
    def __init__(self, rpc_url: str = "http://localhost:8545"):
        self.rpc_url = rpc_url

        # load ../.mnemonic
        with open('../.mnemonic', 'r') as f:
            self.mnemonic = f.read().strip()

        with open( '../testnet_data.json', 'r') as f:
            self.testnet_data = json.load(f)
        
        # TODO: maybe find a better way to do this:
        Account.enable_unaudited_hdwallet_features()

        # get pk and address of the top 20 accounts
        self.accounts = []
        for i in range(20):
            account = Account.from_mnemonic(self.mnemonic, account_path=f"m/44'/60'/0'/0/{i}")
            self.accounts.append({
                'address': account.address,
                'private_key': account.key.hex()
            })

        print('Accounts:', self.accounts)
    
    @external
    def get_eth_balance(self, address: str) -> int:
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        balance = w3.eth.get_balance(address)
        return balance

    @external
    def get_erc20_balance(self, address: str, contract_address: str) -> int:
        erc20_abi = self.testnet_data['token_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        contract = w3.eth.contract(address=contract_address, abi=erc20_abi)
        balance = contract.functions.balanceOf(address).call()
        return balance

    @external
    def get_erc20_allowance(self, owner: str, spender: str, contract_address: str) -> int:
        erc20_abi = self.testnet_data['token_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        contract = w3.eth.contract(address=contract_address, abi=erc20_abi)
        allowance = contract.functions.allowance(owner, spender).call()
        return allowance

    @external
    def get_erc20_info(self, contract_address: str) -> dict:
        erc20_abi = self.testnet_data['token_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        contract = w3.eth.contract(address=contract_address, abi=erc20_abi)
        total_supply = contract.functions.totalSupply().call()
        decimals = contract.functions.decimals().call()
        symbol = contract.functions.symbol().call()
        return {
            'total_supply': total_supply,
            'decimals': decimals,
            'symbol': symbol
        }

    @external
    def get_erc20_transfer_events(self, contract_address: str, from_block: int, to_block: int) -> list:
        erc20_abi = self.testnet_data['token_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        contract = w3.eth.contract(address=contract_address, abi=erc20_abi)
        transfer_events = contract.events.Transfer().get_logs(from_block=from_block, to_block=to_block)
        return transfer_events

    @external
    def get_limit_order_history(self, token: str) -> list:
        
        # on the orderbook:
        # event BuyOrder(address indexed user, address indexed token, uint256 amount, uint256 price, uint256 new_price);
        # event SellOrder(address indexed user, address indexed token, uint256 amount, uint256 price, uint256 new_price);
    
        orderbook_address = self.testnet_data['orderbook_address']
        orderbook_abi = self.testnet_data['orderbook_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)
        
        # find events with the token address
        buy_events = contract.events.BuyOrder().get_logs(from_block=0, to_block='latest', argument_filters={'token': token})
        sell_events = contract.events.SellOrder().get_logs(from_block=0, to_block='latest', argument_filters={'token': token})

        # zip the events together
        events = sorted(buy_events + sell_events, key=lambda x: x['blockNumber'])
        return events
        
    @external
    def get_price(self, token: str) -> int:
        # get the latest price of the token in ETH
        orderbook_address = self.testnet_data['orderbook_address']
        orderbook_abi = self.testnet_data['orderbook_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)

        # get the latest price
        price = contract.functions.get_price(token).call()
        return price

    @external
    def send_eth(self, to: str, amount: int, private_key: str) -> str:
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        account = Account.from_key(private_key)
        
        # Get the nonce right before building the transaction
        nonce = w3.eth.get_transaction_count(account.address)
        
        signed_txn = account.sign_transaction({
            'nonce': nonce,
            'to': to,
            'value': amount,
            'gas': 2000000,
            'gasPrice': w3.to_wei('50', 'gwei')
        })
        
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        return tx_hash.hex()

    @external
    def send_erc20(self, to: str, amount: int, contract_address: str, private_key: str) -> str:
        """
        Send ERC20 tokens with automated gas estimation and current gas price
        
        Args:
            to: Recipient address
            amount: Amount of tokens to send
            contract_address: ERC20 contract address
            private_key: Sender's private key
        
        Returns:
            Transaction hash as hex string
        """
        erc20_abi = self.testnet_data['token_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        account = Account.from_key(private_key)
        contract = w3.eth.contract(address=contract_address, abi=erc20_abi)
        
        # Get the nonce
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Get current gas price with a small buffer (1.1x)
        gas_price = int(w3.eth.gas_price * 1.1)
        
        # Build transaction with empty gas estimate
        tx_params = {
            'nonce': nonce,
            'gasPrice': gas_price,
            'from': account.address
        }
        
        # Estimate gas for this specific transaction
        gas_estimate = contract.functions.transfer(to, amount).estimate_gas(tx_params)
        
        # Add 10% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.1)
        
        # Build final transaction with gas parameters
        tx_params['gas'] = gas_limit
        data = contract.functions.transfer(to, amount).build_transaction(tx_params)
        
        # Sign and send transaction
        signed_txn = account.sign_transaction(data)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        
        return tx_hash.hex()

    def mint_erc20(self, to: str, amount: int, contract_address: str, minter_private_key: str) -> str:
        erc20_abi = self.testnet_data['token_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        minter_account = Account.from_key(minter_private_key)
        contract = w3.eth.contract(address=contract_address, abi=erc20_abi)
        
        # Get the nonce
        nonce = w3.eth.get_transaction_count(minter_account.address)
        
        # Get current gas price with a small buffer (1.1x)
        gas_price = int(w3.eth.gas_price * 1.1)
        
        # Build transaction with empty gas estimate
        tx_params = {
            'nonce': nonce,
            'gasPrice': gas_price,
            'from': minter_account.address
        }
        
        # Estimate gas for this specific transaction
        gas_estimate = contract.functions.mint(to, amount).estimate_gas(tx_params)
        
        # Add 10% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.1)
        
        # Build final transaction with gas parameters
        tx_params['gas'] = gas_limit
        mint_data = contract.functions.mint(to, amount).build_transaction(tx_params)
        
        # Sign and send transaction
        mint_signed_txn = minter_account.sign_transaction(mint_data)
        mint_tx_hash = w3.eth.send_raw_transaction(mint_signed_txn.raw_transaction)
        
        return mint_tx_hash.hex()

    @external
    def approve_erc20(self, spender: str, amount: int, contract_address: str, private_key: str) -> str:
        erc20_abi = self.testnet_data['token_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        account = Account.from_key(private_key)
        contract = w3.eth.contract(address=contract_address, abi=erc20_abi)
        
        # Get the nonce
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Get current gas price with a small buffer (1.1x)
        gas_price = int(w3.eth.gas_price * 1.1)
        
        # Build transaction with empty gas estimate
        tx_params = {
            'nonce': nonce,
            'gasPrice': gas_price,
            'from': account.address
        }
        
        # Estimate gas for this specific transaction
        gas_estimate = contract.functions.approve(spender, amount).estimate_gas(tx_params)
        
        # Add 10% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.1)
        
        # Build final transaction with gas parameters
        tx_params['gas'] = gas_limit
        data = contract.functions.approve(spender, amount).build_transaction(tx_params)
        
        # Sign and send transaction
        signed_txn = account.sign_transaction(data)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        
        return tx_hash.hex()

    # on the contract: function place_limit_buy_order(address token_address, uint256 amount, uint256 limit_price) public {
    @external
    def place_limit_buy_order(self, token_address: str, amount: int, limit_price: int, private_key: str) -> str:
        orderbook_address = self.testnet_data['orderbook_address']
        orderbook_abi = self.testnet_data['orderbook_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        account = Account.from_key(private_key)
        contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)
        
        # Get the nonce
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Get current gas price with a small buffer (1.1x)
        gas_price = int(w3.eth.gas_price * 1.1)
        
        # Build transaction with empty gas estimate
        tx_params = {
            'nonce': nonce,
            'gasPrice': gas_price,
            'from': account.address
        }
        
        # Estimate gas for this specific transaction
        gas_estimate = contract.functions.place_limit_buy_order(token_address, amount, limit_price).estimate_gas(tx_params)
        
        # Add 10% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.1)
        
        # Build final transaction with gas parameters
        tx_params['gas'] = gas_limit
        data = contract.functions.place_limit_buy_order(token_address, amount, limit_price).build_transaction(tx_params)
        
        # Sign and send transaction
        signed_txn = account.sign_transaction(data)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        
        return tx_hash.hex()
    
    # on the contract: function place_limit_sell_order(address token_address, uint256 amount, uint256 limit_price) public {
    @external
    def place_limit_sell_order(self, token_address: str, amount: int, limit_price: int, private_key: str) -> str:
        orderbook_address = self.testnet_data['orderbook_address']
        orderbook_abi = self.testnet_data['orderbook_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        account = Account.from_key(private_key)
        contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)
        
        # Get the nonce
        nonce = w3.eth.get_transaction_count(account.address)
        
        # Get current gas price with a small buffer (1.1x)
        gas_price = int(w3.eth.gas_price * 1.1)
        
        # Build transaction with empty gas estimate
        tx_params = {
            'nonce': nonce,
            'gasPrice': gas_price,
            'from': account.address
        }
        
        # Estimate gas for this specific transaction
        gas_estimate = contract.functions.place_limit_sell_order(token_address, amount, limit_price).estimate_gas(tx_params)
        
        # Add 10% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.1)
        
        # Build final transaction with gas parameters
        tx_params['gas'] = gas_limit
        data = contract.functions.place_limit_sell_order(token_address, amount, limit_price).build_transaction(tx_params)
        
        # Sign and send transaction
        signed_txn = account.sign_transaction(data)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        
        return tx_hash.hex()

@init
def initialize_evm_interface() -> EthereumInterface:
    ei = EthereumInterface()
    return ei


if __name__ == '__main__':
    print('Initializing EVM Interface...')
    ei = initialize_evm_interface()

    orderbook_address = ei.testnet_data['orderbook_address']
    orderbook_abi = ei.testnet_data['orderbook_abi']

    erc20_abi = ei.testnet_data['token_abi']
    erc20_addresses = ei.testnet_data['token_addresses']
    erc20_token_symbols = ei.testnet_data['token_symbols']


    print('Calling get_eth_balance...')
    for account in ei.accounts[:2]:
        eth_balance = ei.get_eth_balance(account['address'])
        print("~"*50)
        print(f'Account: {account["address"]}')
        print(f'  -ETH Balance: {eth_balance}')
        print()

        # get all erc20 balances
        for i, erc20_address in enumerate(erc20_addresses):
            erc20_balance = ei.get_erc20_balance(account['address'], erc20_address)
            print(f'  -{erc20_token_symbols[i]} Balance: {erc20_balance}')
        print()

        # get all erc20 allowances for the orderbook
        for i, erc20_address in enumerate(erc20_addresses):
            erc20_allowance = ei.get_erc20_allowance(account['address'], orderbook_address, erc20_address)
            print(f'  -{erc20_token_symbols[i]} Allowance: {erc20_allowance}')
        print()


    print('Calling get_erc20_info...')
    for erc20_address in erc20_addresses:
        erc20_info = ei.get_erc20_info(erc20_address)
        print("~"*50)
        print(f'ERC20 Contract: {erc20_address}')
        print(f'  -Total Supply: {erc20_info["total_supply"]}')
        print(f'  -Decimals: {erc20_info["decimals"]}')
        print(f'  -Symbol: {erc20_info["symbol"]}')
        print()

        print('Calling get_erc20_transfer_events...')
        transfer_events = ei.get_erc20_transfer_events(erc20_address, 0, 'latest')
        print(f'  -Transfer Events: {len(transfer_events)}')

    print('Calling send_eth...')
    # send 0.00001 from each account to the next account, round robin
    for i in range(len(ei.accounts)):
        account = ei.accounts[i]
        next_account = ei.accounts[(i+1)%len(ei.accounts)]

        before_balance_sender = ei.get_eth_balance(account['address'])
        before_balance_receiver = ei.get_eth_balance(next_account['address'])

        tx_hash = ei.send_eth(next_account['address'], 10000, account['private_key'])

        after_balance_sender = ei.get_eth_balance(account['address'])
        after_balance_receiver = ei.get_eth_balance(next_account['address'])

        print("~"*50)
        print(f'Sender: {account["address"]}')
        print(f'Receiver: {next_account["address"]}')
        print(f'  -Before Balance (Sender): {before_balance_sender}')
        print(f'  -Before Balance (Receiver): {before_balance_receiver}')
        print(f'  -Tx Hash: {tx_hash}')
        print(f'  -After Balance (Sender): {after_balance_sender}')
        print(f'  -After Balance (Receiver): {after_balance_receiver}')
        print()

    print('Calling send_erc20...')
    # send 100 from each account to the next account, round robin
    for i in range(len(ei.accounts)):
        account = ei.accounts[i]
        next_account = ei.accounts[(i+1)%len(ei.accounts)]
        erc20_address = erc20_addresses[0]

        # mint 100 tokens to the sender
        ei.mint_erc20(account['address'], 100, erc20_address, ei.accounts[0]['private_key'])

        before_balance_sender = ei.get_erc20_balance(account['address'], erc20_address)
        before_balance_receiver = ei.get_erc20_balance(next_account['address'], erc20_address)

        tx_hash = ei.send_erc20(next_account['address'], 100, erc20_address, account['private_key'])

        after_balance_sender = ei.get_erc20_balance(account['address'], erc20_address)
        after_balance_receiver = ei.get_erc20_balance(next_account['address'], erc20_address)

        print("~"*50)
        print(f'Sender: {account["address"]}')
        print(f'Receiver: {next_account["address"]}')
        print(f'ERC20 Contract: {erc20_address}')
        print(f'  -Before Balance (Sender): {before_balance_sender}')
        print(f'  -Before Balance (Receiver): {before_balance_receiver}')
        print(f'  -Tx Hash: {tx_hash}')
        print(f'  -After Balance (Sender): {after_balance_sender}')
        print(f'  -After Balance (Receiver): {after_balance_receiver}')
        print()



    print('Calling approve_erc20...')
    # approve 100 from each account to the next account, round robin
    for i in range(len(ei.accounts)):
        account = ei.accounts[i]
        next_account = ei.accounts[(i+1)%len(ei.accounts)]
        erc20_address = erc20_addresses[0]

        before_allowance_sender = ei.get_erc20_allowance(account['address'], orderbook_address, erc20_address)
        before_allowance_receiver = ei.get_erc20_allowance(next_account['address'], orderbook_address, erc20_address)

        tx_hash = ei.approve_erc20(orderbook_address, 100, erc20_address, account['private_key'])

        after_allowance_sender = ei.get_erc20_allowance(account['address'], orderbook_address, erc20_address)
        after_allowance_receiver = ei.get_erc20_allowance(next_account['address'], orderbook_address, erc20_address)

        print("~"*50)
        print(f'Sender: {account["address"]}')
        print(f'Receiver: {next_account["address"]}')
        print(f'ERC20 Contract: {erc20_address}')
        print(f'  -Before Allowance (Sender): {before_allowance_sender}')
        print(f'  -Before Allowance (Receiver): {before_allowance_receiver}')
        print(f'  -Tx Hash: {tx_hash}')
        print(f'  -After Allowance (Sender): {after_allowance_sender}')
        print(f'  -After Allowance (Receiver): {after_allowance_receiver}')
        print()

    print('Testing limit orders...')
    # Place buy and sell orders for the first token
    test_token = erc20_addresses[1]
    price_token = erc20_addresses[0]

    # Get current price
    current_price = ei.get_price(test_token)
    print(f'Current price for {erc20_token_symbols[1]}: {current_price}')
    
    # Place some test orders using the first few accounts
    test_accounts = ei.accounts[:3]
    
    
    # set allowance for the orderbook for each token
    for account in test_accounts:
        ei.approve_erc20(orderbook_address, 10000000000000000000000000000000000000, test_token, account['private_key'])
        ei.approve_erc20(orderbook_address, 10000000000000000000000000000000000000, price_token, account['private_key'])
        print(f'Allowances set for {account["address"]}')

    # Test buy orders
    print('\nPlacing buy orders...')
    for i, account in enumerate(test_accounts):
        
        buy_price = int(current_price)  
        amount = 1  # Amount to buy
        
        print(f'\nAccount {i} placing buy order:')
        print(f'  Address: {account["address"]}')
        print(f'  Amount: {amount}')
        print(f'  Price: {buy_price}')
        
        try:
            tx_hash = ei.place_limit_buy_order(
                test_token,
                amount,
                buy_price,
                account['private_key']
            )
            print(f'  Buy order placed. TX Hash: {tx_hash}')
        except Exception as e:
            print(f'  Error placing buy order: {str(e)}')
    
    # Test sell orders
    print('\nPlacing sell orders...')
    for i, account in enumerate(test_accounts):
        
        sell_price = int(current_price)  
        amount = 1  # Amount to sell
        
        print(f'\nAccount {i} placing sell order:')
        print(f'  Address: {account["address"]}')
        print(f'  Amount: {amount}')
        print(f'  Price: {sell_price}')
        
        try:
            tx_hash = ei.place_limit_sell_order(
                test_token,
                amount,
                sell_price,
                account['private_key']
            )
            print(f'  Sell order placed. TX Hash: {tx_hash}')
        except Exception as e:
            print(f'  Error placing sell order: {str(e)}')
    
    # Get order history
    print('\nGetting order history...')
    order_history = ei.get_limit_order_history(test_token)
    print(f'Total orders found: {len(order_history)}')
    for event in order_history[-5:]:  # Show last 5 orders
        print(f'Order event:')
        print(f'  Block: {event["blockNumber"]}')
        print(f'  Transaction: {event["transactionHash"].hex()}')
        print(f'  Event type: {event["event"]}')
        print(f'  User: {event["args"]["user"]}')
        print(f'  Amount: {event["args"]["amount"]}')
        print(f'  Price: {event["args"]["price"]}')
        print()
