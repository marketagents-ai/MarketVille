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
    def get_swap_history(self, sourceToken: str, targetToken: str) -> list:
        
        orderbook_address = self.testnet_data['orderbook_address']
        orderbook_abi = self.testnet_data['orderbook_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)
        # on the orderbook: event Swap(address indexed user, address indexed sourceToken, address indexed targetToken, uint256 sourceAmount, uint256 targetAmount);
        # find events with targetToken = address_1 or address_2 and sourceToken is the other
        swap_events = contract.events.Swap().get_logs(from_block=0, to_block='latest', argument_filters={'sourceToken': sourceToken, 'targetToken': targetToken})
        
        return swap_events

    @external
    def get_pair_info(self, token0: str, token1: str) -> dict:
        orderbook_address = self.testnet_data['orderbook_address']
        orderbook_abi = self.testnet_data['orderbook_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        orderbook_contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)
        token0_contract = w3.eth.contract(address=token0, abi=self.testnet_data['token_abi'])
        token1_contract = w3.eth.contract(address=token1, abi=self.testnet_data['token_abi'])

        token0_balance = token0_contract.functions.balanceOf(orderbook_address).call()
        token1_balance = token1_contract.functions.balanceOf(orderbook_address).call()
        token0_price_in_token1 = orderbook_contract.functions.get_price(token0, token1).call()
        token1_price_in_token0 = orderbook_contract.functions.get_price(token1, token0).call()

        return {
            'token0_balance': token0_balance,
            'token1_balance': token1_balance,
            'token0_price_in_token1': token0_price_in_token1,
            'token1_price_in_token0': token1_price_in_token0
        }

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

    @external
    def swap(self, source_token_address: str, source_token_amount: int, target_token_address: str, private_key: str) -> str:
        orderbook_address = self.testnet_data['orderbook_address']
        orderbook_abi = self.testnet_data['orderbook_abi']
        w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        account = Account.from_key(private_key)
        orderbook_contract = w3.eth.contract(address=orderbook_address, abi=orderbook_abi)
        
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
        gas_estimate = orderbook_contract.functions.swap(source_token_address, source_token_amount, target_token_address).estimate_gas(tx_params)
        
        # Add 10% buffer to gas estimate
        gas_limit = int(gas_estimate * 1.1)
        
        # Build final transaction with gas parameters
        tx_params['gas'] = gas_limit
        data = orderbook_contract.functions.swap(source_token_address, source_token_amount, target_token_address).build_transaction(tx_params)
        
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

    # get swap history for every token pair
    print('Calling get_swap_history...')
    for i, sourceToken in enumerate(erc20_addresses):
        for j, targetToken in enumerate(erc20_addresses):
            if i == j:
                continue
            swap_history = ei.get_swap_history(sourceToken, targetToken)
            if len(swap_history) > 0:
                print("~"*50)
                print(f'Source Token: {erc20_token_symbols[i]}')
                print(f'Target Token: {erc20_token_symbols[j]}')
                print(f'  -Swap History: {len(swap_history)}')
                print()

    # get pair info for every token pair
    print('Calling get_pair_info...')
    for i, token0 in enumerate(erc20_addresses):
        for j, token1 in enumerate(erc20_addresses):
            if i == j:
                continue
            pair_info = ei.get_pair_info(token0, token1)
            print("~"*50)
            print(f'Token0: {erc20_token_symbols[i]}')
            print(f'Token1: {erc20_token_symbols[j]}')
            print(f'  -Token0 Balance: {pair_info["token0_balance"]}')
            print(f'  -Token1 Balance: {pair_info["token1_balance"]}')
            print(f'  -Token0 Price in Token1: {pair_info["token0_price_in_token1"]}')
            print(f'  -Token1 Price in Token0: {pair_info["token1_price_in_token0"]}')
            print()


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

    print('Calling swap...')

    print("performing 100 random small swaps")
    for i in range(100):
        source_token_address = random.choice(erc20_addresses)
        target_token_address = random.choice(erc20_addresses)
        while source_token_address == target_token_address:
            target_token_address = random.choice(erc20_addresses)
        source_token_amount = random.randint(1, 100)

        account = random.choice(ei.accounts)

        # set allowance for the source token to the orderbook
        tx_hash = ei.approve_erc20(orderbook_address, source_token_amount, source_token_address, account['private_key'])
        print(f"approve {source_token_amount} from {account['address']} to orderbook tx_hash: {tx_hash}")

        # print pair info
        pair_info = ei.get_pair_info(source_token_address, target_token_address)
        print(f"pair info for {source_token_address} and {target_token_address}")
        print()
        print(pair_info)
        print()

        # mint tokens to the user
        ei.mint_erc20(account['address'], source_token_amount, source_token_address, ei.accounts[0]['private_key'])

        print(f"attempting swap from {source_token_address} to {target_token_address} with amount {source_token_amount}")
        tx_hash = ei.swap(source_token_address, source_token_amount, target_token_address, account['private_key'])
        print(f"swap from {source_token_address} to {target_token_address} with amount {source_token_amount} tx_hash: {tx_hash}")