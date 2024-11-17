import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from market_agents.agents.base_agent.agent import Agent
from market_agents.inference.message_models import LLMConfig
from agent_evm_interface.agent_evm_interface import EthereumInterface

async def main():
    # Initialize the Ethereum interface
    eth_interface = EthereumInterface()
    
    # Get all accounts from the interface
    accounts = eth_interface.accounts
    
    # Get token data from testnet data
    token_addresses = eth_interface.testnet_data['token_addresses']
    token_symbols = eth_interface.testnet_data['token_symbols']
    orderbook_address = eth_interface.testnet_data['orderbook_address']

    # Get the external tagged functions to use as tools
    tools = [
        getattr(eth_interface, attr_name)
        for attr_name in dir(eth_interface)
        if callable(getattr(eth_interface, attr_name)) and 
        hasattr(getattr(eth_interface, attr_name), '_external_tagged')
    ]

    # Create an agent with the Ethereum tools
    agent = Agent(
        role="Ethereum Transaction Assistant",
        persona="You are an AI assistant that helps with Ethereum transactions and token management.",
        system="""You are an expert in handling Ethereum transactions and token management.
        When asked about accounts or balances, always check the available addresses first.
        For any transaction, verify sufficient balances before proceeding.
        Always provide clear explanations of what you're doing.
        If any operation fails, provide a clear error message and suggested resolution.""",
        output_format="tool",
        tools=tools,
        llm_config=LLMConfig(
            client="openai",
            model="gpt-4o-mini",
            temperature=0.1  # Slight increase for more natural responses while maintaining consistency
        )
    )

    # Example tasks using data from the interface
    tasks = [
        f"Get the ETH balance for address {accounts[0]['address']}", 
        f"Get the ERC20 balance and allowance info for token {token_addresses[0]} ({token_symbols[0]}) at address {accounts[0]['address']}",
        f"Get info about the ERC20 token at address {token_addresses[0]}",
        f"Get the transfer history for the ERC20 token at {token_addresses[0]}",
        f"Get the swap history between tokens {token_addresses[0]} and {token_addresses[1]}",
        f"Get price and liquidity info for the token pair {token_addresses[0]} and {token_addresses[1]}",
        f"Send 0.1 ETH from {accounts[0]['address']} to {accounts[1]['address']} using private key {accounts[0]['private_key']}",
        f"Send 100 {token_symbols[0]} tokens from contract {token_addresses[0]} from {accounts[0]['address']} to {accounts[1]['address']} using private key {accounts[0]['private_key']}",
        f"Approve the orderbook at {orderbook_address} to spend 1000 {token_symbols[0]} tokens from contract {token_addresses[0]} using private key {accounts[0]['private_key']}",
        f"Swap 100 {token_symbols[0]} tokens from {token_addresses[0]} to {token_symbols[1]} at {token_addresses[1]} using private key {accounts[0]['private_key']}"
    ]

    # Execute tasks with error handling
    for task in tasks:
        try:
            print(f"\n\033[94mTask: {task}\033[0m")  # Blue color for tasks
            result = await agent.execute(task)
            print(f"\033[92mResult: {result}\033[0m")  # Green color for results
        except Exception as e:
            print(f"\033[91mError executing task '{task}': {str(e)}\033[0m")  # Red color for errors

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")