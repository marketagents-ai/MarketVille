import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from market_agents.agents.base_agent.agent import Agent
from market_agents.inference.message_models import LLMConfig
from agent_evm_interface.agent_evm_interface import init_ethereum_interface

async def main():
    # Initialize the Ethereum interface
    eth_interface = init_ethereum_interface()
    
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

    # Example tasks demonstrating all external functions
    tasks = [
        "What Ethereum addresses do I have available?",
        "What ETH balance for this address: '0x395D54F5403176936891B8e10cc735aFE838B965'",
        "Show me the list of whitelisted ERC20 tokens",
    ]

    # Execute tasks with error handling
    for task in tasks:
        try:
            print(f"\nTask: {task}")
            result = await agent.execute(task)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error executing task '{task}': {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")