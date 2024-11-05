import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from market_agents.agents.base_agent.agent import Agent
from market_agents.inference.message_models import LLMConfig
from agent_evm_interface.agent_evm_interface_old import init_ethereum_interface

async def main():
    # Initialize the Ethereum interface
    eth_interface = init_ethereum_interface()
    
    # Get the external tagged functions to use as tools
    tools = []
    for attr_name in dir(eth_interface):
        attr = getattr(eth_interface, attr_name)
        if callable(attr) and hasattr(attr, '_external_tagged'):
            tools.append(attr)

    # Create an agent with the Ethereum tools
    agent = Agent(
        role="Ethereum Transaction Assistant",
        persona="You are an AI assistant that helps with Ethereum transactions and token management.",
        system="""You are an expert in handling Ethereum transactions and token management.
        When asked about accounts or balances, always check the available addresses first.
        For any transaction, verify sufficient balances before proceeding.
        Always provide clear explanations of what you're doing.""",
        output_format="tool",
        tools=tools,
        llm_config=LLMConfig(
            #client="vllm",
            #model="NousResearch/Hermes-3-Llama-3.1-8B",
            client="openai",
            model="gpt-4o-mini",
            temperature=0
        )
    )

    # Example tasks demonstrating all external functions
    tasks = [
        "What Ethereum addresses do I have available?",
        "What ETH balance for this address: '0x395D54F5403176936891B8e10cc735aFE838B965'",
        "Show me the list of whitelisted ERC20 tokens",
    ]

    # Execute tasks
    for task in tasks:
        print(f"\nTask: {task}")
        result = await agent.execute(task)
        print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
