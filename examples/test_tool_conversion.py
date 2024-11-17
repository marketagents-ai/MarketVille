import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_evm_interface.agent_evm_interface import EthereumInterface
from market_agents.agents.tool_caller.utils import function_to_json

def test_tool_conversion():
    # Initialize the Ethereum interface
    eth_interface = EthereumInterface()
    
    # Get all externally tagged functions
    external_tools = [
        getattr(eth_interface, attr_name)
        for attr_name in dir(eth_interface)
        if callable(getattr(eth_interface, attr_name)) and 
        hasattr(getattr(eth_interface, attr_name), '_external_tagged')
    ]

    print(f"\nFound {len(external_tools)} external tools:")
    for tool in external_tools:
        print(f"\n{'-'*50}")
        print(f"Converting {tool.__name__}...")
        
        # Convert the function to JSON schema
        try:
            json_schema = function_to_json(tool)
            print(f"\nJSON Schema:")
            print(json_schema)
        except Exception as e:
            print(f"Error converting {tool.__name__}: {str(e)}")

if __name__ == "__main__":
    test_tool_conversion()
