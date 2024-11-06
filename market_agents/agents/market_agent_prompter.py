import json
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import yaml
import os

def json_to_markdown(data: Union[Dict, List, Any], indent: int = 0) -> str:
    """Convert JSON/dict data to a markdown formatted string."""
    if data is None:
        return "None"
    
    if isinstance(data, str):
        # Properly escape the string for JSON compatibility
        return json.dumps(data)[1:-1]  # Remove the quotes but keep the escaping
    
    if isinstance(data, (int, float, bool)):
        return str(data)
    
    if isinstance(data, list):
        if not data:
            return "[]"
        markdown = "\n"
        for item in data:
            item_str = json_to_markdown(item, indent + 1).lstrip()
            markdown += "  " * indent + "- " + item_str + "\n"
        return markdown.rstrip()
    
    if isinstance(data, dict):
        if not data:
            return "{}"
        markdown = "\n"
        for key, value in data.items():
            key_str = json.dumps(str(key))[1:-1]  # Properly escape the key
            value_str = json_to_markdown(value, indent + 1)
            if isinstance(value, (dict, list)):
                markdown += "  " * indent + f"- {key_str}:{value_str}\n"
            else:
                markdown += "  " * indent + f"- {key_str}: {value_str}\n"
        return markdown.rstrip()
    
    return json.dumps(str(data))[1:-1]  

class AgentPromptVariables(BaseModel):
    environment_name: str
    environment_info: Any
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    perception: Optional[Any] = None
    observation: Optional[Any] = None
    action_space: Dict[str, Any] = {}
    last_action: Optional[Any] = None
    reward: Optional[float] = None
    previous_strategy: Optional[str] = None

class MarketAgentPromptManager(BaseModel):
    prompts: Dict[str, str] = Field(default_factory=dict)
    prompt_file: str = Field(default="market_agents/agents/configs/prompts/market_agent_prompt.yaml")

    def __init__(self, **data: Any):
        super().__init__(**data)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        full_path = os.path.join(project_root, self.prompt_file)
        
        try:
            with open(full_path, 'r') as file:
                self.prompts = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {full_path}")
        
    def format_prompt(self, prompt_type: str, variables: Dict[str, Any]) -> str:
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Auto-convert any JSON/dict/list values to markdown
        markdown_vars = {}
        for key, value in variables.items():
            if isinstance(value, (dict, list)):
                markdown_vars[key] = json_to_markdown(value)
            else:
                markdown_vars[key] = value
        
        try:
            return self.prompts[prompt_type].format(**markdown_vars)
        except Exception as e:
            raise e

    def get_perception_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('perception', variables)

    def get_action_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('action', variables)

    def get_reflection_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('reflection', variables)