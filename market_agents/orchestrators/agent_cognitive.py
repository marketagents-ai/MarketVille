# agent_cognitive.py

import asyncio
from datetime import datetime
import logging
from typing import List, Any, Dict
from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.logger_utils import (
    log_persona,
    log_perception,
    log_reflection
)

class AgentCognitiveProcessor:
    def __init__(self, ai_utils, data_inserter, logger: logging.Logger):
        self.ai_utils = ai_utils
        self.data_inserter = data_inserter
        self.logger = logger

    async def run_parallel_perceive(self, agents: List[MarketAgent], environment_name: str) -> List[Any]:
        perception_prompts = []
        for agent in agents:
            perception_prompt = await agent.perceive(environment_name, return_prompt=True)
            perception_prompts.append(perception_prompt)
        
        perceptions = await self.ai_utils.run_parallel_ai_completion(perception_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        
        # Log personas and perceptions, and update agent states
        for agent, perception in zip(agents, perceptions):
            log_persona(self.logger, agent.index, agent.persona)
            log_perception(
                self.logger, 
                agent.index, 
                perception.json_object.object if perception and perception.json_object else None
            )
            agent.last_perception = perception.json_object.object if perception.json_object else perception.str_content
            
        return perceptions

    async def run_parallel_action(self, agents: List[MarketAgent], environment_name: str) -> List[Any]:
        action_prompts = []
        for agent in agents:
            action_prompt = await agent.generate_action(environment_name, agent.last_perception, return_prompt=True)
            action_prompts.append(action_prompt)
        actions = await self.ai_utils.run_parallel_ai_completion(action_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        return actions

    async def run_parallel_reflect(self, agents: List[MarketAgent], environment_name: str) -> None:
        reflection_prompts = []
        agents_with_observations = []
        for agent in agents:
            if agent.last_observation:
                reflect_prompt = await agent.reflect(environment_name, return_prompt=True)
                reflection_prompts.append(reflect_prompt)
                agents_with_observations.append(agent)
                
        if reflection_prompts:
            reflections = await self.ai_utils.run_parallel_ai_completion(reflection_prompts, update_history=False)
            self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
            
            for agent, reflection in zip(agents_with_observations, reflections):
                if reflection.json_object:
                    # Log reflection
                    log_reflection(self.logger, agent.index, reflection.json_object.object)
                    
                    # Store in agent memory
                    agent.memory.append({
                        "type": "reflection",
                        "content": reflection.json_object.object.get("reflection", ""),
                        "strategy_update": reflection.json_object.object.get("strategy_update", ""),
                        "observation": agent.last_observation,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    self.logger.warning(f"No reflection JSON object for agent {agent.index}")
        else:
            self.logger.info("No reflections generated in this round.")