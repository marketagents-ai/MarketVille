# groupchat_orchestrator.py

import asyncio
import json
import logging
import aiohttp
from typing import List, Dict, Any, Optional
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatActionSpace, GroupChatMessage, GroupChatObservationSpace
from pydantic import Field

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.config import GroupChatConfig, OrchestratorConfig
from market_agents.orchestrators.logger_utils import (
    log_section,
    log_environment_setup,
    log_persona,
    log_perception,
    log_action,
    log_reflection,
    log_round,
    log_completion,
    print_ascii_art,
    log_cohort_formation,
    log_topic_proposal,
    log_sub_round_start,
    log_group_chat_summary,
    log_group_message
)
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter
from datetime import datetime

class GroupChatTracker:
    def __init__(self):
        self.messages: List[GroupChatMessage] = []
        self.topics: Dict[str, str] = {}

    def add_message(self, message: GroupChatMessage):
        self.messages.append(message)

    def add_topic(self, cohort_id: str, topic: str):
        self.topics[cohort_id] = topic

    def get_summary(self):
        return {
            "total_messages": len(self.messages),
            "total_topics": len(self.topics)
        }

class GroupChatOrchestrator(BaseEnvironmentOrchestrator):
    environment_name: str = Field(default='group_chat')
    environments: Dict[str, MultiAgentEnvironment] = Field(default_factory=dict)  # Added this
    trackers: Dict[str, GroupChatTracker] = Field(default_factory=dict)
    orchestrator_config: OrchestratorConfig = Field(default=None)
    groupchat_api_url: str = Field(default="http://localhost:8001")
    agents: List[MarketAgent]
    data_inserter: SimulationDataInserter
    logger: Optional[logging.Logger] = None
    sub_rounds_per_step: int = Field(default=2)
    cohort_size: int = Field(default=2)
    cohorts: Dict[str, List[MarketAgent]] = Field(default_factory=dict)
    agent_dict: Dict[str, MarketAgent] = Field(default_factory=dict)
    topic_proposers: Dict[str, str] = Field(default_factory=dict)
    last_env_state: Dict[str, Any] = Field(default_factory=dict)
    round_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(
        self,
        config: GroupChatConfig,
        orchestrator_config: OrchestratorConfig,
        agents: List[MarketAgent],
        ai_utils,
        data_inserter: SimulationDataInserter,
        logger=None
    ):
        super().__init__(
            config=config,
            agents=agents,
            ai_utils=ai_utils,
            data_inserter=data_inserter,
            logger=logger
        )
        self.orchestrator_config = orchestrator_config
        self.groupchat_api_url = config.groupchat_api_url
        self.sub_rounds_per_step = config.sub_rounds
        self.cohort_size = config.group_size
        self.agents = agents
        self.agent_dict = {agent.id: agent for agent in agents}
        self.logger = logger or logging.getLogger(__name__)

    async def check_api_health(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.groupchat_api_url}/health") as resp:
                    if resp.status == 200:
                        self.logger.info("GroupChat API is healthy")
                        return True
                    else:
                        self.logger.error(f"GroupChat API health check failed: {resp.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Could not connect to GroupChat API: {e}")
            return False

    async def setup_environment(self):
        log_section(self.logger, "CONFIGURING GROUP CHAT ENVIRONMENT")
        
        # Check API health first
        if not await self.check_api_health():
            raise RuntimeError("GroupChat API is not available")
        
        # Register agents
        await self.register_agents()
        
        # Form cohorts and create environments
        await self.form_cohorts()
        
        # Create environments for each cohort
        for cohort_id, cohort_agents in self.cohorts.items():
            group_chat = GroupChat(
                max_rounds=self.config.max_rounds,
                sequential=False,
            )
            group_chat_env = MultiAgentEnvironment(
                name=f"{self.config.name}_{cohort_id}",
                address=f"{self.config.address}_{cohort_id}",
                max_steps=self.config.max_rounds,
                action_space=GroupChatActionSpace(),
                observation_space=GroupChatObservationSpace(),
                mechanism=group_chat
            )
            self.environments[cohort_id] = group_chat_env
            self.trackers[cohort_id] = GroupChatTracker()

    async def register_agents(self):
        # Optionally register agents via API
        async with aiohttp.ClientSession() as session:
            tasks = []
            for agent in self.agents:
                payload = {
                    "id": agent.id,
                    "index": agent.index
                }
                tasks.append(
                    self.register_agent(session, payload)
                )
            results = await asyncio.gather(*tasks)
            # Check results
            for success, agent_id in results:
                if success:
                    self.logger.info(f"Registered agent {agent_id}")
                else:
                    self.logger.error(f"Failed to register agent {agent_id}")

    async def register_agent(self, session, payload):
        try:
            async with session.post(f"{self.groupchat_api_url}/register_agent", json=payload) as resp:
                if resp.status == 200:
                    return True, payload["id"]
                else:
                    self.logger.error(f"Failed to register agent {payload['id']}: {resp.status}")
                    return False, payload["id"]
        except Exception as e:
            self.logger.error(f"Exception while registering agent {payload['id']}: {e}")
            return False, payload["id"]

    async def form_cohorts(self):
        # Use the API to form cohorts
        agent_ids = [agent.id for agent in self.agents]
        payload = {
            "agent_ids": agent_ids,
            "cohort_size": self.cohort_size
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.groupchat_api_url}/form_cohorts", json=payload) as resp:
                if resp.status == 200:
                    cohorts_info = await resp.json()
                    # Assign cohorts to agents and build self.cohorts
                    for cohort in cohorts_info:
                        cohort_id = cohort["cohort_id"]
                        cohort_agent_ids = cohort["agent_ids"]
                        cohort_agents = [self.agent_dict[agent_id] for agent_id in cohort_agent_ids]
                        self.cohorts[cohort_id] = cohort_agents
                        
                        # Create environment for this cohort if not exists
                        if cohort_id not in self.environments:
                            group_chat = GroupChat(
                                max_rounds=self.config.max_rounds,
                                sequential=False,
                            )
                            self.environments[cohort_id] = MultiAgentEnvironment(
                                name=f"{self.config.name}_{cohort_id}",
                                address=f"{self.config.address}_{cohort_id}",
                                max_steps=self.config.max_rounds,
                                action_space=GroupChatActionSpace(),
                                observation_space=GroupChatObservationSpace(),
                                mechanism=group_chat
                            )
                        
                        # Assign environment to agents
                        for agent in cohort_agents:
                            if not hasattr(agent, 'environments') or agent.environments is None:
                                agent.environments = {}
                            agent.cohort_id = cohort_id
                            agent.environments[self.environment_name] = self.environments[cohort_id]
                            
                        log_cohort_formation(self.logger, cohort_id, [agent.index for agent in cohort_agents])
                    self.logger.info(f"Cohorts formed: {[cohort['cohort_id'] for cohort in cohorts_info]}")
                else:
                    self.logger.error(f"Failed to form cohorts: {resp.status}")
                    raise Exception("Failed to form cohorts")

    async def run_environment(self, round_num: int):
        log_round(self.logger, round_num)
        # Select topic proposers via API
        await self.select_topic_proposers()
        # Collect proposed topics from proposers
        await self.collect_proposed_topics(round_num)
        # Run sub-rounds
        for sub_round in range(1, self.sub_rounds_per_step + 1):
            log_sub_round_start(self.logger, 'All Cohorts', sub_round)
            # For each cohort, run the sub-round
            tasks = []
            for cohort_id, cohort_agents in self.cohorts.items():
                task = asyncio.create_task(
                    self.run_group_chat_sub_round(
                        cohort_id=cohort_id,
                        round_num=round_num,
                        sub_round_num=sub_round,
                        cohort_agents=cohort_agents
                    )
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
        # After sub-rounds, run reflection
        log_section(self.logger, "AGENT REFLECTIONS")
        await self.run_reflection(round_num)
        # Store round summary
        round_summary = await self.get_round_summary(round_num)
        self.round_summaries.append(round_summary)

    async def select_topic_proposers(self):
        # For each cohort, select the topic proposer via API
        async with aiohttp.ClientSession() as session:
            tasks = []
            for cohort_id, cohort_agents in self.cohorts.items():
                agent_ids = [agent.id for agent in cohort_agents]
                payload = {
                    "cohort_id": cohort_id,
                    "agent_ids": agent_ids
                }
                task = self.select_proposer(session, cohort_id, payload)
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            # Store proposers
            self.topic_proposers = {}
            for cohort_id, proposer_id in results:
                if proposer_id:
                    self.topic_proposers[cohort_id] = proposer_id
                    self.logger.info(f"Selected proposer {proposer_id} for cohort {cohort_id}")
                else:
                    self.logger.error(f"Failed to select proposer for cohort {cohort_id}")

    async def select_proposer(self, session, cohort_id, payload):
        try:
            async with session.post(f"{self.groupchat_api_url}/select_proposer", json=payload) as resp:
                if resp.status == 200:
                    proposer_info = await resp.json()
                    proposer_id = proposer_info['proposer_id']
                    return cohort_id, proposer_id
                else:
                    self.logger.error(f"Failed to select proposer for cohort {cohort_id}: {resp.status}")
                    return cohort_id, None
        except Exception as e:
            self.logger.error(f"Exception while selecting proposer for cohort {cohort_id}: {e}")
            return cohort_id, None

    async def collect_proposed_topics(self, round_num: int):
        proposer_prompts = []
        proposer_agents = []

        # Collect prompts for topic proposers
        for cohort_id, proposer_id in self.topic_proposers.items():
            proposer = self.agent_dict[proposer_id]
            # Set the topic proposer system message
            good_name = self.orchestrator_config.agent_config.good_name
            proposer.system = f"You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for group discussion about {good_name}."
            # Create the prompt
            prompt = await proposer.generate_action(
                self.environment_name,
                f"Consider recent events, trends, or news related to {good_name}. Propose a specific topic for discussion that would be relevant to market participants.",
                return_prompt=True  # Important: return the prompt instead of executing it
            )
            proposer_prompts.append(prompt)
            proposer_agents.append((cohort_id, proposer))

        # Run all prompts in parallel
        topic_proposals = await self.ai_utils.run_parallel_ai_completion(proposer_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())

        # Post topics via API
        async with aiohttp.ClientSession() as session:
            tasks = []
            for (cohort_id, proposer), proposal in zip(proposer_agents, topic_proposals):
                topic = self.extract_topic_from_proposal(proposal)
                if topic:
                    payload = {
                        "agent_id": proposer.id,
                        "cohort_id": cohort_id,
                        "topic": topic,
                        "round_num": round_num
                    }
                    task = self.propose_topic(session, cohort_id, payload)
                    tasks.append(task)
                    log_topic_proposal(self.logger, cohort_id, proposer.index, topic)
                else:
                    self.logger.error(f"Failed to extract topic from proposer {proposer.id} in cohort {cohort_id}")
            await asyncio.gather(*tasks)

    async def propose_topic(self, session, cohort_id, payload):
        try:
            async with session.post(f"{self.groupchat_api_url}/propose_topic", json=payload) as resp:
                if resp.status == 200:
                    self.logger.info(f"Topic proposed for cohort {cohort_id}")
                else:
                    self.logger.error(f"Failed to propose topic for cohort {cohort_id}: {resp.status}")
        except Exception as e:
            self.logger.error(f"Exception while proposing topic for cohort {cohort_id}: {e}")

    def extract_topic_from_proposal(self, proposal):
        try:
            if proposal.json_object:
                action_content = proposal.json_object.object
                if 'content' in action_content:
                    if isinstance(action_content['content'], dict) and 'action' in action_content['content']:
                        topic = action_content['content']['action']['content']
                    else:
                        topic = action_content['content']
                elif 'action' in action_content:
                    topic = action_content['action']['content']
                else:
                    self.logger.warning("Unexpected topic action structure")
                    topic = None
            else:
                topic = proposal.str_content.strip() if proposal.str_content else None
            return topic
        except Exception as e:
            self.logger.error(f"Error extracting topic: {e}")
            return None

    async def run_group_chat_sub_round(self, cohort_id: str, round_num: int, sub_round_num: int, cohort_agents: List[MarketAgent]):
        # For each agent in the cohort
        async with aiohttp.ClientSession() as session:
            # Get the topic for the cohort
            topic = await self.get_topic(session, cohort_id)
            if not topic:
                self.logger.error(f"No topic found for cohort {cohort_id}")
                return

            # Agents get messages
            messages = await self.get_messages(session, cohort_id)

            # Agents perceive messages
            perception_prompts = []
            for agent in cohort_agents:
                # Set agent's system message
                self.set_agent_system_message(agent, topic, round_num, sub_round_num)
                agent.last_observation = messages
                perception_prompt = await agent.perceive(self.environment_name, return_prompt=True)
                perception_prompts.append(perception_prompt)
            perceptions = await self.ai_utils.run_parallel_ai_completion(perception_prompts, update_history=False)
            self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())

            # Agents generate actions (messages)
            action_prompts = []
            for agent, perception in zip(cohort_agents, perceptions):
                agent.last_perception = perception.json_object.object if perception.json_object else perception.str_content
                action_prompt = await agent.generate_action(self.environment_name, agent.last_perception, return_prompt=True)
                action_prompts.append(action_prompt)
            actions = await self.ai_utils.run_parallel_ai_completion(action_prompts, update_history=False)
            self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())

            # Agents post messages via API
            tasks = []
            for agent, action in zip(cohort_agents, actions):
                content = self.extract_message_content(action)
                if content:
                    payload = {
                        "agent_id": agent.id,
                        "content": content,
                        "cohort_id": cohort_id,
                        "round_num": round_num,
                        "sub_round_num": sub_round_num
                    }
                    task = self.post_message(session, cohort_id, payload)
                    tasks.append(task)
                    log_group_message(self.logger, cohort_id, agent.index, content, sub_round_num)
                else:
                    self.logger.error(f"Failed to extract message content for agent {agent.id}")
            await asyncio.gather(*tasks)

    async def get_topic(self, session, cohort_id):
        try:
            async with session.get(f"{self.groupchat_api_url}/get_topic/{cohort_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    topic = data["topic"]
                    return topic
                else:
                    self.logger.error(f"Failed to get topic for cohort {cohort_id}: {resp.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Exception while getting topic for cohort {cohort_id}: {e}")
            return None

    async def get_messages(self, session, cohort_id):
        try:
            async with session.get(f"{self.groupchat_api_url}/get_messages/{cohort_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    messages = data["messages"]
                    return messages
                else:
                    self.logger.error(f"Failed to get messages for cohort {cohort_id}: {resp.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Exception while getting messages for cohort {cohort_id}: {e}")
            return []

    def sanitize_json_object(self, obj: Any) -> Any:
        """Recursively sanitize all strings in a JSON object"""
        if isinstance(obj, dict):
            return {k: self.sanitize_json_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.sanitize_json_object(item) for item in obj]
        elif isinstance(obj, str):
            return self.sanitize_text(obj)
        else:
            return obj

    def sanitize_text(self, text: str) -> str:
        """Sanitize text to be JSON-safe"""
        import re
        
        # Handle None or non-string input
        if not isinstance(text, str):
            return ""
        
        # Remove emojis and non-ASCII characters first
        text = re.sub(r'[^\x20-\x7E]+', '', text)
        
        # Replace problematic whitespace
        text = re.sub(r'[\n\t\r\f\v]+', ' ', text)
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Handle quotes and backslashes last
        text = text.replace('\\', '\\\\')  # escape backslashes first
        text = text.replace('"', '\\"')    # then escape quotes
        
        return text.strip()

    async def set_agent_system_message(self, cohort_id: str, topic: str, round_num: int, sub_round_num: int):
        """Set system messages for all agents in a cohort"""
        cohort_agents = self.cohorts[cohort_id]
        proposer_id = self.topic_proposers[cohort_id]
        
        # Sanitize topic first
        safe_topic = self.sanitize_text(topic)
        self.logger.debug(f"Original topic: {topic}")
        self.logger.debug(f"Sanitized topic: {safe_topic}")
        
        for agent in cohort_agents:
            try:
                # Create base system message
                if agent.id == proposer_id and sub_round_num == 1:
                    system_message = (
                        f"You are selected as the topic proposer for round {round_num}. "
                        f"Start the discussion about: {safe_topic}"
                    )
                else:
                    system_message = (
                        f"You are participating in round {round_num}, sub-round {sub_round_num}. "
                        f"Discuss the topic: {safe_topic}"
                    )
                
                # Sanitize the final message
                system_message = self.sanitize_text(system_message)
                
                # Validate JSON serialization
                test_json = json.dumps({"system": system_message})
                self.logger.debug(f"""
                ====== System Message Debug ======
                Agent ID: {agent.id}
                Agent Index: {agent.index}
                Is Proposer: {agent.id == proposer_id}
                System Message JSON: {test_json}
                ================================
                """)
                
                # Set the system message
                agent.system = system_message
                
            except Exception as e:
                self.logger.error(f"Error setting system message for agent {agent.index}: {str(e)}")
                self.logger.error(f"Problematic topic: {topic}")
                raise e

    def extract_message_content(self, action):
        try:
            if action.json_object:
                action_content = action.json_object.object
                if 'action' in action_content and 'content' in action_content['action']:
                    content = action_content['action']['content']
                else:
                    content = None
            else:
                content = action.str_content.strip() if action.str_content else None
            return content
        except Exception as e:
            self.logger.error(f"Error extracting message content: {e}")
            return None

    async def post_message(self, session, cohort_id, payload):
        try:
            async with session.post(f"{self.groupchat_api_url}/post_message", json=payload) as resp:
                if resp.status == 200:
                    self.logger.info(f"Message posted by agent {payload['agent_id']} in cohort {cohort_id}")
                else:
                    self.logger.error(f"Failed to post message for cohort {cohort_id}: {resp.status}")
        except Exception as e:
            self.logger.error(f"Exception while posting message for cohort {cohort_id}: {e}")

    async def run_reflection(self, round_num: int):
        # Agents reflect on their observations
        reflection_prompts = []
        agents_with_observations = []
        for agent in self.agents:
            if agent.last_observation:
                reflect_prompt = await agent.reflect(self.environment_name, return_prompt=True)
                reflection_prompts.append(reflect_prompt)
                agents_with_observations.append(agent)
        if reflection_prompts:
            reflections = await self.ai_utils.run_parallel_ai_completion(reflection_prompts, update_history=False)
            self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
            for agent, reflection in zip(agents_with_observations, reflections):
                if reflection.json_object:
                    log_reflection(self.logger, agent.index, reflection.json_object.object)
                    # Store reflection in agent memory
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
            self.logger.info(f"No reflections generated in this round.")

    async def process_environment_state(self, env_state: Any, cohort_agents: List[MarketAgent], cohort_id: str):
        # Since environment state is handled via API, we can process any necessary state here
        # For this implementation, we may not have local environment state to process
        pass  # No local state processing needed

    async def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        # Return a summary of the round
        summary = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'index': agent.index,
                'last_action': agent.last_action,
                'last_observation': agent.last_observation,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'cohorts': {cohort_id: [agent.id for agent in agents] for cohort_id, agents in self.cohorts.items()},
            'topics': self.topic_proposers,
        }
        return summary

    async def process_round_results(self, round_num: int):
        """Process and store results from the current round"""
        try:
            # You can collect data from the agents and store it
            self.data_inserter.insert_round_data(
                round_num=round_num,
                agents=self.agents,
                environments={},  # No local environments in this setup
                config=self.orchestrator_config,  # Use the full config here
                trackers={}  # No trackers since data is stored via API
            )
            self.logger.info(f"Data for round {round_num} inserted successfully.")
        except Exception as e:
            self.logger.error(f"Error processing round {round_num} results: {str(e)}")
            raise e

    def print_summary(self):
        log_section(self.logger, "GROUP CHAT SIMULATION SUMMARY")
        # Since the group chat data is handled via API, you can collect summaries if the API supports it
        # For now, you can summarize based on agent memories
        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index}")
            print(f"  Last action: {agent.last_action}")
            if agent.memory:
                print(f"  Last reflection: {agent.memory[-1]['content']}")
            print()
        # Optionally, print the round summaries
        for summary in self.round_summaries:
            print(f"Round {summary['round']} summary:")
            for agent_state in summary['agent_states']:
                print(f"  Agent {agent_state['index']} last action: {agent_state['last_action']}")
            print()
