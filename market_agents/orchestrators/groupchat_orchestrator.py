# groupchat_orchestrator.py

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatActionSpace, GroupChatObservationSpace
from market_agents.orchestrators.config import GroupChatConfig, OrchestratorConfig
from market_agents.orchestrators.logger_utils import (
    log_perception,
    log_persona,
    log_section,
    log_round,
    log_cohort_formation,
    log_topic_proposal,
    log_sub_round_start,
    log_group_message,
    log_reflection,
)
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter

from market_agents.orchestrators.group_chat.groupchat_api_utils import GroupChatAPIUtils
from market_agents.orchestrators.agent_cognitive import AgentCognitiveProcessor


class GroupChatOrchestrator:
    """
    Orchestrator for the Group Chat simulation.

    Manages the overall flow, including setting up the environment,
    running rounds and sub-rounds, and coordinating agent actions.
    """

    def __init__(
        self,
        config: GroupChatConfig,
        orchestrator_config: OrchestratorConfig,
        agents: List[MarketAgent],
        ai_utils,
        data_inserter: SimulationDataInserter,
        logger=None
    ):
        self.config = config
        self.orchestrator_config = orchestrator_config
        self.agents = agents
        self.ai_utils = ai_utils
        self.data_inserter = data_inserter
        self.logger = logger or logging.getLogger(__name__)

        # Initialize API utils
        self.api_utils = GroupChatAPIUtils(self.config.groupchat_api_url, self.logger)

        # Initialize cognitive processor
        self.cognitive_processor = AgentCognitiveProcessor(ai_utils, data_inserter, self.logger)

        # Agent dictionary for quick lookup
        self.agent_dict = {agent.id: agent for agent in agents}

        # Cohorts: cohort_id -> List[MarketAgent]
        self.cohorts: Dict[str, List[MarketAgent]] = {}

        # Topic proposers: cohort_id -> proposer_id
        self.topic_proposers: Dict[str, str] = {}

        # Round summaries
        self.round_summaries: List[Dict[str, Any]] = []

        # Sub-rounds per round
        self.sub_rounds_per_round = config.sub_rounds

    async def setup_environment(self):
        """
        Sets up the environment by checking API health, registering agents,
        forming cohorts, and assigning agents to cohorts.
        """
        log_section(self.logger, "CONFIGURING GROUP CHAT ENVIRONMENT")

        # Check API health
        if not await self.api_utils.check_api_health():
            raise RuntimeError("GroupChat API is not available")

        # Register agents
        await self.api_utils.register_agents(self.agents)

        # Form cohorts
        agent_ids = [agent.id for agent in self.agents]
        cohorts_info = await self.api_utils.form_cohorts(agent_ids, self.config.group_size)

        # Create environments and assign cohorts
        for cohort in cohorts_info:
            cohort_id = cohort["cohort_id"]
            cohort_agent_ids = cohort["agent_ids"]
            cohort_agents = [self.agent_dict[agent_id] for agent_id in cohort_agent_ids]
            self.cohorts[cohort_id] = cohort_agents

            # Create environment for this cohort
            group_chat = GroupChat(
                max_rounds=self.config.max_rounds,
                sequential=False,
            )
            group_chat_env = MultiAgentEnvironment(
                name=f"group_chat_{cohort_id}",
                address=f"group_chat_{cohort_id}",
                max_steps=self.config.max_rounds,
                action_space=GroupChatActionSpace(),
                observation_space=GroupChatObservationSpace(),
                mechanism=group_chat
            )

            # Assign environment and cohort_id to agents
            for agent in cohort_agents:
                agent.cohort_id = cohort_id
                if not hasattr(agent, 'environments') or agent.environments is None:
                    agent.environments = {}
                agent.environments['group_chat'] = group_chat_env

            log_cohort_formation(self.logger, cohort_id, [agent.index for agent in cohort_agents])

        self.logger.info("Environment setup complete.")
        
        # Verify environments are properly set
        for agent in self.agents:
            if 'group_chat' not in agent.environments:
                self.logger.error(f"Agent {agent.index} missing group_chat environment!")
            else:
                self.logger.info(f"Agent {agent.index} environments: {list(agent.environments.keys())}")

    async def run_environment(self, round_num: int = None):
        """
        Runs the environment for the configured number of rounds.
        
        Args:
            round_num (int, optional): If provided, runs a specific round.
                                        If None, runs all rounds.
        """
        if round_num is not None:
            # Run specific round
            await self.run_round(round_num)
        else:
            # Run all rounds
            for round_num in range(1, self.config.max_rounds + 1):
                await self.run_round(round_num)

    async def run_round(self, round_num: int):
        """
        Runs a single round of the simulation.

        Args:
            round_num (int): The current round number.
        """
        log_round(self.logger, round_num)

        # Select topic proposers
        await self.select_topic_proposers()

        # Collect proposed topics
        await self.collect_proposed_topics(round_num)

        # Run sub-rounds
        for sub_round in range(1, self.sub_rounds_per_round + 1):
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

        # Run reflection
        log_section(self.logger, "AGENT REFLECTIONS")
        await self.cognitive_processor.run_parallel_reflect(self.agents, self.config.name)

        # Store round summary
        round_summary = await self.get_round_summary(round_num)
        self.round_summaries.append(round_summary)

    async def select_topic_proposers(self):
        """
        Selects topic proposers for each cohort using the API.
        """
        tasks = []
        for cohort_id, cohort_agents in self.cohorts.items():
            agent_ids = [agent.id for agent in cohort_agents]
            task = asyncio.create_task(
                self.api_utils.select_proposer(cohort_id, agent_ids)
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        for cohort_id, proposer_id in zip(self.cohorts.keys(), results):
            if proposer_id:
                self.topic_proposers[cohort_id] = proposer_id
                self.logger.info(f"Selected proposer {proposer_id} for cohort {cohort_id}")
            else:
                self.logger.error(f"Failed to select proposer for cohort {cohort_id}")

    async def collect_proposed_topics(self, round_num: int):
        """
        Collects proposed topics from proposers and submits them via the API.

        Args:
            round_num (int): The current round number.
        """
        proposer_agents = []
        proposer_prompts = []

        # Collect prompts for proposers
        for cohort_id, proposer_id in self.topic_proposers.items():
            proposer_agent = self.agent_dict[proposer_id]
            # Set system message for proposer
            good_name = self.orchestrator_config.agent_config.good_name
            proposer_agent.system = f"You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for group discussion about {good_name}."
            prompt = await proposer_agent.generate_action(
                self.config.name,
                f"Consider recent events, trends, or news related to {good_name}. Propose a specific topic for discussion that would be relevant to market participants.",
                return_prompt=True
            )
            proposer_agents.append((cohort_id, proposer_agent))
            proposer_prompts.append(prompt)

        # Run prompts in parallel
        proposals = await self.ai_utils.run_parallel_ai_completion(proposer_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())

        # Submit topics via API
        tasks = []
        for (cohort_id, proposer_agent), proposal in zip(proposer_agents, proposals):
            topic = self.extract_topic_from_proposal(proposal)
            if topic:
                task = asyncio.create_task(
                    self.api_utils.propose_topic(
                        agent_id=proposer_agent.id,
                        cohort_id=cohort_id,
                        topic=topic,
                        round_num=round_num
                    )
                )
                tasks.append(task)
                log_topic_proposal(self.logger, cohort_id, proposer_agent.index, topic)
            else:
                self.logger.error(f"Failed to extract topic from proposer {proposer_agent.id} in cohort {cohort_id}")
        await asyncio.gather(*tasks)

    def extract_topic_from_proposal(self, proposal) -> Optional[str]:
        """
        Extracts the topic from the proposal.

        Args:
            proposal: The proposal response.

        Returns:
            Optional[str]: The extracted topic.
        """
        try:
            if proposal.json_object:
                action_content = proposal.json_object.object
                if 'content' in action_content:
                    topic = action_content['content']
                elif 'action' in action_content and 'content' in action_content['action']:
                    topic = action_content['action']['content']
                else:
                    topic = None
            else:
                topic = proposal.str_content.strip() if proposal.str_content else None
            return topic
        except Exception as e:
            self.logger.error(f"Error extracting topic: {e}")
            return None

    async def run_group_chat_sub_round(
        self,
        cohort_id: str,
        round_num: int,
        sub_round_num: int,
        cohort_agents: List[MarketAgent]
    ):
        """
        Runs a single sub-round for a cohort.

        Args:
            cohort_id (str): The cohort ID.
            round_num (int): The current round number.
            sub_round_num (int): The current sub-round number.
            cohort_agents (List[MarketAgent]): The agents in the cohort.
        """
        try:
            # Get topic and messages from API
            topic = await self.api_utils.get_topic(cohort_id)
            messages = await self.api_utils.get_messages(cohort_id)

            if not topic:
                self.logger.error(f"No topic found for cohort {cohort_id}")
                return

            # Update agents' last observation
            for agent in cohort_agents:
                agent.last_observation = {
                    'messages': messages,
                    'current_topic': topic
                }

            # Agents perceive the messages
            perceptions = await self.cognitive_processor.run_parallel_perceive(cohort_agents, self.config.name)
            # Log personas and perceptions
            for agent, perception in zip(cohort_agents, perceptions):
                log_persona(self.logger, agent.index, agent.persona)
                log_perception(
                    self.logger, 
                    agent.index, 
                    perception.json_object.object if perception and perception.json_object else None
                )
                agent.last_perception = perception.json_object.object if perception.json_object else perception.str_content


            # Agents generate actions (messages)
            actions = await self.cognitive_processor.run_parallel_action(cohort_agents, self.config.name)

            # Post messages via API
            tasks = []
            for agent, action in zip(cohort_agents, actions):
                content = self.extract_message_content(action)
                if content:
                    task = asyncio.create_task(
                        self.api_utils.post_message(
                            agent_id=agent.id,
                            cohort_id=cohort_id,
                            content=content,
                            round_num=round_num,
                            sub_round_num=sub_round_num
                        )
                    )
                    tasks.append(task)
                    agent.last_action = content
                    log_group_message(self.logger, cohort_id, agent.index, content, sub_round_num)
                else:
                    self.logger.error(f"Failed to extract message content for agent {agent.id}")
            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"Error during sub-round {sub_round_num} for cohort {cohort_id}: {e}")

    def extract_message_content(self, action) -> Optional[str]:
        """
        Extracts the message content from the action.

        Args:
            action: The action response.

        Returns:
            Optional[str]: The extracted message content.
        """
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

    async def process_round_results(self, round_num: int):
        """Process and store results from the current round"""
        try:
            # Store round data
            self.data_inserter.insert_round_data(
                round_num=round_num,
                agents=self.agents,
                environments={},
                config=self.orchestrator_config,
                trackers={}
            )
            self.logger.info(f"Data for round {round_num} inserted successfully.")

            # Store round summary
            round_summary = await self.get_round_summary(round_num)
            self.round_summaries.append(round_summary)

        except Exception as e:
            self.logger.error(f"Error processing round {round_num} results: {str(e)}")
            raise e

    async def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        """Return a summary of the round"""
        summary = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'index': agent.index,
                'last_action': agent.last_action,
                'last_observation': agent.last_observation,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'cohorts': {cohort_id: [agent.id for agent in agents] 
                    for cohort_id, agents in self.cohorts.items()},
            'topics': self.topic_proposers,
        }
        return summary

    def print_summary(self):
        """Print a summary of the simulation results"""
        log_section(self.logger, "GROUP CHAT SIMULATION SUMMARY")
        
        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index}")
            print(f"  Last action: {agent.last_action}")
            if agent.memory:
                print(f"  Last reflection: {agent.memory[-1]['content']}")
            print()
        
        # Print round summaries
        for summary in self.round_summaries:
            print(f"Round {summary['round']} summary:")
            for agent_state in summary['agent_states']:
                print(f"  Agent {agent_state['index']} last action: {agent_state['last_action']}")
            print()

