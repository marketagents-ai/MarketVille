# meta_orchestrator.py

import asyncio
import logging
import os
import random
import uuid
from pathlib import Path
from typing import List, Dict, Union
import warnings

import yaml
from market_agents.memecoin_orchestrators.crypto_orchestrator import CryptoOrchestrator
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona, generate_persona, save_persona_to_file
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.memecoin_orchestrators.crypto_agent import CryptoEconomicAgent
from market_agents.memecoin_orchestrators.crypto_models import Crypto, Endowment as CryptoEndowment, Portfolio, Position
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.memecoin_orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.memecoin_orchestrators.config import OrchestratorConfig, load_config
from market_agents.memecoin_orchestrators.insert_simulation_data import SimulationDataInserter
from market_agents.memecoin_orchestrators.logger_utils import (
    log_section,
    log_environment_setup,
    log_agent_init,
    log_round,
    print_ascii_art,
    orchestration_logger,
    log_completion
)

warnings.filterwarnings("ignore", module="pydantic")

class MetaOrchestrator:
    def __init__(self, config: OrchestratorConfig, environment_order: List[str] = None):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.ai_utils = self._initialize_ai_utils()
        self.data_inserter = self._initialize_data_inserter()
        self.logger = orchestration_logger
        self.environment_order = environment_order or config.environment_order
        self.environment_orchestrators = {}

    def _initialize_ai_utils(self):
        # Initialize AI utilities
        oai_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )
        return ai_utils

    def _initialize_data_inserter(self):
        db_config = self.config.database_config
        db_params = {
            'dbname': db_config.db_name,
            'user': db_config.db_user,
            'password': db_config.db_password,
            'host': db_config.db_host,
            'port': db_config.db_port
        }
        data_inserter = SimulationDataInserter(db_params)
        return data_inserter

    def load_or_generate_personas(self) -> List[Persona]:
        personas_dir = Path("./market_agents/agents/personas/generated_personas")
        existing_personas = []

        if os.path.exists(personas_dir):
            for filename in os.listdir(personas_dir):
                if filename.endswith(".yaml"):
                    with open(os.path.join(personas_dir, filename), 'r') as file:
                        persona_data = yaml.safe_load(file)
                        existing_personas.append(Persona(**persona_data))

        while len(existing_personas) < self.config.num_agents:
            new_persona = generate_persona()
            existing_personas.append(new_persona)
            save_persona_to_file(new_persona, personas_dir)

        return existing_personas[:self.config.num_agents]

    def generate_agents(self):
        log_section(self.logger, "Generating Agents")
        personas = self.load_or_generate_personas()
        num_agents = len(personas)

        # Initialize EthereumInterface here for all agents
        from agent_evm_interface.agent_evm_interface import EthereumInterface
        ethereum_interface = EthereumInterface()
        
        for i, persona in enumerate(personas):
            agent_uuid = str(uuid.uuid4())
            llm_config = random.choice(self.config.llm_configs).dict() if len(self.config.llm_configs) > 1 else self.config.llm_configs[0].dict()

            persona.role = "trader"
            agent_config = self.config.agent_config.dict()

            # Get Ethereum account for this agent
            account = ethereum_interface.accounts.pop()

            # Create initial portfolio
            initial_portfolio = Portfolio(
                cash=agent_config.get('initial_cash', 1000),
                coins=[
                    Crypto(
                        symbol=agent_config.get('coin_name', 'TOKEN'),
                        positions=[Position(
                            quantity=agent_config.get('initial_coin', 100),
                            purchase_price=1.0
                        )]
                    )
                ]
            )

            # Create the endowment with portfolio
            endowment = CryptoEndowment(
                initial_portfolio=initial_portfolio,
                agent_id=agent_uuid
            )

            # Create the CryptoEconomicAgent with Ethereum details
            economic_agent = CryptoEconomicAgent(
                id=agent_uuid,
                endowment=endowment,
                max_relative_spread=agent_config.get('max_relative_spread', 0.2),
                coin=agent_config.get('coin_name', 'TOKEN'),
                ethereum_address=account['address'],  # Add Ethereum address here
                private_key=account['private_key']    # Add private key here
            )

            # Create MarketAgent with the same Ethereum details
            agent = MarketAgent.create(
                agent_id=agent_uuid,
                use_llm=agent_config.get('use_llm', True),
                llm_config=llm_config,
                environments={},
                protocol=ACLMessage,
                persona=persona,
                econ_agent=economic_agent
            )

            agent.last_perception = None
            agent.last_observation = None
            agent.last_step = None
            agent.index = i
            self.agents.append(agent)
            log_agent_init(self.logger, agent.index, False, persona)

    def _initialize_environment_orchestrators(self) -> Dict[str, BaseEnvironmentOrchestrator]:
        orchestrators = {}
        
        # Generate agents first if not already done
        if not self.agents:
            self.generate_agents()
        
        # Initialize orchestrators based on environment order
        for env_name in self.environment_order:
            env_config = self.config.environment_configs.get(env_name)
            if not env_config:
                self.logger.warning(f"Configuration for environment '{env_name}' not found.")
                continue
                
            if env_name == 'crypto_market':
                orchestrator = CryptoOrchestrator(
                    config=env_config,
                    orchestrator_config=self.config,
                    agents=self.agents,
                    ai_utils=self.ai_utils,
                    data_inserter=self.data_inserter,
                    logger=self.logger
                )
            else:
                self.logger.warning(f"Unknown environment: {env_name}")
                continue
            
            orchestrators[env_name] = orchestrator
            self.logger.info(f"Initialized {env_name} environment")
            
        return orchestrators

    async def run_simulation(self):
        # Initialize environment orchestrators
        self.environment_orchestrators = self._initialize_environment_orchestrators()
        
        # Set up each environment before starting simulation
        for env_name, orchestrator in self.environment_orchestrators.items():
            self.logger.info(f"Setting up {env_name} environment...")
            await orchestrator.setup_environment()  # Properly await setup
            self.logger.info(f"Setup complete for {env_name} environment")
        
        # Run simulation rounds - each round includes environments in sequence
        for round_num in range(1, self.config.max_rounds + 1):
            log_round(self.logger, round_num)
            
            # Run each environment in sequence within the same round
            for env_name in self.environment_order:
                orchestrator = self.environment_orchestrators.get(env_name)
                if orchestrator is None:
                    self.logger.warning(f"No orchestrator found for environment '{env_name}'. Skipping.")
                    continue
                    
                log_environment_setup(self.logger, env_name)
                try:
                    # Run the environment for this round
                    await orchestrator.run_environment(round_num)
                    # Process results but maintain environment assignments
                    await orchestrator.process_round_results(round_num)
                    
                    self.logger.info(f"Completed {env_name} environment for round {round_num}")
                except Exception as e:
                    self.logger.error(f"Error running {env_name} environment: {str(e)}")
                    raise e

        # Print summaries for each environment
        for orchestrator in self.environment_orchestrators.values():
            orchestrator.print_summary()

    async def start(self):
        print_ascii_art()
        log_section(self.logger, "Simulation Starting")
        await self.run_simulation()
        log_completion(self.logger, "Simulation completed successfully")

if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    async def main():
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Run the market simulation.')
        parser.add_argument('--environments', nargs='+', help='List of environments to run (e.g., crypto_market)')
        args = parser.parse_args()

        # Load configuration from orchestrator_config.yaml
        config_path = Path("market_agents/memecoin_orchestrators/orchestrator_config.yaml")
        config = load_config(config_path=config_path)

        # If environments are specified in command-line arguments, use them
        if args.environments:
            environment_order = args.environments
        else:
            environment_order = config.environment_order

        # Initialize MetaOrchestrator with the loaded config and specified environments
        orchestrator = MetaOrchestrator(config, environment_order=environment_order)
        # Start the simulation
        await orchestrator.start()

    asyncio.run(main())
