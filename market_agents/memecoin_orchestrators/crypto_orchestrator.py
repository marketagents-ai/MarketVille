# crypto_orchestrator.py

import asyncio
from datetime import datetime
import json
import logging
from typing import List, Dict, Any

from market_agents.memecoin_orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import EnvironmentStep, MultiAgentEnvironment
from market_agents.environments.mechanisms.crypto import (
    CryptoMarketAction,
    CryptoMarketActionSpace,
    CryptoMarketGlobalObservation,
    CryptoMarketObservationSpace,
    CryptoMarketMechanism,
    GlobalCryptoMarketAction
)
from market_agents.memecoin_orchestrators.crypto_agent import CryptoEconomicAgent
from market_agents.memecoin_orchestrators.crypto_models import (
    MarketAction,
    Trade,
    OrderType
)
from market_agents.inference.message_models import LLMOutput
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.memecoin_orchestrators.config import CryptoConfig, OrchestratorConfig
from market_agents.memecoin_orchestrators.logger_utils import (
    log_section,
    log_environment_setup,
    log_running,
    log_action,
    log_round,
)
from market_agents.memecoin_orchestrators.insert_simulation_data import SimulationDataInserter
from market_agents.memecoin_orchestrators.agent_cognitive import AgentCognitiveProcessor

from agent_evm_interface.agent_evm_interface import EthereumInterface

# Define CryptoTracker for tracking crypto market-specific data
class CryptoTracker:
    def __init__(self):
        self.all_trades: List[Trade] = []
        self.price_history: List[float] = []
        self.agent_portfolios: Dict[str, float] = {}

    def add_trade(self, trade: Trade):
        self.all_trades.append(trade)

    def update_price_history(self, price: float):
        self.price_history.append(price)

    def update_agent_portfolio(self, agent_id: str, portfolio_value: float):
        self.agent_portfolios[agent_id] = portfolio_value

    def get_summary(self):
        return {
            "total_trades": len(self.all_trades),
            "price_history": self.price_history,
            "agent_portfolios": self.agent_portfolios
        }

# Implement the CryptoOrchestrator class
class CryptoOrchestrator(BaseEnvironmentOrchestrator):
    def __init__(
        self,
        config: CryptoConfig,
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
        self.environment_name = 'crypto_market'
        self.environment = None
        self.tracker = CryptoTracker()
        self.agent_rewards: Dict[str, float] = {}
        self.cognitive_processor = AgentCognitiveProcessor(ai_utils, data_inserter, logger, tool_mode=self.orchestrator_config.tool_mode)

        # Initialize EthereumInterface
        self.ethereum_interface = EthereumInterface()
        # Use the first account as the minter and funder
        self.minter_account = self.ethereum_interface.accounts[0]
        self.minter_private_key = self.minter_account['private_key']
        self.token_addresses = self.ethereum_interface.testnet_data['token_addresses']
        try:
            self.quote_token_address = self.ethereum_interface.get_token_address('USDC')
            self.doge_token_address = self.ethereum_interface.get_token_address('DOGE')
            self.orderbook_address = self.ethereum_interface.testnet_data['orderbook_address']
            
            self.logger.info(f"USDC address: {self.quote_token_address}")
            self.logger.info(f"DOGE address: {self.doge_token_address}")
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error getting token addresses: {str(e)}")
            raise

    async def setup_environment(self):
        """Sets up the crypto market environment and assigns it to agents"""
        log_section(self.logger, "CONFIGURING CRYPTO MARKET ENVIRONMENT")

        INITIAL_DOGE_AMOUNT = 1000 
        INITIAL_USDC_AMOUNT = 10000
        INITIAL_ETH_AMOUNT = 1000000000000000000
        for agent in self.agents:
            # Mint DOGE to agent's address
            tx_hash = self.ethereum_interface.mint_erc20(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_DOGE_AMOUNT,
                contract_address=self.doge_token_address,
                minter_private_key=self.minter_private_key
            )
            self.logger.info(f"Minted DOGE to agent {agent.id}. TxHash: {tx_hash}")

            # Mint USDC to agent's address
            tx_hash = self.ethereum_interface.mint_erc20(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_USDC_AMOUNT,
                contract_address=self.quote_token_address,
                minter_private_key=self.minter_private_key
            )
            self.logger.info(f"Minted USDC to agent {agent.id}. TxHash: {tx_hash}")

            # Send ETH to agent's address for gas fees
            tx_hash = self.ethereum_interface.send_eth(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_ETH_AMOUNT,
                private_key=self.minter_private_key
            )
            self.logger.info(f"Sent ETH to agent {agent.id}. TxHash: {tx_hash}")

        # Create the crypto market mechanism
        crypto_mechanism = CryptoMarketMechanism(
            max_rounds=self.config.max_rounds,
            coin=self.orchestrator_config.agent_config.asset_name,
            ethereum_interface=self.ethereum_interface
        )
        crypto_mechanism.setup()

        # Create agents dictionary for the environment
        agents_dict = {}
        for agent in self.agents:
            if isinstance(agent.economic_agent, CryptoEconomicAgent):
                agents_dict[agent.id] = agent.economic_agent
                crypto_mechanism.register_agent(agent.id, agent.economic_agent)

        # Set up the multi-agent environment
        self.environment = MultiAgentEnvironment(
            name=self.config.name,
            address=self.config.address,
            max_steps=self.config.max_rounds,
            action_space=CryptoMarketActionSpace(),
            observation_space=CryptoMarketObservationSpace(),
            mechanism=crypto_mechanism,
            agents=agents_dict
        )

        # Assign the environment to agents
        for agent in self.agents:
            if not hasattr(agent, 'environments') or agent.environments is None:
                agent.environments = {}
            agent.environments[self.environment_name] = self.environment

        log_environment_setup(self.logger, self.environment_name)
        self.logger.info("Crypto market environment setup complete.")

        # Verify environments are properly set
        for agent in self.agents:
            if self.environment_name not in agent.environments:
                self.logger.error(f"Agent {agent.index} missing crypto market environment!")
            else:
                self.logger.info(f"Agent {agent.index} environments: {list(agent.environments.keys())}")

        return self.environment

    async def run_environment(self, round_num: int):
        log_running(self.logger, self.environment_name)
        env = self.environment

        # Reset agents' pending orders at the beginning of the round
        for agent in self.agents:
            agent.economic_agent.reset_pending_orders()

        # Set system messages for agents
        self.set_agent_system_messages(round_num, env.mechanism.coin_name)

        log_section(self.logger, "AGENT PERCEPTIONS")
        # Run agents' perception in parallel
        await self.cognitive_processor.run_parallel_perceive(self.agents, self.environment_name)

        log_section(self.logger, "AGENT ACTIONS")
        # Run agents' action generation in parallel
        actions = await self.cognitive_processor.run_parallel_action(self.agents, self.environment_name)

        actions_map = {action.source_id: action for action in actions}

        # Collect actions from agents
        agent_actions = {}
        for agent in self.agents:
            action = actions_map.get(agent.id)
            if action:
                try:
                    action_content = action.json_object.object if action.json_object else json.loads(action.str_content or '{}')
                    agent.last_action = action_content
                    order_type = action_content.get('order_type')
                    if order_type in [OrderType.BUY.value, OrderType.SELL.value, OrderType.HOLD.value]:
                        if order_type != OrderType.HOLD.value:
                            price = action_content.get('price')
                            quantity = action_content.get('quantity')
                            if price is not None and quantity is not None:
                                market_action = MarketAction(
                                    order_type=OrderType(order_type),
                                    price=price,
                                    quantity=quantity
                                )
                            else:
                                raise ValueError(f"Price and quantity must be specified for 'buy' and 'sell' orders")
                        else:
                            market_action = MarketAction(order_type=OrderType.HOLD)
                        agent_actions[agent.id] = CryptoMarketAction(agent_id=agent.id, action=market_action)
                        # Update agent's pending orders
                        agent.economic_agent.pending_orders.append(market_action)
                        log_action(self.logger, agent.index, f"Order: {market_action}")
                    else:
                        raise ValueError(f"Invalid order_type: {order_type}")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self.logger.error(f"Error creating CryptoMarketAction for agent {agent.index}: {str(e)}")
            else:
                self.logger.warning(f"No action found for agent {agent.index}")

        # Create global action and step the environment
        global_action = GlobalCryptoMarketAction(actions=agent_actions)
        try:
            env_state = env.step(global_action)
        except Exception as e:
            self.logger.error(f"Error in environment {self.environment_name}: {str(e)}")
            raise e

        self.logger.info(f"Completed {self.environment_name} step")

        # Process the environment state
        if isinstance(env_state.global_observation, CryptoMarketGlobalObservation):
            self.process_environment_state(env_state)

        # Store the last environment state
        self.last_env_state = env_state

        # Run reflection step
        log_section(self.logger, "AGENT REFLECTIONS")
        await self.cognitive_processor.run_parallel_reflect(
            self.agents,
            self.environment_name
        )

    def set_agent_system_messages(self, round_num: int, coin: str):
        for agent in self.agents:
            # Get correct token address based on coin name
            token_address = self.ethereum_interface.get_token_address(coin)  # Use coin parameter instead of hardcoded DOGE
            
            # Get price for the correct token pair
            pair_info = self.ethereum_interface.get_pair_info(
                token_address,
                self.quote_token_address
            )
            
            # Convert price from wei (1e18) to normal units
            current_price = pair_info['token0_price_in_token1'] / 1e18
            
            # Get balances and convert from wei
            token_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                token_address
            ) / 1e18
            
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.quote_token_address
            ) / 1e18
            
            # Calculate portfolio value in USDC (using normal units now)
            portfolio_value = usdc_balance + (token_balance * current_price)
            
            agent.system = (
                f"Round {round_num}: You have {usdc_balance:.2f} USDC and {token_balance:.2f} units of {coin}. "
                f"Current market price: {current_price:.4f} USDC per {coin}. "
                f"Portfolio value: {portfolio_value:.2f} USDC. "
                f"Try to trade whenever there's opportunities for profit"
            )
    def process_environment_state(self, env_state: EnvironmentStep):
        global_observation = env_state.global_observation
        if not isinstance(global_observation, CryptoMarketGlobalObservation):
            self.logger.error(f"Unexpected global observation type: {type(global_observation)}")
            return

        # Get current market prices from EVM
        current_prices = {}
        for token_address in self.token_addresses:
            pair_info = self.ethereum_interface.get_pair_info(
                token_address,
                self.quote_token_address
            )
            current_prices[token_address] = pair_info['token0_price_in_token1']

        # Track trades and calculate rewards
        agent_rewards = {}
        self.logger.info(f"Processing {len(global_observation.all_trades)} trades")
        
        log_section(self.logger, "TRADES")
        for trade in global_observation.all_trades:
            try:
                buyer = next(agent for agent in self.agents if agent.id == trade.buyer_id)
                seller = next(agent for agent in self.agents if agent.id == trade.seller_id)
                
                # Get current market price for reward calculation
                current_price = current_prices[trade.token_address]
                
                # Process the trade for both agents
                buyer.economic_agent.process_trade(trade)
                seller.economic_agent.process_trade(trade)
                
                # Calculate rewards using current market price
                buyer_reward = buyer.economic_agent.calculate_trade_reward(trade, current_price)
                seller_reward = seller.economic_agent.calculate_trade_reward(trade, current_price)
                
                # Update agent balances from EVM
                self.update_agent_balances(buyer.economic_agent)
                self.update_agent_balances(seller.economic_agent)
                
                agent_rewards[buyer.id] = agent_rewards.get(buyer.id, 0) + buyer_reward
                agent_rewards[seller.id] = agent_rewards.get(seller.id, 0) + seller_reward
                
                self.tracker.add_trade(trade)
                
            except Exception as e:
                self.logger.error(f"Error processing trade: {str(e)}")
                self.logger.exception("Exception details:")

        # Store rewards and update environment state
        env_state.info['agent_rewards'] = agent_rewards
        self.agent_rewards = agent_rewards
        self.last_env_state = env_state

    def update_agent_balances(self, agent: CryptoEconomicAgent):
        """Update agent balances from EVM"""
        # Get ETH balance
        eth_balance = self.ethereum_interface.get_eth_balance(agent.ethereum_address)
        eth_balance_readable = eth_balance / 1e18
        # Get token balances
        token_balances = {}
        for token_address in self.token_addresses:
            balance = self.ethereum_interface.get_erc20_balance(
                agent.ethereum_address,
                token_address
            )
            token_balances[token_address] = balance
        
        # Update agent's portfolio
        agent.endowment.current_portfolio.cash = eth_balance_readable
        for token_address, balance in token_balances.items():
            agent.update_token_balance(self.token_symbols[token_address], balance)

    def get_round_summary(self, round_num: int) -> dict:
        # Return a summary of the round
        summary = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'ethereum_address': agent.economic_agent.ethereum_address,
                'portfolio_value': self.tracker.agent_portfolios.get(agent.id, 0.0),
                'last_action': agent.last_action,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'environment_state': self.environment.get_global_state(),
            'tracker_summary': self.tracker.get_summary()
        }
        return summary

    async def process_round_results(self, round_num: int):
        # Save round data to the database
        try:
            self.data_inserter.insert_round_data(
                round_num,
                self.agents,
                {self.environment_name: self.environment},
                self.orchestrator_config,
                {self.environment_name: self.tracker}
            )
            self.logger.info(f"Data for round {round_num} inserted successfully.")
        except Exception as e:
            self.logger.error(f"Error inserting data for round {round_num}: {str(e)}")

    async def run(self):
        await self.setup_environment()
        for round_num in range(1, self.config.max_rounds + 1):
            log_round(self.logger, round_num)
            await self.run_environment(round_num)
            await self.process_round_results(round_num)
        # Print simulation summary after all rounds
        self.print_summary()

    def print_summary(self):
        log_section(self.logger, "CRYPTO MARKET SIMULATION SUMMARY")

        # Calculate final portfolio values and rewards
        total_portfolio_value = 0.0
        WEI_TO_ETH = 1000000000000000000  # 1 ETH = 10^18 wei

        for agent in self.agents:
            current_price = self.environment.mechanism.current_price
            eth_balance = self.ethereum_interface.get_eth_balance(agent.economic_agent.ethereum_address)
            token_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.token_addresses[0]
            )
            eth_balance_readable = eth_balance / WEI_TO_ETH
            portfolio_value = eth_balance_readable + token_balance * current_price
            total_portfolio_value += portfolio_value
            self.logger.info(f"Agent {agent.index} final portfolio value: {portfolio_value:.4f} ETH")

        summary = self.tracker.get_summary()
        print(f"\nCrypto Market Environment:")
        print(f"Total number of trades: {summary['total_trades']}")
        print(f"Final price: {self.environment.mechanism.current_price:.4f} ETH per token")
        print(f"Price history: {summary['price_history']}")

        print("\nFinal Agent Portfolios:")
        for agent in self.agents:
            print(f"Agent {agent.index}:")
            current_price = self.environment.mechanism.current_price
            eth_balance = self.ethereum_interface.get_eth_balance(agent.economic_agent.ethereum_address)
            token_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.token_addresses[0]
            )
            eth_balance_readable = eth_balance / WEI_TO_ETH
            portfolio_value = eth_balance_readable + token_balance * current_price
            print(f"  ETH Balance: {eth_balance_readable:.4f} ETH")
            print(f"  {self.environment.mechanism.coin_name} Balance: {token_balance}")
            print(f"  Portfolio Value: {portfolio_value:.4f} ETH")
            if agent.memory:
                print(f"  Last Reflection: {agent.memory[-1]['content']}")
            print()