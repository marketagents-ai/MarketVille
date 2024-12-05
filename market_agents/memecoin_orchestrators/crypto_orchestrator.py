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

        # Get token decimals
        usdc_info = self.ethereum_interface.get_erc20_info(self.quote_token_address)
        doge_info = self.ethereum_interface.get_erc20_info(self.doge_token_address)
        usdc_decimals = usdc_info['decimals']
        doge_decimals = doge_info['decimals']

        # readable amounts
        READABLE_AMOUNTS = {
            'USDC': 10_000,
            'DOGE': 1_000,
            'ETH': 0.1
        }

        # Convert to raw amounts with proper decimals
        INITIAL_AMOUNTS = {
            'USDC': int(READABLE_AMOUNTS['USDC'] * (10 ** usdc_decimals)),
            'DOGE': int(READABLE_AMOUNTS['DOGE'] * (10 ** doge_decimals)),
            'ETH': int(READABLE_AMOUNTS['ETH'] * (10 ** 18))
        }

        for agent in self.agents:
            # Mint USDC
            tx_hash = self.ethereum_interface.mint_erc20(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_AMOUNTS['USDC'],
                contract_address=self.quote_token_address,
                minter_private_key=self.minter_private_key
            )
            self.logger.info(f"Minted {READABLE_AMOUNTS['USDC']} USDC to agent {agent.id}. TxHash: {tx_hash}")

            # Mint DOGE
            tx_hash = self.ethereum_interface.mint_erc20(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_AMOUNTS['DOGE'],
                contract_address=self.doge_token_address,
                minter_private_key=self.minter_private_key
            )
            self.logger.info(f"Minted {READABLE_AMOUNTS['DOGE']} DOGE to agent {agent.id}. TxHash: {tx_hash}")

            # Send ETH for gas
            tx_hash = self.ethereum_interface.send_eth(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_AMOUNTS['ETH'],
                private_key=self.minter_private_key
            )
            self.logger.info(f"Sent {READABLE_AMOUNTS['ETH']} ETH to agent {agent.id}. TxHash: {tx_hash}")

            # Log initial balances
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.quote_token_address
            ) / (10 ** usdc_decimals)
            
            doge_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.doge_token_address
            ) / (10 ** doge_decimals)
            
            eth_balance = self.ethereum_interface.get_eth_balance(
                agent.economic_agent.ethereum_address
            ) / (10 ** 18)
            
            self.logger.info(f"""
                Initial balances for agent {agent.id}:
                - USDC: {usdc_balance:.2f}
                - DOGE: {doge_balance:.2f}
                - ETH: {eth_balance:.4f}
            """)

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
            token_address = self.ethereum_interface.get_token_address(coin)
            
            # Get token decimals
            token_info = self.ethereum_interface.get_erc20_info(token_address)
            quote_token_info = self.ethereum_interface.get_erc20_info(self.quote_token_address)
            token_decimals = token_info['decimals']
            quote_decimals = quote_token_info['decimals']
            
            # Get price for the correct token pair
            pair_info = self.ethereum_interface.get_pair_info(
                token_address,
                self.quote_token_address
            )
            
            current_price = pair_info['token0_price_in_token1'] / 1e18

            token_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                token_address
            ) / (10 ** token_decimals)
            
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.quote_token_address
            ) / (10 ** quote_decimals)
            
            # Calculate portfolio value in USDC (using normal units now)
            portfolio_value = usdc_balance + (token_balance * current_price)
            
            agent.system = (
                f"Round {round_num}: You have {usdc_balance:.2f} USDC and {token_balance:.2f} units of {coin}. "
                f"Current market price: {current_price:.4f} USDC per {coin}. "
                f"Portfolio value: {portfolio_value:.2f} USDC. "
                f"Try to trade whenever there's opportunities for profit"
            )
            # Add logging of the system message
            self.logger.info(f"Agent {agent.index} system message:")
            self.logger.info(f"Raw price from pair_info: {pair_info['token0_price_in_token1']}")
            self.logger.info(f"Token decimals: {token_decimals}, Quote decimals: {quote_decimals}")
            self.logger.info(f"Converted price: {current_price}")
            self.logger.info(f"System message: {agent.system}")
            self.logger.info("---")

    def process_environment_state(self, env_state: EnvironmentStep):
        global_observation = env_state.global_observation
        if not isinstance(global_observation, CryptoMarketGlobalObservation):
            self.logger.error(f"Unexpected global observation type: {type(global_observation)}")
            return

        # Adjust if your main pair is different.
        main_symbol = "DOGE"
        main_token_address = self.doge_token_address

        try:
            # Attempt to get pair info for DOGE/USDC only
            pair_info = self.ethereum_interface.get_pair_info(
                main_token_address,
                self.quote_token_address
            )
            current_price = pair_info['token0_price_in_token1']
        except Exception as e:
            self.logger.warning(f"Could not get pair info for {main_symbol}-USDC due to insufficient liquidity: {e}")
            current_price = 0

        # Track trades and calculate rewards
        agent_rewards = {}
        all_trades = global_observation.all_trades
        self.logger.info(f"Processing {len(all_trades)} trades")


        for trade in all_trades:
            try:
                # This code remains the same, processing trades using the known current_price
                buyer = next(agent for agent in self.agents if agent.id == trade.buyer_id)
                seller = None
                if trade.seller_id != 'Orderbook':
                    seller = next(agent for agent in self.agents if agent.id == trade.seller_id)

                # Use current_price as fallback if pair_info is not available
                price_for_reward = current_price if current_price != 0 else 1e18  # fallback price if needed
                buyer.economic_agent.process_trade(trade)
                buyer_reward = buyer.economic_agent.calculate_trade_reward(trade, price_for_reward)
                agent_rewards[buyer.id] = agent_rewards.get(buyer.id, 0) + buyer_reward

                if seller:
                    seller.economic_agent.process_trade(trade)
                    seller_reward = seller.economic_agent.calculate_trade_reward(trade, price_for_reward)
                    agent_rewards[seller.id] = agent_rewards.get(seller.id, 0) + seller_reward

                self.update_agent_balances(buyer.economic_agent)
                if seller:
                    self.update_agent_balances(seller.economic_agent)

                self.tracker.add_trade(trade)
                self.logger.info(f"Processed trade: {trade.buyer_id} bought {trade.quantity} {trade.coin} "
                                f"from {trade.seller_id} at {trade.price} USDC")

            except Exception as ex:
                self.logger.error(f"Error processing trade {trade}: {str(ex)}")
                self.logger.exception("Exception details:")


    def update_agent_balances(self, agent: CryptoEconomicAgent):
        """Update agent balances from EVM"""
        # Get ETH balance
        eth_balance = self.ethereum_interface.get_eth_balance(agent.ethereum_address)
        eth_balance_readable = eth_balance / 1e18
        
        # Get token balances
        token_balances = {}
        for symbol, token_address in self.token_addresses.items():
            balance = self.ethereum_interface.get_erc20_balance(
                agent.ethereum_address,
                token_address
            )
            token_balances[symbol] = balance
        
        # Update agent's portfolio
        agent.endowment.current_portfolio.cash = eth_balance_readable
        for symbol, balance in token_balances.items():
            agent.update_token_balance(symbol, balance)

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
                self.token_addresses["USDC"]
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
                self.token_addresses["USDC"]
            )
            eth_balance_readable = eth_balance / WEI_TO_ETH
            portfolio_value = eth_balance_readable + token_balance * current_price
            print(f"  ETH Balance: {eth_balance_readable:.4f} ETH")
            print(f"  {self.environment.mechanism.coin_name} Balance: {token_balance}")
            print(f"  Portfolio Value: {portfolio_value:.4f} ETH")
            if agent.memory:
                print(f"  Last Reflection: {agent.memory[-1]['content']}")
            print()