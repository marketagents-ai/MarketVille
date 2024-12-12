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
        self.trades: List[Trade] = []
        self.agent_portfolios: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        
    def add_trade(self, trade: Trade, round_num: int):
        """Add a trade to the tracker"""
        self.trades.append(trade)
        
        # Update volume history
        if trade.coin not in self.volume_history:
            self.volume_history[trade.coin] = []
        self.volume_history[trade.coin].append(trade.quantity * trade.price)

    def update_price_history(self, price: float, token: str):
        """Update price history for a specific token"""
        if token not in self.price_history:
            self.price_history[token] = []
        self.price_history[token].append(price)

    def get_summary(self) -> dict:
        """Get summary statistics"""
        summary = {
            'total_trades': len(self.trades),
            'total_volume': sum(
                trade.quantity * trade.price 
                for trade in self.trades
            ) if self.trades else 0,
            'price_history': self.price_history,
            'volume_history': self.volume_history
        }
        return summary

    def get_trades_data(self) -> List[Dict]:
        """Get trades data in format suitable for database insertion"""
        return [
            {
                'buyer_id': str(trade.buyer_id),
                'seller_id': str(trade.seller_id),
                'quantity': float(trade.quantity),
                'price': float(trade.price),
                'coin': trade.coin,
                'round': getattr(trade, 'round', 0),  # Default to 0 if round not set
                'timestamp': getattr(trade, 'timestamp', datetime.now()),
                'tx_hash': getattr(trade, 'tx_hash', None)
            }
            for trade in self.trades
        ]

    def add_round_data(self, volume: float, prices: Dict[str, float]):
        """Add data for a single round"""
        for token, price in prices.items():
            self.update_price_history(price, token)
            if token not in self.volume_history:
                self.volume_history[token] = []
            self.volume_history[token].append(volume)
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
            # Get quote token (USDC) address
            self.quote_token_address = self.ethereum_interface.get_token_address('USDC')
            self.logger.info(f"USDC address: {self.quote_token_address}")
            
            # Get addresses for all configured trading tokens
            self.trading_token_addresses = {}
            for token in self.orchestrator_config.agent_config.tokens:
                token_address = self.ethereum_interface.get_token_address(token)
                self.trading_token_addresses[token] = token_address
                self.logger.info(f"{token} address: {token_address}")
            
            self.orderbook_address = self.ethereum_interface.testnet_data['orderbook_address']
        except (ValueError, KeyError) as e:
            self.logger.error(f"Error getting token addresses: {str(e)}")
            raise

    async def setup_environment(self):
        """Sets up the crypto market environment and assigns it to agents"""
        log_section(self.logger, "CONFIGURING CRYPTO MARKET ENVIRONMENT")

        # Get token info for all supported tokens
        token_info = {}
        token_decimals = {}
        token_addresses = {}

        # Get USDC info first as it's the quote token
        usdc_info = self.ethereum_interface.get_erc20_info(self.quote_token_address)
        token_info['USDC'] = usdc_info
        token_decimals['USDC'] = usdc_info['decimals']
        token_addresses['USDC'] = self.quote_token_address

        # Get info for all trading tokens from config
        supported_tokens = self.orchestrator_config.agent_config.tokens
        for token in supported_tokens:
            token_address = self.ethereum_interface.get_token_address(token)
            token_addresses[token] = token_address
            token_info[token] = self.ethereum_interface.get_erc20_info(token_address)
            token_decimals[token] = token_info[token]['decimals']

        # Define readable amounts for all tokens
        READABLE_AMOUNTS = {
            'USDC': 10_000,  # Initial USDC allocation
            'ETH': 0.1,      # Initial ETH for gas
        }

        # Add initial amounts for trading tokens from config
        for allocation in self.orchestrator_config.agent_config.initial_allocations:
            READABLE_AMOUNTS[allocation['token']] = allocation['quantity']

        # Convert to raw amounts with proper decimals
        INITIAL_AMOUNTS = {
            'USDC': int(READABLE_AMOUNTS['USDC'] * (10 ** token_decimals['USDC'])),
            'ETH': int(READABLE_AMOUNTS['ETH'] * (10 ** 18))
        }
        
        # Add raw amounts for trading tokens
        for token in supported_tokens:
            if token in READABLE_AMOUNTS:
                INITIAL_AMOUNTS[token] = int(READABLE_AMOUNTS[token] * (10 ** token_decimals[token]))

        for agent in self.agents:
            # Mint USDC (quote token)
            tx_hash = self.ethereum_interface.mint_erc20(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_AMOUNTS['USDC'],
                contract_address=self.quote_token_address,
                minter_private_key=self.minter_private_key
            )
            self.logger.info(f"Minted {READABLE_AMOUNTS['USDC']} USDC to agent {agent.id}. TxHash: {tx_hash}")

            # Mint trading tokens
            for token in supported_tokens:
                tx_hash = self.ethereum_interface.mint_erc20(
                    to=agent.economic_agent.ethereum_address,
                    amount=INITIAL_AMOUNTS[token],
                    contract_address=token_addresses[token],
                    minter_private_key=self.minter_private_key
                )
                self.logger.info(f"Minted {READABLE_AMOUNTS[token]} {token} to agent {agent.id}. TxHash: {tx_hash}")

            # Send ETH for gas
            tx_hash = self.ethereum_interface.send_eth(
                to=agent.economic_agent.ethereum_address,
                amount=INITIAL_AMOUNTS['ETH'],
                private_key=self.minter_private_key
            )
            self.logger.info(f"Sent {READABLE_AMOUNTS['ETH']} ETH to agent {agent.id}. TxHash: {tx_hash}")

            # Log initial balances
            balance_info = []
            
            # USDC balance
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.quote_token_address
            ) / (10 ** token_decimals['USDC'])
            balance_info.append(f"- USDC: {usdc_balance:.2f}")

            # Trading token balances
            for token in supported_tokens:
                token_balance = self.ethereum_interface.get_erc20_balance(
                    agent.economic_agent.ethereum_address,
                    token_addresses[token]
                ) / (10 ** token_decimals[token])
                balance_info.append(f"- {token}: {token_balance:.2f}")

            # ETH balance
            eth_balance = self.ethereum_interface.get_eth_balance(
                agent.economic_agent.ethereum_address
            ) / (10 ** 18)
            balance_info.append(f"- ETH: {eth_balance:.4f}")

            # Log all balances
            self.logger.info(f"""
                Initial balances for agent {agent.id}:
                {chr(10).join(balance_info)}
            """)

            # Initialize portfolio positions for all tokens
            for token in supported_tokens:
                agent.economic_agent.endowment.initial_portfolio.update_crypto(
                    symbol=token,
                    quantity=READABLE_AMOUNTS[token],
                    purchase_price=1.0  # Initial reference price
                )

        # Create the crypto market mechanism with multiple tokens
        crypto_mechanism = CryptoMarketMechanism(
            max_rounds=self.config.max_rounds,
            tokens=supported_tokens,  # Pass all supported tokens from config
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

        # Set system messages for agents with multi-token support
        self.set_agent_system_messages(round_num)

        log_section(self.logger, "AGENT PERCEPTIONS")
        await self.cognitive_processor.run_parallel_perceive(self.agents, self.environment_name)

        log_section(self.logger, "AGENT ACTIONS")
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
                            token = action_content.get('token')
                            if token not in env.mechanism.tokens:
                                raise ValueError(f"Invalid token: {token}")
                            price = action_content.get('price')
                            quantity = action_content.get('quantity')
                            if price is not None and quantity is not None:
                                market_action = MarketAction(
                                    order_type=OrderType(order_type),
                                    token=token,
                                    price=price,
                                    quantity=quantity
                                )
                            else:
                                raise ValueError("Price and quantity must be specified for 'buy' and 'sell' orders")
                        else:
                            market_action = MarketAction(order_type=OrderType.HOLD)
                        agent_actions[agent.id] = CryptoMarketAction(agent_id=agent.id, action=market_action)
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
            
            if isinstance(env_state.global_observation, CryptoMarketGlobalObservation):
                self.process_environment_state(env_state)
                self.last_env_state = env_state
            else:
                self.logger.error(f"Unexpected global observation type: {type(env_state.global_observation)}")

        except Exception as e:
            self.logger.error(f"Error in environment {self.environment_name}: {str(e)}")
            raise e

        self.logger.info(f"Completed {self.environment_name} step")

        log_section(self.logger, "AGENT REFLECTIONS")
        await self.cognitive_processor.run_parallel_reflect(
            self.agents,
            self.environment_name
        )

    def set_agent_system_messages(self, round_num: int):
        """Set system messages for agents with support for multiple tokens"""
        for agent in self.agents:
            # Get balances and prices for all supported tokens
            token_info = []
            portfolio_value = 0.0
            
            # Get USDC info first
            usdc_address = self.ethereum_interface.get_token_address('USDC')
            usdc_info = self.ethereum_interface.get_erc20_info(usdc_address)
            usdc_decimals = usdc_info['decimals']
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                usdc_address
            ) / (10 ** usdc_decimals)
            portfolio_value += usdc_balance
            token_info.append(f"USDC Balance: {usdc_balance:.2f}")

            # Get info for all trading tokens
            for token in self.environment.mechanism.tokens:
                try:
                    token_address = self.ethereum_interface.get_token_address(token)
                    token_info_data = self.ethereum_interface.get_erc20_info(token_address)
                    token_decimals = token_info_data['decimals']
                    
                    # Get token balance
                    token_balance = self.ethereum_interface.get_erc20_balance(
                        agent.economic_agent.ethereum_address,
                        token_address
                    ) / (10 ** token_decimals)
                    
                    # Get current price
                    pair_info = self.ethereum_interface.get_pair_info(
                        token_address,
                        usdc_address
                    )
                    current_price = pair_info['token0_price_in_token1'] / 1e18
                    
                    # Calculate token value in USDC
                    token_value = token_balance * current_price
                    portfolio_value += token_value
                    
                    token_info.append(
                        f"{token} Balance: {token_balance:.4f} (Price: {current_price:.4f} USDC, Value: {token_value:.2f} USDC)"
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Error getting info for {token}: {str(e)}")
                    continue

            # Create system message with all token information
            token_status = "\n".join(token_info)
            agent.system = (
                f"Round {round_num}:\n"
                f"{token_status}\n"
                f"Total Portfolio Value: {portfolio_value:.2f} USDC\n"
                f"Available Trading Pairs: {', '.join([f'{token}/USDC' for token in self.environment.mechanism.tokens])}\n"
                f"Try to trade whenever there's an opportunity for profit. "
                f"Don't be afraid to make trades when conditions are favorable."
            )
            
            # Log the system message
            self.logger.info(f"Agent {agent.index} system message:")
            self.logger.info(agent.system)
            self.logger.info("---")

    def process_environment_state(self, env_state: EnvironmentStep):
        global_observation = env_state.global_observation
        if not isinstance(global_observation, CryptoMarketGlobalObservation):
            self.logger.error(f"Unexpected global observation type: {type(global_observation)}")
            return

        # Update agent states with their observations
        for agent_id, agent_observation in global_observation.observations.items():
            try:
                agent = next((agent for agent in self.agents if agent.id == agent_id), None)
                if agent:
                    agent.last_observation = agent_observation
                    agent.last_step = env_state
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id} state: {str(e)}")

        # Initialize tracker if not exists
        if not hasattr(self, 'tracker'):
            self.tracker = CryptoTracker()

        # Process trades from global observation
        self.logger.info(f"Processing {len(global_observation.all_trades)} trades")
        log_section(self.logger, "TRADES")

        for trade in global_observation.all_trades:
            try:
                # Skip processing if it's a HOLD action
                if trade.action_type == OrderType.HOLD:
                    self.logger.info(f"Agent {trade.buyer_id} holds {trade.coin}")
                    continue

                # Process buyer side
                if trade.buyer_id != "MARKET":
                    try:
                        buyer = next(agent for agent in self.agents if agent.id == trade.buyer_id)
                        buyer.economic_agent.process_trade(trade)
                        self.update_agent_balances(buyer.economic_agent)
                    except StopIteration:
                        self.logger.warning(f"Buyer {trade.buyer_id} not found in agents")

                # Process seller side
                if trade.seller_id not in ["MARKET", "Orderbook"]:
                    try:
                        seller = next(agent for agent in self.agents if agent.id == trade.seller_id)
                        seller.economic_agent.process_trade(trade)
                        self.update_agent_balances(seller.economic_agent)
                    except StopIteration:
                        self.logger.warning(f"Seller {trade.seller_id} not found in agents")

                # Add trade to tracker with current round
                self.tracker.add_trade(trade, env_state.current_round)
                
                # Update price history
                self.tracker.update_price_history(trade.price, trade.coin)
                
                # Log trade details
                self.logger.info(
                    f"Trade executed - {trade.coin}: "
                    f"Price: {trade.price:.6f} USDC, "
                    f"Quantity: {trade.quantity:.6f}, "
                    f"Buyer: {trade.buyer_id}, "
                    f"Seller: {trade.seller_id}"
                )

            except Exception as ex:
                self.logger.error(f"Error processing trade {trade}: {str(ex)}")
                self.logger.exception("Exception details:")

        # Log current prices from global observation
        self.logger.info("Current market prices:")
        for token, price in global_observation.current_prices.items():
            self.logger.info(f"{token}: {price:.6f} USDC")


    def update_agent_balances(self, agent: CryptoEconomicAgent):
        """Update agent balances from EVM"""
        # Get ETH balance
        eth_balance = self.ethereum_interface.get_eth_balance(agent.ethereum_address)
        eth_balance_readable = eth_balance / 1e18
        
        # Get only the token balance for agent's assigned coin
        token_address = self.token_addresses.get(agent.coin)
        if token_address:
            balance = self.ethereum_interface.get_erc20_balance(
                agent.ethereum_address,
                token_address
            )
            # Update agent's portfolio
            agent.update_token_balance(agent.coin, balance)
        
        # Update ETH balance
        agent.update_cash_balance(eth_balance_readable)

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

        # Constants for decimal conversion
        WEI_TO_ETH = 1e18  # 1 ETH = 10^18 wei
        
        # Get token decimals for all tokens
        token_info = {}
        for token in self.environment.mechanism.tokens:
            token_address = self.ethereum_interface.get_token_address(token)
            info = self.ethereum_interface.get_erc20_info(token_address)
            token_info[token] = {
                'decimals': info['decimals'],
                'address': token_address
            }
        
        # Get USDC info
        usdc_info = self.ethereum_interface.get_erc20_info(self.quote_token_address)
        USDC_DECIMALS = usdc_info['decimals']

        # Calculate final portfolio values and rewards
        total_portfolio_value = 0.0

        for agent in self.agents:
            # Get current prices for all tokens
            current_prices = self.environment.mechanism.current_prices
            
            # Get ETH balance
            eth_balance = self.ethereum_interface.get_eth_balance(agent.economic_agent.ethereum_address)
            eth_balance_readable = eth_balance / WEI_TO_ETH
            
            # Initialize portfolio calculation
            portfolio_value = 0.0
            balance_info = []

            # Get USDC balance
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.economic_agent.ethereum_address,
                self.quote_token_address
            ) / (10 ** USDC_DECIMALS)
            portfolio_value += usdc_balance
            balance_info.append(f"USDC: {usdc_balance:.2f}")

            # Get balances for all trading tokens
            for token, info in token_info.items():
                token_balance = self.ethereum_interface.get_erc20_balance(
                    agent.economic_agent.ethereum_address,
                    info['address']
                ) / (10 ** info['decimals'])
                
                token_value = token_balance * current_prices.get(token, 0)
                portfolio_value += token_value
                
                balance_info.append(
                    f"{token}: {token_balance:.4f} (Value: {token_value:.2f} USDC)"
                )

            total_portfolio_value += portfolio_value

            # Log agent summary
            self.logger.info(f"\nAgent {agent.index} Summary:")
            self.logger.info(f"ETH Balance: {eth_balance_readable:.4f}")
            for balance in balance_info:
                self.logger.info(balance)
            self.logger.info(f"Total Portfolio Value: {portfolio_value:.2f} USDC")
            
            # Log agent's trading history
            trades = [t for t in self.tracker.trades if t.buyer_id == agent.id or t.seller_id == agent.id]
            if trades:
                self.logger.info("\nTrading History:")
                for trade in trades:
                    self.logger.info(
                        f"{'Bought' if trade.buyer_id == agent.id else 'Sold'} "
                        f"{trade.quantity:.4f} {trade.coin} @ {trade.price:.4f} USDC"
                    )

        # Log overall market summary
        self.logger.info("\nOverall Market Summary:")
        self.logger.info(f"Total Portfolio Value: {total_portfolio_value:.2f} USDC")
        self.logger.info("Final Token Prices:")
        for token, price in current_prices.items():
            self.logger.info(f"{token}: {price:.4f} USDC")
        
        # Log trading volume statistics using tracker's trades
        if self.tracker.trades:
            total_volume = sum(t.quantity * t.price for t in self.tracker.trades)
            avg_price = sum(t.price for t in self.tracker.trades) / len(self.tracker.trades)
            self.logger.info(f"\nTotal Trading Volume: {total_volume:.2f} USDC")
            self.logger.info(f"Average Trade Price: {avg_price:.4f} USDC")
            self.logger.info(f"Total Number of Trades: {len(self.tracker.trades)}")