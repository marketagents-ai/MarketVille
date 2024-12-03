# crypto_market.py

import logging
import random
from typing import Any, List, Dict, Type, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
from market_agents.environments.environment import (
    EnvironmentHistory, Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, MultiAgentEnvironment
)
from market_agents.memecoin_orchestrators.crypto_models import OrderType, MarketAction, Trade
from market_agents.memecoin_orchestrators.crypto_agent import CryptoEconomicAgent
from agent_evm_interface.agent_evm_interface import EthereumInterface
logger = logging.getLogger(__name__)


class MarketSummary(BaseModel):
    trades_count: int = Field(default=0, description="Number of trades executed")
    average_price: float = Field(default=0.0, description="Average price of trades")
    total_volume: int = Field(default=0, description="Total volume of trades")
    price_range: Tuple[float, float] = Field(default=(0.0, 0.0), description="Range of prices")


class CryptoMarketAction(LocalAction):
    action: MarketAction

    @field_validator('action')
    def validate_action(cls, v):
        if v.order_type in [OrderType.BUY, OrderType.SELL]:
            if v.quantity <= 0:
                raise ValueError("Quantity must be positive for buy and sell orders")
            if v.price <= 0:
                raise ValueError("Price must be positive for buy and sell orders")
        return v

    @classmethod
    def sample(cls, agent_id: str) -> 'CryptoMarketAction':
        order_type = random.choice(list(OrderType))
        if order_type == OrderType.HOLD:
            action = MarketAction(order_type=order_type)
        else:
            random_price = random.uniform(0.01, 1.0)
            random_quantity = random.randint(1, 1000)
            action = MarketAction(order_type=order_type, price=random_price, quantity=random_quantity)
        return cls(agent_id=agent_id, action=action)

    @classmethod
    def action_schema(cls) -> Dict[str, Any]:
        return MarketAction.model_json_schema()


class GlobalCryptoMarketAction(GlobalAction):
    actions: Dict[str, CryptoMarketAction]


class CryptoMarketObservation(BaseModel):
    trades: List[Trade] = Field(default_factory=list, description="List of trades the agent participated in")
    market_summary: MarketSummary = Field(default_factory=MarketSummary, description="Summary of market activity")
    current_price: float = Field(default=0.1, description="Current market price")
    portfolio_value: float = Field(default=0.0, description="Total value of the agent's portfolio")
    eth_balance: int = Field(default=0, description="Agent's ETH balance")
    token_balance: int = Field(default=0, description="Agent's token balance")
    price_history: List[float] = Field(default_factory=list, description="Historical prices")


class CryptoMarketLocalObservation(LocalObservation):
    observation: CryptoMarketObservation


class CryptoMarketGlobalObservation(GlobalObservation):
    observations: Dict[str, CryptoMarketLocalObservation]
    all_trades: List[Trade] = Field(default_factory=list, description="All trades executed in this round")
    market_summary: MarketSummary = Field(default_factory=MarketSummary, description="Summary of market activity")
    current_price: float = Field(default=0.1, description="Current market price")
    price_history: List[float] = Field(default_factory=list, description="Historical prices")


class CryptoMarketActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [CryptoMarketAction]

    @classmethod
    def get_action_schema(cls) -> Dict[str, Any]:
        return MarketAction.model_json_schema()


class CryptoMarketObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [CryptoMarketLocalObservation]


class CryptoMarketMechanism(Mechanism):
    max_rounds: int = Field(default=100, description="Maximum number of trading rounds")
    current_round: int = Field(default=0, description="Current round number")
    trades: List[Trade] = Field(default_factory=list, description="List of executed trades")
    coin_name: str = Field(default="DOGE", description="Cryptocurrency being traded")
    current_price: float = Field(default=0.1, description="Current market price")
    price_history: List[float] = Field(default_factory=lambda: [0.1])
    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    agent_registry: Dict[str, Any] = Field(default_factory=dict, description="Registry of agents")
    ethereum_interface: EthereumInterface = Field(default_factory=EthereumInterface, description="Ethereum Interface")
    token_addresses: Dict[str, str] = Field(default_factory=dict, description="Token addresses")
    orderbook_address: str = Field(default="", description="Orderbook contract address")
    minter_private_key: str = Field(default="", description="Private key of the minter account")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def register_agent(self, agent_id: str, agent: CryptoEconomicAgent):
        """Register an agent with the mechanism."""
        if not isinstance(agent, CryptoEconomicAgent):
            raise ValueError(f"Agent must be a CryptoEconomicAgent, got {type(agent)}")
        
        self.agent_registry[str(agent_id)] = agent
        logger.info(f"Registered agent {agent_id} with address {agent.ethereum_address}")

    def setup(self):
        """Initialize token addresses, blockchain parameters, and current price."""
        self.token_addresses = self.ethereum_interface.testnet_data['token_addresses']
        self.orderbook_address = self.ethereum_interface.testnet_data['orderbook_address']
        self.minter_private_key = self.ethereum_interface.accounts[0]['private_key']
        
        # Initialize current price from orderbook
        try:
            token_address = self.ethereum_interface.get_token_address(self.coin_name)
            usdc_address = self.ethereum_interface.get_token_address('USDC')
            
            pair_info = self.ethereum_interface.get_pair_info(
                token_address,
                usdc_address
            )
            
            base_unit_price = pair_info['token0_price_in_token1']
            self.current_price = self._convert_to_decimal_price(base_unit_price)
            if self.current_price == 0:
                self.current_price = 0.1
                
        except Exception as e:
            logger.warning(f"Failed to get initial price from blockchain: {str(e)}. Using default price of 0.1")
            self.current_price = 0.0
            
        # Initialize price history with current price
        self.price_history = [self.current_price]

    def step(self, action: GlobalCryptoMarketAction) -> EnvironmentStep:
        self.current_round += 1

        # Process actions (buy/sell/hold) using the EthereumInterface
        new_trades = self._process_actions(action.actions)
        self.trades.extend(new_trades)

        # Update price based on recent trades
        self._update_price(new_trades)

        market_summary = self._create_market_summary(new_trades)
        observations = self._create_observations(market_summary)
        done = self.current_round >= self.max_rounds

        return EnvironmentStep(
            global_observation=CryptoMarketGlobalObservation(
                observations=observations,
                all_trades=new_trades,
                market_summary=market_summary,
                current_price=self.current_price,
                price_history=self.price_history
            ),
            done=done,
            info={"current_round": self.current_round}
        )

    def _process_actions(self, actions: Dict[str, CryptoMarketAction]) -> List[Trade]:
        trades = []
        buy_orders = []
        sell_orders = []

        # Collect buy and sell orders
        for agent_id, action in actions.items():
            agent = self.agent_registry.get(agent_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found in registry.")
                continue

            market_action = action.action
            if market_action.order_type == OrderType.HOLD:
                logger.info(f"Agent {agent_id} holds.")
                continue

            try:
                if market_action.order_type == OrderType.BUY:
                    buy_orders.append((agent_id, agent, market_action))
                elif market_action.order_type == OrderType.SELL:
                    sell_orders.append((agent_id, agent, market_action))
            except Exception as e:
                logger.error(f"Error processing action for agent {agent_id}: {str(e)}")
                continue

        buy_orders.sort(key=lambda x: x[2].price, reverse=True)
        sell_orders.sort(key=lambda x: x[2].price)

        # Match orders
        while buy_orders and sell_orders:
            buyer_id, buyer, buy_order = buy_orders[0]
            seller_id, seller, sell_order = sell_orders[0]

            # Check if prices match (within tolerance)
            if buy_order.price >= sell_order.price:
                # Calculate trade price as midpoint
                trade_price = (buy_order.price + sell_order.price) / 2
                
                # Calculate trade quantity
                trade_quantity = min(buy_order.quantity, sell_order.quantity)

                # Execute the trade
                try:
                    # Transfer tokens between agents
                    self._execute_p2p_trade(
                        buyer=buyer,
                        seller=seller,
                        price=trade_price,
                        quantity=trade_quantity
                    )

                    # Record the trade
                    trade = Trade(
                        trade_id=len(self.trades),
                        buyer_id=buyer_id,
                        seller_id=seller_id,
                        price=trade_price,
                        bid_price=buy_order.price,
                        ask_price=sell_order.price,
                        quantity=trade_quantity,
                        coin=self.coin_name
                    )
                    trades.append(trade)
                    logger.info(f"Matched trade: {buyer_id} buys {trade_quantity} {self.coin_name} " +
                              f"from {seller_id} at {trade_price} USDC")

                    # Update order quantities
                    buy_order.quantity -= trade_quantity
                    sell_order.quantity -= trade_quantity

                    # Remove fulfilled orders
                    if buy_order.quantity == 0:
                        buy_orders.pop(0)
                    if sell_order.quantity == 0:
                        sell_orders.pop(0)

                except Exception as e:
                    logger.error(f"Failed to execute trade: {str(e)}")
                    logger.exception("Full traceback:")
                    # Remove problematic orders
                    buy_orders.pop(0)
                    sell_orders.pop(0)
            else:
                break

        return trades

    def _execute_p2p_trade(self, buyer: CryptoEconomicAgent, seller: CryptoEconomicAgent, 
                          price: float, quantity: int) -> None:
        """Execute a peer-to-peer trade between two agents."""
        # Get token addresses
        token_address = self.ethereum_interface.get_token_address(self.coin_name)
        usdc_address = self.ethereum_interface.get_token_address('USDC')
        
        # Get decimals
        token_decimals = self.ethereum_interface.get_erc20_info(token_address)['decimals']
        usdc_decimals = self.ethereum_interface.get_erc20_info(usdc_address)['decimals']
        
        # Convert amounts to proper decimals
        usdc_amount = int(price * quantity * (10 ** usdc_decimals))
        token_amount = int(quantity * (10 ** token_decimals))

        # Verify balances
        if not self._verify_balances(buyer, seller, usdc_amount, token_amount, usdc_address, token_address):
            raise ValueError("Insufficient balance for trade")

        # Execute the transfers
        self._transfer_tokens(buyer, seller, usdc_amount, token_amount, usdc_address, token_address)

    def _verify_balances(self, buyer: CryptoEconomicAgent, seller: CryptoEconomicAgent,
                        usdc_amount: int, token_amount: int,
                        usdc_address: str, token_address: str) -> bool:
        """Verify that both parties have sufficient balances for the trade."""
        try:
            # Check buyer's USDC balance
            buyer_usdc_balance = self.ethereum_interface.get_erc20_balance(
                buyer.ethereum_address,
                usdc_address
            )
            if buyer_usdc_balance < usdc_amount:
                logger.error(f"Buyer {buyer.id} has insufficient USDC balance. " +
                           f"Has: {buyer_usdc_balance}, Needs: {usdc_amount}")
                return False

            # Check seller's token balance
            seller_token_balance = self.ethereum_interface.get_erc20_balance(
                seller.ethereum_address,
                token_address
            )
            if seller_token_balance < token_amount:
                logger.error(f"Seller {seller.id} has insufficient {self.coin_name} balance. " +
                           f"Has: {seller_token_balance}, Needs: {token_amount}")
                return False

            # Check allowances
            buyer_usdc_allowance = self.ethereum_interface.get_erc20_allowance(
                owner=buyer.ethereum_address,
                spender=self.orderbook_address,
                contract_address=usdc_address
            )
            if buyer_usdc_allowance < usdc_amount:
                tx_hash = self.ethereum_interface.approve_erc20(
                    spender=self.orderbook_address,
                    amount=usdc_amount,
                    contract_address=usdc_address,
                    private_key=buyer.private_key
                )
                logger.info(f"Buyer {buyer.id} approved {usdc_amount} USDC. TxHash: {tx_hash}")

            seller_token_allowance = self.ethereum_interface.get_erc20_allowance(
                owner=seller.ethereum_address,
                spender=self.orderbook_address,
                contract_address=token_address
            )
            if seller_token_allowance < token_amount:
                tx_hash = self.ethereum_interface.approve_erc20(
                    spender=self.orderbook_address,
                    amount=token_amount,
                    contract_address=token_address,
                    private_key=seller.private_key
                )
                logger.info(f"Seller {seller.id} approved {token_amount} {self.coin_name}. TxHash: {tx_hash}")

            return True

        except Exception as e:
            logger.error(f"Error verifying balances: {str(e)}")
            return False

    def _transfer_tokens(self, buyer: CryptoEconomicAgent, seller: CryptoEconomicAgent,
                        usdc_amount: int, token_amount: int,
                        usdc_address: str, token_address: str) -> None:
        """Execute the token transfers between buyer and seller."""
        try:
            # Transfer USDC from buyer to seller
            tx_hash = self.ethereum_interface.send_erc20(
                to_address=seller.ethereum_address,
                amount=usdc_amount,
                contract_address=usdc_address,
                private_key=buyer.private_key
            )
            logger.info(f"USDC transfer complete. TxHash: {tx_hash}")

            # Transfer tokens from seller to buyer
            tx_hash = self.ethereum_interface.send_erc20(
                to_address=buyer.ethereum_address,
                amount=token_amount,
                contract_address=token_address,
                private_key=seller.private_key
            )
            logger.info(f"Token transfer complete. TxHash: {tx_hash}")

        except Exception as e:
            logger.error(f"Error executing transfers: {str(e)}")
            raise

    def _execute_buy(self, agent: CryptoEconomicAgent, market_action: MarketAction) -> Optional[Trade]:
        """Agent buys tokens using USDC."""
        source_token_address = self.ethereum_interface.get_token_address('USDC')
        target_token_address = self.ethereum_interface.get_token_address(self.coin_name)

        # Get token decimals
        usdc_decimals = self.ethereum_interface.get_erc20_info(source_token_address)['decimals']
        token_decimals = self.ethereum_interface.get_erc20_info(target_token_address)['decimals']

        # Convert amounts to proper decimals
        usdc_amount = int(market_action.price * market_action.quantity * (10 ** usdc_decimals))
        token_amount = int(market_action.quantity * (10 ** token_decimals))

        # Check USDC balance (compare in same units)
        usdc_balance = self.ethereum_interface.get_erc20_balance(
            agent.ethereum_address,
            source_token_address
        )
        if usdc_balance < usdc_amount:
            logger.error(f"Agent {agent.id} has insufficient USDC balance. " + 
                        f"Has: {usdc_balance / 10**usdc_decimals}, " +
                        f"Needs: {market_action.price * market_action.quantity}")
            return None

        # Rest of the buy logic with proper decimal handling...
        allowance = self.ethereum_interface.get_erc20_allowance(
            owner=agent.ethereum_address,
            spender=self.orderbook_address,
            contract_address=source_token_address
        )
        
        if allowance < usdc_amount:
            tx_hash = self.ethereum_interface.approve_erc20(
                spender=self.orderbook_address,
                amount=usdc_amount,
                contract_address=source_token_address,
                private_key=agent.private_key
            )
            logger.info(f"Agent {agent.id} approved {usdc_amount/(10**usdc_decimals)} USDC. TxHash: {tx_hash}")

        tx_hash = self.ethereum_interface.swap(
            source_token_address=source_token_address,
            source_token_amount=usdc_amount,
            target_token_address=target_token_address,
            private_key=agent.private_key
        )
        logger.info(f"Agent {agent.id} executed buy {market_action.quantity} {self.coin_name} for {usdc_amount/(10**usdc_decimals)} USDC. TxHash: {tx_hash}")

        return Trade(
            trade_id=len(self.trades),
            buyer_id=agent.id,
            seller_id="MARKET_MAKER",
            price=market_action.price,
            bid_price=market_action.price,
            ask_price=market_action.price,
            quantity=market_action.quantity,
            coin=self.coin_name
        )

    def _execute_sell(self, agent: CryptoEconomicAgent, market_action: MarketAction) -> Optional[Trade]:
        """Agent sells tokens for USDC."""
        source_token_address = self.ethereum_interface.get_token_address(self.coin_name)
        target_token_address = self.ethereum_interface.get_token_address('USDC')

        # Get token decimals
        token_decimals = self.ethereum_interface.get_erc20_info(source_token_address)['decimals']
        usdc_decimals = self.ethereum_interface.get_erc20_info(target_token_address)['decimals']

        # Convert quantity to proper decimals
        token_amount = int(market_action.quantity * (10 ** token_decimals))

        # Check token balance
        token_balance = self.ethereum_interface.get_erc20_balance(
            agent.ethereum_address,
            source_token_address
        )
        if token_balance < token_amount:
            logger.error(f"Agent {agent.id} has insufficient {self.coin_name} balance. Has: {token_balance/(10**token_decimals)}, Needs: {market_action.quantity}")
            return None

        # Rest of the sell logic with proper decimal handling...
        allowance = self.ethereum_interface.get_erc20_allowance(
            owner=agent.ethereum_address,
            spender=self.orderbook_address,
            contract_address=source_token_address
        )
        
        if allowance < token_amount:
            tx_hash = self.ethereum_interface.approve_erc20(
                spender=self.orderbook_address,
                amount=token_amount,
                contract_address=source_token_address,
                private_key=agent.private_key
            )
            logger.info(f"Agent {agent.id} approved {market_action.quantity} {self.coin_name}. TxHash: {tx_hash}")

        tx_hash = self.ethereum_interface.swap(
            source_token_address=source_token_address,
            source_token_amount=token_amount,
            target_token_address=target_token_address,
            private_key=agent.private_key
        )
        logger.info(f"Agent {agent.id} executed sell {market_action.quantity} {self.coin_name} for {market_action.price * market_action.quantity} USDC. TxHash: {tx_hash}")

        return Trade(
            trade_id=len(self.trades),
            buyer_id="Orderbook",
            seller_id=agent.id,
            price=market_action.price,
            bid_price=market_action.price,
            ask_price=market_action.price,
            quantity=market_action.quantity,
            coin=self.coin_name
        )
    def _update_price(self, trades: List[Trade]):
        if trades:
            prices = [trade.price for trade in trades]
            self.current_price = sum(prices) / len(prices)
        # Optionally, fetch current price from the orderbook contract
        # self.current_price = self.ethereum_interface.get_current_price(...)
        self.price_history.append(self.current_price)

    def _create_observations(self, market_summary: MarketSummary) -> Dict[str, CryptoMarketLocalObservation]:
        observations = {}
        for agent_id, agent in self.agent_registry.items():
            # Fetch agent balances
            eth_balance = self.ethereum_interface.get_eth_balance(agent.ethereum_address)
            token_balance = self.ethereum_interface.get_erc20_balance(
                agent.ethereum_address,
                self.token_addresses['USDC']
            )

            # Calculate portfolio value
            portfolio_value = eth_balance + token_balance * self.current_price  # Simplified

            observation = CryptoMarketObservation(
                trades=[],  # Trades involving the agent can be added here
                market_summary=market_summary,
                current_price=self.current_price,
                portfolio_value=portfolio_value,
                eth_balance=eth_balance,
                token_balance=token_balance,
                price_history=self.price_history.copy()
            )

            observations[agent_id] = CryptoMarketLocalObservation(
                agent_id=agent_id,
                observation=observation
            )

        return observations
    
    def _convert_to_decimal_price(self, base_unit_price: int, decimals: int = 18) -> float:
        return base_unit_price / (10 ** decimals)

    def _convert_to_base_units(self, decimal_price: float, decimals: int = 18) -> int:
        return int(decimal_price * (10 ** decimals))

    def _create_market_summary(self, trades: List[Trade]) -> MarketSummary:
        if not trades:
            return MarketSummary(trades_count=0, average_price=self.current_price, total_volume=0, price_range=(self.current_price, self.current_price))

        prices = [trade.price for trade in trades]
        total_volume = sum(trade.quantity for trade in trades)
        return MarketSummary(
            trades_count=len(trades),
            average_price=sum(prices) / len(prices),
            total_volume=total_volume,
            price_range=(min(prices), max(prices))
        )

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "current_price": self.current_price,
            "price_history": self.price_history,
            "trades": [trade.model_dump() for trade in self.trades],
        }

    def reset(self) -> None:
        self.current_round = 0
        self.trades = []
        self.current_price = 0.1
        self.price_history = [self.current_price]


class CryptoMarket(MultiAgentEnvironment):
    name: str = Field(default="Crypto Market", description="Name of the crypto market")
    action_space: CryptoMarketActionSpace = Field(default_factory=CryptoMarketActionSpace, description="Action space of the crypto market")
    observation_space: CryptoMarketObservationSpace = Field(default_factory=CryptoMarketObservationSpace, description="Observation space of the crypto market")
    mechanism: CryptoMarketMechanism = Field(default_factory=CryptoMarketMechanism, description="Mechanism of the crypto market")
    agents: Dict[str, CryptoEconomicAgent] = Field(default_factory=dict, description="Dictionary of agents in the market")

    def __init__(self, agents: Dict[str, CryptoEconomicAgent], **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        self.mechanism.agent_registry = {}
        
        # Fix: Properly register agents with string IDs
        for agent_id, agent in agents.items():
            str_id = str(agent_id)
            if hasattr(agent, 'economic_agent'):
                # If agent is wrapped, register the economic agent
                self.mechanism.agent_registry[str_id] = agent.economic_agent
            else:
                # If agent is direct CryptoEconomicAgent instance
                self.mechanism.agent_registry[str_id] = agent
            
            # Ensure agent has its ID set
            if hasattr(agent, 'economic_agent'):
                agent.economic_agent.id = str_id
            else:
                agent.id = str_id

        # Setup mechanism after registry is populated
        self.mechanism.setup()

    def reset(self) -> GlobalObservation:
        self.current_step = 0
        self.history = EnvironmentHistory()
        self.mechanism.reset()
        observations = self.mechanism._create_observations(MarketSummary())

        return CryptoMarketGlobalObservation(
            observations=observations,
            all_trades=[],
            market_summary=MarketSummary(),
            current_price=self.mechanism.current_price,
            price_history=self.mechanism.price_history.copy()
        )

    def step(self, actions: GlobalAction) -> EnvironmentStep:
        step_result = self.mechanism.step(actions)
        self.current_step += 1
        self.update_history(actions, step_result)
        return step_result

    def render(self):
        pass
