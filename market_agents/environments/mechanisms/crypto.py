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


    def setup(self):
        """Initialize token addresses and other blockchain parameters."""
        self.token_addresses = self.ethereum_interface.testnet_data['token_addresses']
        self.orderbook_address = self.ethereum_interface.testnet_data['orderbook_address']
        self.minter_private_key = self.ethereum_interface.accounts[0]['private_key']  # Use the first account as minter

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
                    trade = self._execute_buy(agent, market_action)
                elif market_action.order_type == OrderType.SELL:
                    trade = self._execute_sell(agent, market_action)
                else:
                    logger.warning(f"Invalid order type from agent {agent_id}: {market_action.order_type}")
                    continue
                if trade:
                    trades.append(trade)
            except Exception as e:
                logger.error(f"Error processing action for agent {agent_id}: {str(e)}")

        return trades

    def _execute_buy(self, agent: CryptoEconomicAgent, market_action: MarketAction) -> Optional[Trade]:
        """Agent buys tokens using ETH."""
        source_token_address = self.ethereum_interface.testnet_data['eth_address']  # ETH address placeholder
        target_token_address = self.token_addresses[0]  # Assuming the first token is the one being traded

        # Agent must approve the orderbook to spend their ETH (handled internally)
        # Since ETH doesn't require approval, we can proceed directly

        # Execute swap on the orderbook contract
        tx_hash = self.ethereum_interface.swap(
            source_token_address=source_token_address,
            source_token_amount=market_action.price * market_action.quantity,
            target_token_address=target_token_address,
            private_key=agent.private_key
        )
        logger.info(f"Agent {agent.id} executed buy. TxHash: {tx_hash}")

        # Record the trade
        trade = Trade(
            trade_id=len(self.trades),
            buyer_id=agent.id,
            seller_id="Orderbook",
            price=market_action.price,
            bid_price=market_action.price,
            ask_price=market_action.price,
            quantity=market_action.quantity,
            coin=self.coin
        )
        return trade

    def _execute_sell(self, agent: CryptoEconomicAgent, market_action: MarketAction) -> Optional[Trade]:
        """Agent sells tokens for ETH."""
        source_token_address = self.token_addresses[0]  # Assuming the first token is the one being traded
        target_token_address = self.ethereum_interface.testnet_data['eth_address']  # ETH address placeholder

        # Agent must approve the orderbook to spend their tokens
        allowance = self.ethereum_interface.get_erc20_allowance(
            owner=agent.ethereum_address,
            spender=self.orderbook_address,
            contract_address=source_token_address
        )
        if allowance < market_action.quantity:
            # Approve the required amount
            tx_hash = self.ethereum_interface.approve_erc20(
                spender=self.orderbook_address,
                amount=market_action.quantity,
                contract_address=source_token_address,
                private_key=agent.private_key
            )
            logger.info(f"Agent {agent.id} approved {market_action.quantity} tokens. TxHash: {tx_hash}")

        # Execute swap on the orderbook contract
        tx_hash = self.ethereum_interface.swap(
            source_token_address=source_token_address,
            source_token_amount=market_action.quantity,
            target_token_address=target_token_address,
            private_key=agent.private_key
        )
        logger.info(f"Agent {agent.id} executed sell. TxHash: {tx_hash}")

        # Record the trade
        trade = Trade(
            trade_id=len(self.trades),
            buyer_id="Orderbook",
            seller_id=agent.id,
            price=market_action.price,
            bid_price=market_action.price,
            ask_price=market_action.price,
            quantity=market_action.quantity,
            coin=self.coin
        )
        return trade

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
                self.token_addresses[0]
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
