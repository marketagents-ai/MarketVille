# crypto_agent.py

from typing import List, Optional
from pydantic import Field, computed_field
import random
import logging
from market_agents.memecoin_orchestrators.crypto_models import (
    MarketAction,
    OrderType,
    Position,
    Crypto,
    Trade,
    Endowment,
    Portfolio
)
from market_agents.economics.econ_agent import EconomicAgent as BaseEconomicAgent

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CryptoEconomicAgent(BaseEconomicAgent):
    endowment: Endowment
    max_relative_spread: float = Field(default=0.05)
    pending_orders: List[MarketAction] = Field(default_factory=list)
    archived_endowments: List[Endowment] = Field(default_factory=list)
    risk_aversion: float = Field(default=0.5)
    expected_return: float = Field(default=0.05)
    coin: str = Field(default="DOGE")

    def archive_endowment(self, new_portfolio: Optional[Portfolio] = None):
        self.archived_endowments.append(self.endowment.model_copy(deep=True))
        if new_portfolio is None:
            new_endowment = self.endowment.model_copy(deep=True, update={"trades": []})
        else:
            new_endowment = self.endowment.model_copy(
                deep=True,
                update={"trades": [], "initial_portfolio": new_portfolio}
            )
        self.endowment = new_endowment

    @computed_field
    @property
    def current_portfolio(self) -> Portfolio:
        return self.endowment.current_portfolio

    @computed_field
    @property
    def current_cash(self) -> float:
        return self.current_portfolio.cash

    @computed_field
    @property
    def current_crypto_quantity(self) -> int:
        return self.current_portfolio.get_crypto_quantity(self.coin)

    def calculate_average_cost(self) -> float:
        crypto = next((c for c in self.current_portfolio.coins if c.symbol == self.coin), None)
        if crypto:
            return crypto.average_cost()
        return 0.0

    def generate_order(self, market_price: float) -> Optional[MarketAction]:
        average_cost = self.calculate_average_cost()
        if self.current_crypto_quantity > 0:
            expected_profit = market_price - average_cost
            expected_return = expected_profit / average_cost if average_cost != 0 else 0.0
        else:
            expected_return = 0.0

        # Decide whether to buy or sell based on expected return
        if expected_return > self.expected_return:
            # Consider selling
            min_sell_price = market_price - market_price * self.max_relative_spread
            price = random.uniform(min_sell_price, market_price)
            quantity = self.determine_quantity_to_sell()
            if quantity > 0:
                order = MarketAction(order_type=OrderType.SELL, price=price, quantity=quantity)
                self.pending_orders.append(order)
                return order
        elif expected_return < -self.expected_return:
            # Consider buying
            max_buy_price = market_price + market_price * self.max_relative_spread
            price = random.uniform(market_price, max_buy_price)
            quantity = self.determine_quantity_to_buy(price)
            if quantity > 0:
                order = MarketAction(order_type=OrderType.BUY, price=price, quantity=quantity)
                self.pending_orders.append(order)
                return order
        else:
            # Hold
            return None

    def determine_quantity_to_buy(self, price: float) -> int:
        affordable_quantity = int(self.current_cash // price)
        if affordable_quantity <= 0:
            return 0
        quantity = max(1, int(affordable_quantity * (1 - self.risk_aversion)))
        return quantity

    def determine_quantity_to_sell(self) -> int:
        current_quantity = self.current_crypto_quantity
        if current_quantity <= 0:
            return 0
        quantity = max(1, int(current_quantity * self.risk_aversion))
        return quantity

    def process_trade(self, trade: Trade):
        matching_order = next(
            (order for order in self.pending_orders
             if order.price == trade.price and
             order.quantity == trade.quantity and
             order.order_type == (OrderType.BUY if trade.buyer_id == self.id else OrderType.SELL)),
            None
        )
        if matching_order:
            self.pending_orders.remove(matching_order)
        else:
            logger.warning(f"Trade processed but matching order not found for agent {self.id}")

        self.endowment.add_trade(trade)

    def reset_pending_orders(self):
        self.pending_orders = []

    def calculate_portfolio_value(self, market_price: float) -> float:
        crypto_value = self.current_crypto_quantity * market_price
        total_value = self.current_cash + crypto_value
        return total_value

    def calculate_total_cost(self) -> float:
        total_cost = 0.0
        for crypto in self.current_portfolio.coins:
            if crypto.symbol == self.coin:
                total_cost += sum(position.quantity * position.purchase_price for position in crypto.positions)
        return total_cost

    def calculate_unrealized_profit(self, market_price: float) -> float:
        total_cost = self.calculate_total_cost()
        crypto_value = self.current_crypto_quantity * market_price
        profit = crypto_value - total_cost
        return profit

    def calculate_utility(self, market_price: float) -> float:
        profit = self.calculate_unrealized_profit(market_price)
        utility = self.current_cash + profit
        return utility

    def print_status(self, market_price: float):
        print(f"\nAgent ID: {self.id}")
        print(f"Current Portfolio:")
        print(f"  Cash: {self.current_cash:.2f}")
        print(f"  {self.coin} Coins: {self.current_crypto_quantity}")
        average_cost = self.calculate_average_cost()
        print(f"  Average Cost: {average_cost:.2f}")
        total_cost = self.calculate_total_cost()
        print(f"  Total Cost Basis: {total_cost:.2f}")
        portfolio_value = self.calculate_portfolio_value(market_price)
        unrealized_profit = self.calculate_unrealized_profit(market_price)
        print(f"  Portfolio Value: {portfolio_value:.2f}")
        print(f"  Unrealized Profit: {unrealized_profit:.2f}")

    # Override methods from BaseEconomicAgent that are not applicable to crypto trading
    def is_buyer(self, good_name: str) -> bool:
        return good_name == self.coin

    def is_seller(self, good_name: str) -> bool:
        return good_name == self.coin

    def get_current_value(self, good_name: str) -> Optional[float]:
        if good_name == self.coin:
            return self.calculate_portfolio_value(market_price=1.0)  # Placeholder
        return None

    def get_current_cost(self, good_name: str) -> Optional[float]:
        if good_name == self.coin:
            return self.calculate_total_cost()
        return None

    def calculate_individual_surplus(self) -> float:
        # Calculate surplus as current utility minus initial utility
        initial_crypto_quantity = self.endowment.initial_portfolio.get_crypto_quantity(self.coin)
        initial_portfolio_value = self.endowment.initial_portfolio.cash + initial_crypto_quantity * 1.0  # Assuming initial market price is $1.0
        current_portfolio_value = self.calculate_portfolio_value(market_price=1.0)  # Using the same placeholder
        surplus = current_portfolio_value - initial_portfolio_value
        return surplus
    
    def remove_positions(self, quantity_to_sell):
        # Implement FIFO (First-In, First-Out) for positions
        quantity_remaining = quantity_to_sell
        total_cost_basis = 0.0
        total_quantity_sold = 0

        while quantity_remaining > 0 and self.current_portfolio.coins:
            crypto = self.current_portfolio.coins[0]  # Assuming only one coin
            position = crypto.positions[0]
            quantity_sold = min(position.quantity, quantity_remaining)
            total_cost_basis += position.purchase_price * quantity_sold
            position.quantity -= quantity_sold
            quantity_remaining -= quantity_sold
            total_quantity_sold += quantity_sold

            if position.quantity == 0:
                crypto.positions.pop(0)
                if not crypto.positions:
                    self.current_portfolio.coins.pop(0)

        if quantity_remaining > 0:
            raise ValueError("Not enough coins to sell.")

        average_cost_basis = total_cost_basis / total_quantity_sold
        return average_cost_basis, total_quantity_sold

    def add_position(self, purchase_price, quantity):
        # Add new position to the portfolio
        coin = self.coin
        crypto = next((c for c in self.current_portfolio.coins if c.symbol == coin), None)
        if not crypto:
            crypto = Crypto(symbol=coin, positions=[])
            self.current_portfolio.coins.append(crypto)
        position = Position(quantity=quantity, purchase_price=purchase_price)
        crypto.positions.append(position)

