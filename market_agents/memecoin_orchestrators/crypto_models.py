# crypto_models.py

from pydantic import BaseModel, Field, computed_field, model_validator
from functools import cached_property
from typing import List, Dict, Self, Optional
from enum import Enum
from copy import deepcopy
from datetime import datetime
import os
import json
import tempfile
from pathlib import Path


class SavableBaseModel(BaseModel):
    name: str

    def save_to_json(self, folder_path: str) -> str:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name.replace(' ', '_')}_{timestamp}.json"
        file_path = os.path.join(folder_path, filename)

        try:
            data = self.model_dump(mode='json')
            with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
                json.dump(data, temp_file, indent=2)
            os.replace(temp_file.name, file_path)
            print(f"State saved to {file_path}")
        except Exception as e:
            print(f"Error saving state to {file_path}")
            print(f"Error message: {str(e)}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise

        return file_path

    @classmethod
    def load_from_json(cls, file_path: str) -> Self:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls.model_validate(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}")
            print(f"Error message: {str(e)}")
            with open(file_path, 'r') as f:
                print(f"File contents:\n{f.read()}")
            raise


class OrderType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MarketAction(BaseModel):
    order_type: OrderType = Field(default=OrderType.HOLD, description="Type of order: 'buy', 'sell', or 'hold'")
    token: Optional[str] = Field(default=None, description="Token symbol to trade")
    price: Optional[float] = Field(default=None, description="Price of the order (not applicable for 'hold')")
    quantity: Optional[int] = Field(default=None, ge=0, description="Quantity of the order (not applicable for 'hold')")

    @model_validator(mode='after')
    def validate_order_type_and_fields(self):        
        if self.order_type == OrderType.HOLD:
            if self.price is not None or self.quantity is not None or self.token is not None:
                raise ValueError("Hold orders should not specify price, quantity, or token")
        else:
            if None in (self.price, self.quantity, self.token):
                raise ValueError(f"Buy and sell orders must specify price, quantity, and token. Got: price={self.price}, quantity={self.quantity}, token={self.token}")
            if self.price <= 0 or self.quantity <= 0:
                raise ValueError("Price and quantity must be positive for buy and sell orders")
        return self


class CryptoOrder(MarketAction):
    agent_id: str

    @computed_field
    @property
    def is_buy_order(self) -> bool:
        return self.order_type == OrderType.BUY


class Trade(BaseModel):
    trade_id: int = Field(..., description="Unique identifier for the trade")
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller")
    price: float = Field(..., description="The price at which the trade was executed")
    bid_price: float = Field(ge=0, description="The bid price")
    ask_price: float = Field(ge=0, description="The ask price")
    quantity: int = Field(default=1, description="The quantity traded")
    coin: str = Field(default="DOGE", description="The symbol of the coin traded")
    action_type: OrderType
    tx_hash: Optional[str] = Field(default=None, description="Transaction hash from the blockchain")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the trade")

    @model_validator(mode='after')
    def rational_trade(self):
        if self.ask_price > self.bid_price:
            raise ValueError(f"Ask price {self.ask_price} is more than bid price {self.bid_price}")
        return self


class Position(BaseModel):
    quantity: int
    purchase_price: float


class Crypto(BaseModel):
    symbol: str
    positions: List[Position] = Field(default_factory=list)

    def total_quantity(self) -> int:
        return sum(position.quantity for position in self.positions)

    def average_cost(self) -> float:
        total_quantity = self.total_quantity()
        if total_quantity == 0:
            return 0.0
        total_cost = sum(position.quantity * position.purchase_price for position in self.positions)
        return total_cost / total_quantity


class Portfolio(BaseModel):
    cash: float
    coins: List[Crypto] = Field(default_factory=list)

    @computed_field
    @cached_property
    def coins_dict(self) -> Dict[str, int]:
        return {coin.symbol: coin.total_quantity() for coin in self.coins}

    def update_crypto(self, symbol: str, quantity: int, purchase_price: Optional[float] = None):
        crypto = next((c for c in self.coins if c.symbol == symbol), None)
        if crypto is None:
            if quantity > 0:
                crypto = Crypto(symbol=symbol, positions=[])
                self.coins.append(crypto)
            else:
                return
        if crypto is not None:
            if purchase_price is not None and quantity > 0:
                crypto.positions.append(Position(quantity=quantity, purchase_price=purchase_price))
            elif quantity < 0:
                self.sell_crypto_positions(crypto, -quantity)
            if crypto.total_quantity() == 0:
                self.coins.remove(crypto)

    def sell_crypto_positions(self, crypto: Crypto, quantity_to_sell: int):
        positions = crypto.positions
        quantity_remaining = quantity_to_sell
        while quantity_remaining > 0 and positions:
            position = positions[0]
            if position.quantity > quantity_remaining:
                position.quantity -= quantity_remaining
                quantity_remaining = 0
            else:
                quantity_remaining -= position.quantity
                positions.pop(0)

    def get_crypto_quantity(self, symbol: str) -> int:
        crypto = next((c for c in self.coins if c.symbol == symbol), None)
        if crypto:
            return crypto.total_quantity()
        return 0


class Endowment(BaseModel):
    initial_portfolio: Portfolio
    trades: List[Trade] = Field(default_factory=list)
    agent_id: str

    @computed_field
    @property
    def current_portfolio(self) -> Portfolio:
        temp_portfolio = deepcopy(self.initial_portfolio)

        for trade in self.trades:
            if trade.buyer_id == self.agent_id:
                temp_portfolio.cash -= trade.price * trade.quantity
                temp_portfolio.update_crypto(
                    symbol=trade.coin,
                    quantity=trade.quantity,
                    purchase_price=trade.price
                )
            elif trade.seller_id == self.agent_id:
                temp_portfolio.cash += trade.price * trade.quantity
                temp_portfolio.update_crypto(
                    symbol=trade.coin,
                    quantity=-trade.quantity
                )
            else:
                raise ValueError(f"Trade {trade} not for agent {self.agent_id}")

        return temp_portfolio

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        if 'current_portfolio' in self.__dict__:
            del self.__dict__['current_portfolio']

    def simulate_trade(self, trade: Trade) -> Portfolio:
        temp_portfolio = deepcopy(self.current_portfolio)

        if trade.buyer_id == self.agent_id:
            temp_portfolio.cash -= trade.price * trade.quantity
            temp_portfolio.update_crypto(
                symbol=trade.coin,
                quantity=trade.quantity,
                purchase_price=trade.price
            )
        elif trade.seller_id == self.agent_id:
            temp_portfolio.cash += trade.price * trade.quantity
            temp_portfolio.update_crypto(
                symbol=trade.coin,
                quantity=-trade.quantity
            )

        return temp_portfolio
