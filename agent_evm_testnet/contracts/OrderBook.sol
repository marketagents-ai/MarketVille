// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract OrderBook is Ownable {
    mapping(address => uint256) prices;
    uint256 public fee = 1; 
    address public price_token;
    
    event FeeUpdate(uint256 oldFee, uint256 newFee);
    event BuyOrder(address indexed user, address indexed token, uint256 amount, uint256 price, uint256 new_price);
    event SellOrder(address indexed user, address indexed token, uint256 amount, uint256 price, uint256 new_price);
    
    // add constructor for ownable
    constructor() Ownable(msg.sender) {}

    function set_price_token(address token) public onlyOwner {
        price_token = token;
    }

    function set_fee(uint256 new_fee) public onlyOwner {
        require(new_fee <= 50, "Fee cannot exceed 5%");
        emit FeeUpdate(fee, new_fee);
        fee = new_fee;
    }


    function get_token_balance(address token) public view returns (uint256) {
        IERC20 token = IERC20(token);
        return token.balanceOf(address(this));
    }

    function set_price(address token, uint256 new_price) public onlyOwner {
        require(price_token != address(0), "Price token not set");
        prices[token] = new_price;
    }

    function set_price_batch(address[] memory tokens, uint256[] memory new_prices) public onlyOwner {
        require(price_token != address(0), "Price token not set");
        require(tokens.length == new_prices.length, "Array length mismatch");
        for (uint256 i = 0; i < tokens.length; i++) {
            prices[tokens[i]] = new_prices[i];
        }
    }

    function get_price(address token) public view returns (uint256) {
        return prices[token];
    }

    function place_limit_buy_order(address token_address, uint256 amount, uint256 limit_price) public {
        require(price_token != address(0), "Price token not set");
        require(token_address != price_token, "Token cannot be the same as price token");
        require(amount > 0, "Amount must be greater than 0");
        require(limit_price > 0, "Limit price must be greater than 0");
        require(token_address != address(0), "Invalid token address");
        
        IERC20 token = IERC20(token_address);
        IERC20 price_token_instance = IERC20(price_token);
        
        uint256 current_price = prices[token_address];

        // check that price is set and is less than or equal to the limit price
        require(current_price > 0, "Current price not set");
        require(current_price <= limit_price, "Current price exceeds limit price");

        uint256 total_price = amount * current_price;
        uint256 fee_amount = total_price * fee / 1000;
        uint256 final_price = total_price + fee_amount;

        require(price_token_instance.transferFrom(msg.sender, address(this), final_price), "Transfer failed, check balance and allowance");
        require(token.transfer(msg.sender, amount), "Transfer failed, check contract balance");

        // TODO: update price properly
        uint256 new_price = (current_price * 1000 + limit_price) / 1001;
        prices[token_address] = new_price;

        emit BuyOrder(msg.sender, token_address, amount, current_price, new_price);
    }

    function place_limit_sell_order(address token_address, uint256 amount, uint256 limit_price) public {
        require(price_token != address(0), "Price token not set");
        require(token_address != price_token, "Token cannot be the same as price token");
        require(amount > 0, "Amount must be greater than 0");
        require(limit_price > 0, "Limit price must be greater than 0");
        require(token_address != address(0), "Invalid token address");
        
        IERC20 token = IERC20(token_address);
        IERC20 price_token_instance = IERC20(price_token);
        
        uint256 current_price = prices[token_address];

        // check that price is set and is greater than or equal to the limit price
        require(current_price > 0, "Current price not set");
        require(current_price >= limit_price, "Current price is less than limit price");

        uint256 total_price = amount * current_price;
        uint256 fee_amount = total_price * fee / 1000;
        uint256 final_price = total_price - fee_amount;

        require(token.transferFrom(msg.sender, address(this), amount), "Transfer failed, check balance and allowance");
        require(price_token_instance.transfer(msg.sender, final_price), "Transfer failed, check contract balance");
        // TODO: update price properly
        uint256 new_price = (current_price * 1000 + limit_price) / 1001;
        prices[token_address] = new_price;
        emit SellOrder(msg.sender, token_address, amount, current_price, new_price);
    }
}