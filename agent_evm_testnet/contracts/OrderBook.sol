// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract OrderBook is Ownable {
    // User => token => amount provided since last withdrawal
    mapping(address => mapping(address => uint256)) public individual_liquidity;
    // Token => total balance including fees
    mapping(address => uint256) public total_pool_balance;
    uint256 public fee = 1; // 0.1% = 1/1000

    event Deposit(address indexed user, address indexed token, uint256 amount);
    event Withdrawal(address indexed user, address indexed token, uint256 amount);
    event Swap(address indexed user, address indexed sourceToken, address indexed targetToken, uint256 sourceAmount, uint256 targetAmount);
    event FeeUpdate(uint256 oldFee, uint256 newFee);

    // add constructor for ownable
    constructor() Ownable(msg.sender) {}

    function set_fee(uint256 new_fee) public onlyOwner {
        require(new_fee <= 50, "Fee cannot exceed 5%");
        emit FeeUpdate(fee, new_fee);
        fee = new_fee;
    }

    function get_price(address sell_token_address, address buy_token_address) public view returns (uint256) {
        uint256 sell_balance = total_pool_balance[sell_token_address];
        uint256 buy_balance = total_pool_balance[buy_token_address];
        require(sell_balance > 0 && buy_balance > 0, "Insufficient liquidity");
        
        // Price = buy_balance / sell_balance * 1e18 (for precision)
        return (buy_balance * 1e18) / sell_balance;
    }

    function deposit(address token_address, uint256 amount) public {
        require(amount > 0, "Amount must be greater than 0");
        require(token_address != address(0), "Invalid token address");
        
        IERC20 token = IERC20(token_address);
        
        // Only try to withdraw if there's existing liquidity
        if (individual_liquidity[msg.sender][token_address] > 0) {
            withdraw(token_address);
        }
        
        require(token.transferFrom(msg.sender, address(this), amount), "Transfer failed");
        
        individual_liquidity[msg.sender][token_address] += amount;
        total_pool_balance[token_address] += amount;
        
        emit Deposit(msg.sender, token_address, amount);
    }

    function withdraw(address token_address) public {
        uint256 amount = individual_liquidity[msg.sender][token_address];
        require(amount > 0, "No liquidity to withdraw");
        
        individual_liquidity[msg.sender][token_address] = 0;
        total_pool_balance[token_address] -= amount;
        
        IERC20 token = IERC20(token_address);
        require(token.transfer(msg.sender, amount), "Transfer failed");
        
        emit Withdrawal(msg.sender, token_address, amount);
    }

    function swap(address source_token_address, uint256 source_token_amount, address target_token_address) public {
        require(source_token_amount > 0, "Amount must be greater than 0");
        require(source_token_address != target_token_address, "Cannot swap same token");
        
        uint256 price = get_price(source_token_address, target_token_address);
        uint256 target_amount = (source_token_amount * price) / 1e18;
        
        // Calculate fee
        uint256 fee_amount = (target_amount * fee) / 1000;
        uint256 final_target_amount = target_amount - fee_amount;
        
        require(total_pool_balance[target_token_address] >= final_target_amount, "Insufficient liquidity");
        
        IERC20 source_token = IERC20(source_token_address);
        IERC20 target_token = IERC20(target_token_address);
        
        // Transfer source tokens from user to contract
        require(source_token.transferFrom(msg.sender, address(this), source_token_amount), "Source transfer failed");
        
        // Transfer target tokens to user
        require(target_token.transfer(msg.sender, final_target_amount), "Target transfer failed");
        
        total_pool_balance[source_token_address] += source_token_amount;
        total_pool_balance[target_token_address] -= final_target_amount;
        
        emit Swap(msg.sender, source_token_address, target_token_address, source_token_amount, final_target_amount);
    }
}