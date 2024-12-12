// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract OrderBook is Ownable {
    // User => Token A => Token B => amount provided
    mapping(address => mapping(address => mapping(address => uint256))) public individual_liquidity;
    // Token A => Token B => total balance including fees
    mapping(address => mapping(address => uint256)) public total_pool_balance;
    uint256 public fee = 1; // 0.1% = 1/1000

    event Deposit(
        address indexed user,
        address indexed tokenA,
        address indexed tokenB,
        uint256 amountA,
        uint256 amountB
    );
    event Withdrawal(
        address indexed user,
        address indexed tokenA,
        address indexed tokenB,
        uint256 amountA,
        uint256 amountB
    );
    event Swap(
        address indexed user,
        address indexed sourceToken,
        address indexed targetToken,
        uint256 sourceAmount,
        uint256 targetAmount
    );
    event FeeUpdate(uint256 oldFee, uint256 newFee);

    // Pass msg.sender to the Ownable constructor
    constructor() Ownable(msg.sender) {}

    function set_fee(uint256 new_fee) public onlyOwner {
        require(new_fee <= 50, "Fee cannot exceed 5%");
        emit FeeUpdate(fee, new_fee);
        fee = new_fee;
    }

    function get_price(address sell_token_address, address buy_token_address) public view returns (uint256) {
        uint256 sell_balance = total_pool_balance[sell_token_address][buy_token_address];
        uint256 buy_balance = total_pool_balance[buy_token_address][sell_token_address];
        require(sell_balance > 0 && buy_balance > 0, "Insufficient liquidity");

        // Price = (buy_balance * 1e18) / sell_balance
        return (buy_balance * 1e18) / sell_balance;
    }

    function deposit(
        address tokenA_address,
        address tokenB_address,
        uint256 amountA,
        uint256 amountB
    ) public {
        require(amountA > 0 && amountB > 0, "Amounts must be greater than 0");
        require(tokenA_address != address(0) && tokenB_address != address(0), "Invalid token address");
        require(tokenA_address != tokenB_address, "Tokens must be different");

        IERC20 tokenA = IERC20(tokenA_address);
        IERC20 tokenB = IERC20(tokenB_address);

        // Transfer tokens from user to contract
        require(tokenA.transferFrom(msg.sender, address(this), amountA), "Token A transfer failed");
        require(tokenB.transferFrom(msg.sender, address(this), amountB), "Token B transfer failed");

        // Update individual liquidity
        individual_liquidity[msg.sender][tokenA_address][tokenB_address] += amountA;
        individual_liquidity[msg.sender][tokenB_address][tokenA_address] += amountB;

        // Update total pool balances
        total_pool_balance[tokenA_address][tokenB_address] += amountA;
        total_pool_balance[tokenB_address][tokenA_address] += amountB;

        emit Deposit(msg.sender, tokenA_address, tokenB_address, amountA, amountB);
    }

    function withdraw(address tokenA_address, address tokenB_address) public {
        uint256 amountA = individual_liquidity[msg.sender][tokenA_address][tokenB_address];
        uint256 amountB = individual_liquidity[msg.sender][tokenB_address][tokenA_address];
        require(amountA > 0 && amountB > 0, "No liquidity to withdraw");

        // Reset individual liquidity
        individual_liquidity[msg.sender][tokenA_address][tokenB_address] = 0;
        individual_liquidity[msg.sender][tokenB_address][tokenA_address] = 0;

        // Update total pool balances
        total_pool_balance[tokenA_address][tokenB_address] -= amountA;
        total_pool_balance[tokenB_address][tokenA_address] -= amountB;

        // Transfer tokens back to user
        IERC20 tokenA = IERC20(tokenA_address);
        IERC20 tokenB = IERC20(tokenB_address);

        require(tokenA.transfer(msg.sender, amountA), "Token A transfer failed");
        require(tokenB.transfer(msg.sender, amountB), "Token B transfer failed");

        emit Withdrawal(msg.sender, tokenA_address, tokenB_address, amountA, amountB);
    }

    function swap(
        address source_token_address,
        uint256 source_token_amount,
        address target_token_address
    ) public {
        require(source_token_amount > 0, "Amount must be greater than 0");
        require(source_token_address != target_token_address, "Cannot swap same token");

        uint256 source_reserve = total_pool_balance[source_token_address][target_token_address];
        uint256 target_reserve = total_pool_balance[target_token_address][source_token_address];
        require(source_reserve > 0 && target_reserve > 0, "Insufficient liquidity");

        // Apply fee
        uint256 amount_in_with_fee = source_token_amount * (1000 - fee);
        uint256 numerator = amount_in_with_fee * target_reserve;
        uint256 denominator = (source_reserve * 1000) + amount_in_with_fee;
        uint256 target_amount = numerator / denominator;

        require(target_amount > 0, "Insufficient output amount");
        require(target_amount <= target_reserve, "Insufficient liquidity");

        // Transfer source tokens from user to contract
        IERC20 source_token = IERC20(source_token_address);
        require(
            source_token.transferFrom(msg.sender, address(this), source_token_amount),
            "Source transfer failed"
        );

        // Transfer target tokens to user
        IERC20 target_token = IERC20(target_token_address);
        require(
            target_token.transfer(msg.sender, target_amount),
            "Target transfer failed"
        );

        // Update reserves
        total_pool_balance[source_token_address][target_token_address] += source_token_amount;
        total_pool_balance[target_token_address][source_token_address] -= target_amount;

        emit Swap(msg.sender, source_token_address, target_token_address, source_token_amount, target_amount);
    }
}
