import logging
from typing import List
from colorama import Fore, Back, Style
import json
import pyfiglet

def print_ascii_art():
    ascii_art = pyfiglet.figlet_format("MARKET AGENTS", font="slant")
    print(Fore.CYAN + ascii_art + Style.RESET_ALL)

import logging

def setup_logger(name: str = "MarketSimulation", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    
    # Add a stream handler to log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Prevent propagation to avoid double logging
    logger.propagate = False
    
    return logger

# Create a single, centralized logger instance
orchestartion_logger = setup_logger()

def log_section(logger: logging.Logger, message: str):
    border = "======================================"
    logger.info(f"{Fore.MAGENTA}{border}{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}🔥 {message.upper()} 🔥{Style.RESET_ALL}")
    logger.info(f"{Fore.MAGENTA}{border}{Style.RESET_ALL}")

def log_round(logger: logging.Logger, round_num: int):
    logger.info(f"{Fore.GREEN}🔔 ROUND {round_num:02d} BEGINS 🔔{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}🎲 Let the market dynamics unfold! 🎲{Style.RESET_ALL}")

def log_agent_init(logger: logging.Logger, agent_id: int, is_buyer: bool, persona):
    agent_type = "🛒 Buyer" if is_buyer else "💼 Seller"
    trader_type = " | ".join(persona.trader_type)
    
    logger.info(f"{Fore.BLUE}🤖 Agent {agent_id:02d} | {agent_type} | {trader_type} | Initialized{Style.RESET_ALL}")

def log_persona(logger: logging.Logger, agent_index: int, persona: str):
    logger.info(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
    logger.info(f"{Fore.MAGENTA}Current Agent:{Style.RESET_ALL} Agent {agent_index} with persona:{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}{persona}{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}======================================{Style.RESET_ALL}")

def log_environment_setup(logger: logging.Logger, env_name: str):
    logger.info(f"{Fore.GREEN}🏛️ Entering the {env_name.upper()} ECOSYSTEM 🏛️{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}📈 Where market forces shape destinies 📉{Style.RESET_ALL}")

def log_completion(logger: logging.Logger, message: str):
    logger.info(f"{Fore.GREEN}🎉 {message} 🚀{Style.RESET_ALL}")

def log_skipped(logger: logging.Logger, message: str):
    logger.info(f"{Fore.RED}⏭️ {message} (Unexpected market shift!){Style.RESET_ALL}")

def log_running(logger: logging.Logger, env_name: str):
    logger.info(f"{Fore.GREEN}🏁 The {env_name} market is now ACTIVE 🏁{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}💥 Prepare for economic disruption! 💥{Style.RESET_ALL}")

def log_perception(logger: logging.Logger, agent_id: int, perception: str):
    logger.info(f"{Back.BLUE}{Fore.WHITE}👁️ Agent {agent_id} perceives: {Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}{perception}{Style.RESET_ALL}")

def log_raw_action(logger: logging.Logger, agent_id: int, action: dict):
    logger.info(f"{Back.YELLOW}{Fore.BLACK}🔧 Agent {agent_id} executes: {Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}{action}{Style.RESET_ALL}")

def log_action(logger: logging.Logger, agent_id: int, action: str):
    if "Bid" in action:
        emoji = "💰"
        color = Fore.GREEN
    elif "Ask" in action:
        emoji = "💵"
        color = Fore.YELLOW
    elif "reflects" in action.lower():
        emoji = "💭"
        color = Fore.MAGENTA
    elif "perceives" in action.lower():
        emoji = "👁️"
        color = Fore.CYAN
    else:
        emoji = "🔧"
        color = Fore.WHITE
    logger.info(f"{color}{emoji} Agent {agent_id:02d} executes: {action}{Style.RESET_ALL}")

def log_market_update(logger: logging.Logger, update: str):
    logger.info(f"{Back.CYAN}{Fore.BLACK}📢 MARKET INSIGHT:{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}{update}{Style.RESET_ALL}")

def log_reflection(logger: logging.Logger, agent_id: int, reflection: str):
    logger.info(f"{Back.MAGENTA}{Fore.WHITE}💭 Agent {agent_id:02d} reflects:{Style.RESET_ALL}")
    logger.info(f"{Fore.MAGENTA}'{reflection}'{Style.RESET_ALL}")

def log_trade(logger: logging.Logger, buyer_id: int, seller_id: int, item: str, price: float):
    logger.info(f"{Back.GREEN}{Fore.BLACK}💰 TRANSACTION ALERT 💰{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}🤝 Agent {buyer_id:02d} acquires {item} from Agent {seller_id:02d} at ${price:.2f}{Style.RESET_ALL}")

def log_leaderboard(logger: logging.Logger, rankings: list):
    logger.info(f"{Back.YELLOW}{Fore.BLACK}🏆 PERFORMANCE RANKINGS 🏆{Style.RESET_ALL}")
    for rank, (agent_id, score) in enumerate(rankings, 1):
        indicator = ["🥇", "🥈", "🥉"][rank-1] if rank <= 3 else "  "
        if rank == 1:
            color = Fore.YELLOW
        elif rank == 2:
            color = Fore.WHITE
        elif rank == 3:
            color = Fore.RED
        else:
            color = Fore.BLUE
        logger.info(f"{color}{indicator} #{rank}: Agent {agent_id:02d} - ${score:.2f}{Style.RESET_ALL}")

# Add these new logging functions for group chat
def log_topic_proposal(logger: logging.Logger, cohort_id: str, proposer_id: int, topic: str):
    logger.info(f"{Back.BLUE}{Fore.WHITE}📢 TOPIC PROPOSAL - {cohort_id.upper()} 📢{Style.RESET_ALL}")
    logger.info(f"{Back.WHITE}{Fore.BLUE}🎯 Proposer: Agent {proposer_id:02d}{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}💬 Topic: {topic}{Style.RESET_ALL}")
    logger.info(f"{Fore.BLUE}{'─' * 50}{Style.RESET_ALL}")

def log_group_message(logger: logging.Logger, cohort_id: str, agent_id: int, message: str, sub_round: int):
    # Rotate colors for different agents to make it easier to follow conversations
    # Use a list of colors that can handle any number of agents
    agent_colors = [
        (Fore.GREEN, Back.BLACK),
        (Fore.YELLOW, Back.BLACK), 
        (Fore.BLUE, Back.BLACK),
        (Fore.MAGENTA, Back.BLACK),
        (Fore.CYAN, Back.BLACK),
        (Fore.RED, Back.BLACK),
        (Fore.WHITE, Back.BLACK)
    ]
    # Use modulo to cycle through colors for any number of agents
    color_pair = agent_colors[agent_id % len(agent_colors)]
    
    logger.info(f"{Back.WHITE}{Fore.BLACK}💬 {cohort_id.upper()} - Round {sub_round}{Style.RESET_ALL}")
    logger.info(f"{color_pair[1]}{color_pair[0]}🤖 Agent {agent_id:02d} says:{Style.RESET_ALL}")
    logger.info(f"{color_pair[0]}{message}{Style.RESET_ALL}")
    logger.info(f"{Fore.WHITE}{'─' * 50}{Style.RESET_ALL}")

def log_cohort_formation(logger: logging.Logger, cohort_id: str, agent_indices: List[int]):
    logger.info(f"{Back.GREEN}{Fore.BLACK}🎯 COHORT FORMATION 🎯{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}📋 {cohort_id.upper()}: Agents {agent_indices}{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}{'─' * 50}{Style.RESET_ALL}")

def log_sub_round_start(logger: logging.Logger, cohort_id: str, sub_round: int):
    logger.info(f"{Back.YELLOW}{Fore.BLACK}🔄 SUB-ROUND {sub_round} - {cohort_id.upper()} 🔄{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}{'─' * 50}{Style.RESET_ALL}")

def log_group_chat_summary(logger: logging.Logger, cohort_id: str, messages_count: int, topic: str):
    logger.info(f"{Back.MAGENTA}{Fore.WHITE}📊 GROUP CHAT SUMMARY - {cohort_id.upper()} 📊{Style.RESET_ALL}")
    logger.info(f"{Fore.MAGENTA}📝 Total Messages: {messages_count}{Style.RESET_ALL}")
    logger.info(f"{Fore.MAGENTA}💭 Topic Discussed: {topic}{Style.RESET_ALL}")
    logger.info(f"{Fore.MAGENTA}{'═' * 50}{Style.RESET_ALL}")