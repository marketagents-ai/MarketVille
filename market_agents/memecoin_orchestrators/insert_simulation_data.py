# insert_simulation_data.py

from decimal import Decimal
import psycopg2
import psycopg2.extras
import os
import uuid
from typing import List, Dict, Any
from psycopg2.extensions import register_adapter, AsIs
import json
import logging
from datetime import datetime
from market_agents.memecoin_orchestrators.crypto_models import OrderType
from market_agents.memecoin_orchestrators.setup_database import create_database, create_tables

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Type {type(obj)} not serializable")

def serialize_memory_data(memory_data):
    if isinstance(memory_data, dict):
        return {k: serialize_memory_data(v) for k, v in memory_data.items()}
    elif isinstance(memory_data, list):
        return [serialize_memory_data(v) for v in memory_data]
    elif isinstance(memory_data, datetime):
        return memory_data.isoformat()
    elif hasattr(memory_data, 'model_dump'):
        return serialize_memory_data(memory_data.model_dump())
    elif hasattr(memory_data, '__dict__'):
        return serialize_memory_data(vars(memory_data))
    elif isinstance(memory_data, (str, int, float, bool, type(None))):
        return memory_data
    else:
        return str(memory_data)

def validate_json(data):
    """Validate if data is JSON or can be parsed as JSON."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return None

class SimulationDataInserter:
    def __init__(self, db_params):
        create_database(db_params)
        self.conn = psycopg2.connect(**db_params)
        self.cursor = self.conn.cursor()

        create_tables(db_params)

    def __del__(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def insert_agents(self, agents_data):
        query = """
            INSERT INTO agents (id, role, is_llm, active, current_round, max_iter, llm_config, config)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                role = EXCLUDED.role,
                is_llm = EXCLUDED.is_llm,
                active = EXCLUDED.active,
                current_round = EXCLUDED.current_round,
                max_iter = EXCLUDED.max_iter,
                llm_config = EXCLUDED.llm_config,
                config = EXCLUDED.config
            RETURNING id
        """
        agent_id_map = {}
        for agent in agents_data:
            try:
                agent_id = uuid.UUID(str(agent['id'])) if isinstance(agent['id'], (str, int)) else agent['id']

                # Set default values for required fields
                is_llm = agent.get('is_llm', True)
                max_iter = agent.get('max_iter', 100)
                llm_config = agent.get('llm_config', {})

                with self.conn.cursor() as cur:
                    cur.execute(query, (
                        agent_id,
                        agent['role'],
                        is_llm,
                        agent.get('active', True),
                        agent.get('current_round', 1),
                        max_iter,
                        json.dumps(llm_config),
                        json.dumps(agent.get('config', {}))
                    ))
                    inserted_id = cur.fetchone()
                    if inserted_id:
                        agent_id_map[str(agent['id'])] = inserted_id[0]
                        logging.info(f"Successfully inserted/updated agent: {agent_id}")
                    else:
                        logging.warning(f"No id returned for agent: {agent['id']}")
                self.conn.commit()
            except Exception as e:
                logging.error(f"Error processing agent {agent.get('id')}: {str(e)}")
                logging.error(f"Agent data: {agent}")
                self.conn.rollback()
                continue

        if not agent_id_map:
            logging.error("No agents were successfully inserted/updated")
        else:
            logging.info(f"Successfully processed {len(agent_id_map)} agents")

        return agent_id_map

    def check_tables_exist(self):
        cursor = self.conn.cursor()
        # Check if the 'agents' table exists
        cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name=%s)", ('agents',))
        exists = cursor.fetchone()[0]
        cursor.close()
        return exists

    def insert_agent_memories(self, memories: List[Dict[str, Any]]):
        query = """
        INSERT INTO agent_memories (agent_id, step_id, memory_data)
        VALUES (%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for memory in memories:
                    agent_id = uuid.UUID(str(memory['agent_id']))
                    cur.execute(query, (
                        agent_id,
                        memory['step_id'],
                        json.dumps(memory['memory_data'], default=str)
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(memories)} agent memories into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting agent memories: {e}")
            raise

    def insert_allocations(self, allocations: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO allocations (agent_id, cash, initial_cash, positions, initial_positions)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for allocation in allocations:
                    agent_id = agent_id_map.get(str(allocation['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {allocation['agent_id']}")
                        continue

                    cur.execute(query, (
                        agent_id,
                        allocation['cash'],
                        allocation['initial_cash'],
                        json.dumps(allocation['positions'], default=json_serial),
                        json.dumps(allocation['initial_positions'], default=json_serial)
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(allocations)} allocations into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting allocations: {str(e)}")
            raise

    def insert_agent_positions(self, positions_data: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO agent_positions (agent_id, round, cash, positions)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (agent_id, round) DO UPDATE SET
            cash = EXCLUDED.cash,
            positions = EXCLUDED.positions
        """
        try:
            with self.conn.cursor() as cur:
                for position in positions_data:
                    agent_id = agent_id_map.get(str(position['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {position['agent_id']}")
                        continue

                    cur.execute(query, (
                        agent_id,
                        position['round'],
                        position['cash'],
                        json.dumps(position['positions'], default=json_serial)
                    ))
            self.conn.commit()
            logging.info(f"Inserted or updated {len(positions_data)} agent positions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting agent positions: {str(e)}")
            raise


    def insert_orders(self, orders: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO orders (agent_id, order_type, quantity, price)
        VALUES (%s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for order in orders:
                    agent_id = agent_id_map.get(str(order['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {order['agent_id']}")
                        continue
                    # Convert quantity and price to Decimal if they are not None
                    quantity = Decimal(str(order['quantity'])) if order['quantity'] is not None else None
                    price = Decimal(str(order['price'])) if order['price'] is not None else None

                    cur.execute(query, (
                        agent_id,
                        order['order_type'],
                        quantity,
                        price
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(orders)} orders into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting orders: {str(e)}")
            raise

    def insert_trades(self, trades_data: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO trades (buyer_id, seller_id, quantity, price, round)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for trade in trades_data:
                    buyer_id = agent_id_map.get(str(trade['buyer_id']))
                    seller_id = agent_id_map.get(str(trade['seller_id']))

                    if buyer_id is None or seller_id is None:
                        logging.error(f"No matching UUID found for buyer_id: {trade['buyer_id']} or seller_id: {trade['seller_id']}")
                        continue

                    cur.execute(query, (
                        buyer_id,
                        seller_id,
                        trade['quantity'],
                        trade['price'],
                        trade['round']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(trades_data)} trades into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting trades: {str(e)}")
            raise

    def insert_interactions(self, interactions: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO interactions (agent_id, round, task, response)
        VALUES (%s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for interaction in interactions:
                    agent_id = agent_id_map.get(str(interaction['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {interaction['agent_id']}")
                        continue
                    cur.execute(query, (
                        agent_id,
                        interaction['round'],
                        interaction['task'],
                        interaction['response']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(interactions)} interactions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting interactions: {str(e)}")
            raise

    def insert_observations(self, observations: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO observations (memory_id, environment_name, observation)
        VALUES (%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for observation in observations:
                    memory_id = agent_id_map.get(str(observation['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {observation['memory_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        observation['environment_name'],
                        psycopg2.extras.Json(observation['observation'])
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(observations)} observations into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting observations: {str(e)}")
            raise

    def insert_reflections(self, reflections: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO reflections (memory_id, environment_name, reflection, self_reward, environment_reward, total_reward, strategy_update)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for reflection in reflections:
                    memory_id = agent_id_map.get(str(reflection['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {reflection['memory_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        reflection['environment_name'],
                        reflection['reflection'],
                        reflection['self_reward'],
                        reflection['environment_reward'],
                        reflection['total_reward'],
                        reflection['strategy_update']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(reflections)} reflections into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting reflections: {str(e)}")
            raise

    def insert_perceptions(self, perceptions: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO perceptions (memory_id, environment_name, monologue, strategy, confidence)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for perception in perceptions:
                    memory_id = agent_id_map.get(str(perception['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {perception['memory_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        perception['environment_name'],
                        perception['monologue'],
                        perception['strategy'],
                        perception['confidence']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(perceptions)} perceptions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting perceptions: {str(e)}")
            raise

    def insert_actions(self, actions: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO actions (
            memory_id,
            environment_name,
            action
        )
        VALUES (%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for action in actions:
                    memory_id = agent_id_map.get(str(action['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {action['memory_id']}")
                        continue
                    environment_name = action['environment_name']
                    action_data = action['action']

                    cur.execute(query, (
                        memory_id,
                        environment_name,
                        json.dumps(action_data, default=str)
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(actions)} actions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting actions: {str(e)}")
            raise

    def insert_ai_requests(self, ai_requests):
        requests_data = []
        for request in ai_requests:
            start_time = request.start_time
            end_time = request.end_time
            if isinstance(start_time, (float, int)):
                start_time = datetime.fromtimestamp(start_time)
            if isinstance(end_time, (float, int)):
                end_time = datetime.fromtimestamp(end_time)

            total_time = (end_time - start_time).total_seconds()

            # Extract system message
            system_message = next((msg['content'] for msg in request.completion_kwargs.get('messages', []) if msg['role'] == 'system'), None)

            requests_data.append({
                'prompt_context_id': str(request.source_id),
                'start_time': start_time,
                'end_time': end_time,
                'total_time': total_time,
                'model': request.completion_kwargs.get('model', ''),
                'max_tokens': request.completion_kwargs.get('max_tokens', None),
                'temperature': request.completion_kwargs.get('temperature', None),
                'messages': request.completion_kwargs.get('messages', []),
                'system': system_message,
                'tools': request.completion_kwargs.get('tools', []),
                'tool_choice': request.completion_kwargs.get('tool_choice', {}),
                'raw_response': request.raw_result,
                'completion_tokens': request.usage.completion_tokens if request.usage else None,
                'prompt_tokens': request.usage.prompt_tokens if request.usage else None,
                'total_tokens': request.usage.total_tokens if request.usage else None
            })

        if requests_data:
            try:
                self._insert_ai_requests_to_db(requests_data)
            except Exception as e:
                logging.error(f"Error inserting AI requests: {e}")

    def _insert_ai_requests_to_db(self, requests_data):
        query = """
        INSERT INTO requests
        (prompt_context_id, start_time, end_time, total_time, model,
        max_tokens, temperature, messages, system, tools, tool_choice,
        raw_response, completion_tokens, prompt_tokens, total_tokens)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for request in requests_data:
                    cur.execute(query, (
                        request['prompt_context_id'],
                        request['start_time'],
                        request['end_time'],
                        request['total_time'],
                        request['model'],
                        request['max_tokens'],
                        request['temperature'],
                        json.dumps(request['messages']),
                        request['system'],
                        json.dumps(request.get('tools', [])),
                        json.dumps(request.get('tool_choice', {})),
                        json.dumps(request['raw_response']),
                        request['completion_tokens'],
                        request['prompt_tokens'],
                        request['total_tokens']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(requests_data)} AI requests")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting AI requests: {str(e)}")
            raise

    def insert_groupchat_messages(self, messages: List[Dict[str, Any]], round_num: int, agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO groupchat (message_id, agent_id, round, sub_round, cohort_id, content, timestamp, topic)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for message in messages:
                    # Get the mapped agent ID from the database
                    agent_id = agent_id_map.get(str(message['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {message['agent_id']}")
                        continue

                    cur.execute(query, (
                        message['message_id'],
                        agent_id,  # Use the mapped agent_id
                        round_num,
                        message['sub_round'],
                        message['cohort_id'],
                        message['content'],
                        message['timestamp'],
                        message.get('topic')
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(messages)} group chat messages")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting group chat messages: {str(e)}")
            raise

    def insert_round_data(
        self,
        round_num: int,
        agents: List[Any],
        environments: Dict[str, Any],
        orchestrator_config: Any,
        trackers: Dict[str, Any],
    ):
        """
        Insert simulation data for a specific round.

        Args:
            round_num (int): The current round number
            agents (List[Any]): List of agent objects
            environments (Dict[str, Any]): Dictionary of environment objects
            orchestrator_config (Any): Orchestrator configuration object
            trackers (Dict[str, Any]): Dictionary of tracker objects for each environment
        """
        try:
            # Insert agent data
            agents_data = [
                {
                    'id': str(agent.id),
                    'role': agent.role,
                    'is_llm': agent.use_llm,
                    'active': True,
                    'current_round': round_num,
                    'max_iter': orchestrator_config.max_rounds,
                    'llm_config': agent.llm_config if isinstance(agent.llm_config, dict) else agent.llm_config.dict(),
                    'config': {
                        'index': getattr(agent, 'index', 0),
                        'name': getattr(agent, 'name', 'Unknown'),
                        'type': agent.__class__.__name__,
                        'ethereum_address': getattr(agent.economic_agent, 'ethereum_address', None),
                    }
                }
                for agent in agents
            ]
            agent_id_map = self.insert_agents(agents_data)

            # Insert agent memories
            memories_data = [
                {
                    'agent_id': str(agent.id),
                    'step_id': round_num,
                    'memory_data': serialize_memory_data(agent.memory[-1] if agent.memory else {})
                }
                for agent in agents
            ]
            self.insert_agent_memories(memories_data)

            # Insert allocations and positions
            allocations_data = []
            positions_data = []
            for agent in agents:
                economic_agent = agent.economic_agent if hasattr(agent, 'economic_agent') else agent

                # Get balances
                cash_balance = getattr(economic_agent.endowment.current_portfolio, 'cash', 0)
                initial_cash = getattr(economic_agent.endowment.initial_portfolio, 'cash', cash_balance)

                # Positions
                positions = {}
                initial_positions = {}
                for coin in economic_agent.endowment.current_portfolio.coins:
                    positions[coin.symbol] = sum(pos.quantity for pos in coin.positions)
                for coin in economic_agent.endowment.initial_portfolio.coins:
                    initial_positions[coin.symbol] = sum(pos.quantity for pos in coin.positions)

                allocation = {
                    'agent_id': str(agent.id),
                    'cash': cash_balance,
                    'initial_cash': initial_cash,
                    'positions': positions,
                    'initial_positions': initial_positions
                }
                allocations_data.append(allocation)

                # Positions data for the round
                positions_data.append({
                    'agent_id': str(agent.id),
                    'round': round_num,
                    'cash': cash_balance,
                    'positions': positions
                })

            if allocations_data:
                self.insert_allocations(allocations_data, agent_id_map)

            if positions_data:
                self.insert_agent_positions(positions_data, agent_id_map)

            # Insert orders and trades for crypto_market
            if 'crypto_market' in environments:
                logging.info("Processing crypto market data")
                crypto_env = environments['crypto_market']
                tracker = trackers.get('crypto_market')
                # Orders data
                orders_data = []
                for agent in agents:
                    for order in agent.economic_agent.pending_orders:
                        orders_data.append({
                            'agent_id': str(agent.id),
                            'order_type': order.order_type.value,
                            'quantity': order.quantity if order.order_type != OrderType.HOLD else None,
                            'price': order.price if order.order_type != OrderType.HOLD else None
                        })
                if orders_data:
                    self.insert_orders(orders_data, agent_id_map)

                # Trades data
                trades_data = []
                if tracker and hasattr(tracker, 'all_trades'):
                    for trade in tracker.all_trades:
                        trades_data.append({
                            'buyer_id': trade.buyer_id,
                            'seller_id': trade.seller_id,
                            'quantity': trade.quantity,
                            'price': trade.price,
                            'round': round_num
                        })
                    if trades_data:
                        self.insert_trades(trades_data, agent_id_map)

            # Insert interactions
            interactions_data = [
                {
                    'agent_id': str(agent.id),
                    'round': round_num,
                    'task': interaction['type'],
                    'response': serialize_memory_data(interaction['content'])
                }
                for agent in agents
                for interaction in agent.interactions
            ]
            if interactions_data:
                self.insert_interactions(interactions_data, agent_id_map)

            # Insert perceptions, actions, observations, reflections
            perceptions_data = []
            actions_data = []
            observations_data = []
            reflections_data = []

            for agent in agents:
                if agent.last_perception is not None:
                    perception = validate_json(agent.last_perception)
                    if perception is not None:
                        perceptions_data.append({
                            'memory_id': str(agent.id),
                            'environment_name': 'crypto_market',  # Set environment name appropriately
                            'monologue': str(perception.get('monologue', '')),
                            'strategy': str(perception.get('strategy', '')),
                            'confidence': perception.get('confidence', 0)
                        })
                    else:
                        logging.warning(f"Invalid JSON perception data for agent {agent.id}: {agent.last_perception}")

                if hasattr(agent, 'last_action') and agent.last_action:
                    actions_data.append({
                        'memory_id': str(agent.id),
                        'environment_name': 'crypto_market',  # Set environment name appropriately
                        'action': serialize_memory_data(agent.last_action)
                    })

                if agent.last_observation:
                    observations_data.append({
                        'memory_id': str(agent.id),
                        'environment_name': 'crypto_market',  # Set environment name appropriately
                        'observation': serialize_memory_data(agent.last_observation)
                    })

                if agent.memory and agent.memory[-1]['type'] == 'reflection':
                    reflection = agent.memory[-1]
                    reflections_data.append({
                        'memory_id': str(agent.id),
                        'environment_name': 'crypto_market',  # Set environment name appropriately
                        'reflection': reflection.get('content', ''),
                        'self_reward': reflection.get('self_reward', 0),
                        'environment_reward': reflection.get('environment_reward', 0),
                        'total_reward': reflection.get('total_reward', 0),
                        'strategy_update': reflection.get('strategy_update', '')
                    })

            if perceptions_data:
                self.insert_perceptions(perceptions_data, agent_id_map)
            if actions_data:
                self.insert_actions(actions_data, agent_id_map)
            if observations_data:
                self.insert_observations(observations_data, agent_id_map)
            if reflections_data:
                self.insert_reflections(reflections_data, agent_id_map)

            # Insert group chat messages
            if 'group_chat' in environments:
                logging.info("Processing group chat data")
                # Assuming group_chat environment uses cohort_id to group agents
                groupchat_data = []
                for agent in agents:
                    cohort_id = getattr(agent, 'cohort_id', None)
                    if cohort_id:
                        environment = agent.environments['group_chat']
                        messages = environment.mechanism.messages
                        current_topic = environment.mechanism.current_topic

                        for message in messages:
                            if message.round_num == round_num:
                                groupchat_data.append({
                                    'message_id': str(uuid.uuid4()),
                                    'agent_id': str(message.agent_id),
                                    'round': round_num,
                                    'sub_round': message.sub_round_num,
                                    'cohort_id': cohort_id,
                                    'content': message.content,
                                    'timestamp': message.timestamp,
                                    'topic': current_topic
                                })
                if groupchat_data:
                    self.insert_groupchat_messages(groupchat_data, round_num, agent_id_map)

        except Exception as e:
            logging.error(f"Error inserting data for round {round_num}: {str(e)}")
            logging.exception("Exception details:")
            raise

def addapt_uuid(uuid_value):
    return AsIs(f"'{uuid_value}'")

# Register the UUID adapter
register_adapter(uuid.UUID, addapt_uuid)
