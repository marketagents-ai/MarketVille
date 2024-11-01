# MarketVille
an agent based market simulation


## EMV Testnet
* install nodejs
* `poetry install`
* `poetry shell`
* `./run_node.sh ganache_config.json start`
* `./run_node.sh ganache_config.json start`
* ganache is now running
* `python init_sim.py` launches test tokens and mints to addresses


## Other `run_node.sh` commands:
* start: starts node
* stop: stops node
* restart: restarts node
* status: shows node status
* regen-keys: regenerates the wallets in the config


## EVM Agent Interface
* all functions are available in `agent_evm_interface/agent_evm_interface.py`
* keep `@external` and `@init` tags, and function description in that format if you want to remain compatible with the function context system
* run `agent_evm_interface_example.py` to test node




NOTE: early version, probably has bugs, especially during install