# MarketVille
an agent based market simulation


## EMV Testnet
* `./hardhat-daemon.sh start`
* `poetry shell`
* `poetry install`
* `python testnet_deployer.py`

at this point hardhat testnet should be running with an orderbook with some orders on it.

this will generate 2 files:
* `../.mnemonic` <- mnemonic used to init the network
* `../testnet_data.json` <- contains ABIs and addresses


## EVM Agent Interface
* need access to `.mnemonic` and `testnet_data.json` (leave in the current structure and all is well)


## Test Agent Interface
* `cd agent_evm_interface`
* `python agent_evm_interface.py`


NOTE: early version, probably has bugs, especially during install