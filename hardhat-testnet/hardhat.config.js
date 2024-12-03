require("@nomicfoundation/hardhat-toolbox");
require("@openzeppelin/hardhat-upgrades");
module.exports = {
 solidity: {
 version: "0.8.20",
 settings: {
 optimizer: {
 enabled: true,
 runs: 200
 }
 }
 },
 networks: {
 hardhat: {
 allowUnlimitedContractSize: true,
 accounts: {
   mnemonic: "candy maple cake sugar pudding cream honey rich smooth crumble sweet treat"
 }
 }
 },
 paths: {
 sources: "./contracts",
 tests: "./test",
 cache: "./cache",
 artifacts: "./artifacts"
 }
};
