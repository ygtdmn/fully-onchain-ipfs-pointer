# Fully On-Chain IPFS Pointer

An IPFS pointer (CID) that is generated fully on-chain with on-chain content, hash function, and renderer.

# Deployed Contracts

| Chain    | Contract                        | Address                                    |
| -------- | ------------------------------- | ------------------------------------------ |
| Ethereum | FullyOnChainIPFSPointer         | 0xA089eCa1a0299570197e00d5Ec9b02a1Caa7EcaB |
| Ethereum | FullyOnChainIPFSPointerRenderer | 0x796cD364C089F048DAC4982E5c93f1D2F26C5d2E |
| Ethereum | IPFSLibrary                     | 0xD8ac6c2834B91A7Af1c7D575100CD5E6546a4FF4 |
| Ethereum | Ephemera                        | 0xCb337152b6181683010D07e3f00e7508cd348BC7 |

# How to Deploy

`bun install`

`forge script script/Deploy.s.sol:Deploy --rpc-url mainnet --broadcast --verify`

# How to Find Vanity CID

`cd vanity-cid`

`go run generate-vanity.go`
