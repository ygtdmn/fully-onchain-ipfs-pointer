// SPDX-License-Identifier: MIT
pragma solidity >=0.8.25 <0.9.0;

import { FullyOnChainIPFSPointer } from "../src/FullyOnChainIPFSPointer.sol";
import { FullyOnChainIPFSPointerRenderer } from "../src/FullyOnChainIPFSPointerRenderer.sol";
import { BaseScript } from "./Base.s.sol";
import { FileStore } from "../src/interfaces/EthFS.sol";

contract Deploy is BaseScript {
    function run()
        public
        broadcast
        returns (FullyOnChainIPFSPointer fullyOnChainIpfsPointer, FullyOnChainIPFSPointerRenderer renderer)
    {
        string memory metadata =
            unicode"\"name\": \"Fully On-Chain IPFS Pointer\",\"description\": \"An IPFS pointer (CID) that is generated fully on-chain with on-chain content, hash function, and renderer.\"";
        renderer = new FullyOnChainIPFSPointerRenderer(metadata, FileStore(0xFe1411d6864592549AdE050215482e4385dFa0FB));
        fullyOnChainIpfsPointer = new FullyOnChainIPFSPointer(address(renderer));
    }
}
