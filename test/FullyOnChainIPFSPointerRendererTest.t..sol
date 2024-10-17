// SPDX-License-Identifier: MIT
pragma solidity >=0.8.25 <0.9.0;

import { Test } from "forge-std/src/Test.sol";
import { console2 } from "forge-std/src/console2.sol";

import { FullyOnChainIPFSPointerRenderer } from "../src/FullyOnChainIPFSPointerRenderer.sol";
import { IPFSLibrary } from "../src/IPFSLibrary.sol";
import "../src/interfaces/EthFS.sol";

contract FullyOnChainIPFSPointerRendererTest is Test {
    function testFork_Example() external {
        // Silently pass this test if there is no API key.
        string memory alchemyApiKey = vm.envOr("API_KEY_ALCHEMY", string(""));
        if (bytes(alchemyApiKey).length == 0) {
            return;
        }

        // Otherwise, run the test against the mainnet fork.
        vm.createSelectFork({ urlOrAlias: "mainnet" });

        FullyOnChainIPFSPointerRenderer renderer =
            new FullyOnChainIPFSPointerRenderer("", FileStore(0xFe1411d6864592549AdE050215482e4385dFa0FB));
        string memory image = renderer.renderImage();
        // string memory image = string(renderer.getIpfsContent());
        console2.log(image);
    }
}
