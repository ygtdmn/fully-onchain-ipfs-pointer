// SPDX-License-Identifier: MIT
pragma solidity >=0.8.25;

import "@manifoldxyz/creator-core-solidity/contracts/core/IERC721CreatorCore.sol";
import "@manifoldxyz/creator-core-solidity/contracts/core/IERC1155CreatorCore.sol";
import "@manifoldxyz/creator-core-solidity/contracts/extensions/ICreatorExtensionTokenURI.sol";
import "@openzeppelin/contracts/utils/introspection/ERC165.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

import "./FullyOnChainIPFSPointerRenderer.sol";

/**
 * @title FullyOnChainIPFSPointer
 * @notice An IPFS pointer (CID) that is generated fully on-chain with on-chain content, hash function, and renderer.
 */
contract FullyOnChainIPFSPointer is ICreatorExtensionTokenURI, ERC165, Ownable {
    FullyOnChainIPFSPointerRenderer public metadataRenderer;
    bool public minted;

    /**
     * @dev Constructor that sets the initial metadata renderer
     * @param _metadataRenderer Address of the metadata renderer contract
     */
    constructor(address _metadataRenderer) Ownable() {
        metadataRenderer = FullyOnChainIPFSPointerRenderer(_metadataRenderer);
    }

    /**
     * @dev Allows the owner to set a new metadata renderer
     * @param _metadataRenderer Address of the new metadata renderer contract
     */
    function setMetadataRenderer(address _metadataRenderer) public onlyOwner {
        metadataRenderer = FullyOnChainIPFSPointerRenderer(_metadataRenderer);
    }

    /**
     * @dev Implements the tokenURI function from ICreatorExtensionTokenURI
     * @return string The metadata URI for the token
     */
    function tokenURI(address, uint256) external view override returns (string memory) {
        return metadataRenderer.renderMetadata();
    }

    /**
     * @dev Mints a new token using the CreatorCore contract
     * @param creatorContractAddress Address of the CreatorCore contract
     */
    function mint(address creatorContractAddress) external onlyOwner {
        require(!minted, "Already minted");
        address[] memory dest = new address[](1);
        uint256[] memory quantities = new uint256[](1);
        string[] memory uris = new string[](1);

        dest[0] = msg.sender;
        quantities[0] = 1;

        IERC1155CreatorCore(creatorContractAddress).mintExtensionNew(dest, quantities, uris);
        minted = true;
    }

    /**
     * @dev Implements supportsInterface to declare supported interfaces
     * @param interfaceId The interface identifier, as specified in ERC-165
     * @return bool True if the contract supports the interface, false otherwise
     */
    function supportsInterface(bytes4 interfaceId) public view virtual override(IERC165, ERC165) returns (bool) {
        return interfaceId == type(ICreatorExtensionTokenURI).interfaceId || super.supportsInterface(interfaceId);
    }
}
