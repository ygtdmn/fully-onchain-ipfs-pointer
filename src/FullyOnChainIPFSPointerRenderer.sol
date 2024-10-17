// SPDX-License-Identifier: MIT
pragma solidity >=0.8.25;

import "@openzeppelin/contracts/access/Ownable.sol";
import "solady/utils/Base64.sol";
import "./interfaces/EthFS.sol";
import "./IPFSLibrary.sol";

/// @title FullyOnChainIPFSPointerRenderer
/// @notice Renderer, hash function, and IPFS pointer generator all in one contract.
contract FullyOnChainIPFSPointerRenderer is Ownable {
    string public metadata;
    FileStore public ethFs;

    /// @notice Constructor to initialize the contract
    /// @param _metadata Initial metadata string
    /// @param _ethFs Address of the EthFS contract
    constructor(string memory _metadata, FileStore _ethFs) Ownable() {
        metadata = _metadata;
        ethFs = _ethFs;
    }

    /// @notice Updates the metadata string
    /// @param _metadata New metadata string
    /// @dev Only callable by the contract owner
    function setMetadata(string memory _metadata) external onlyOwner {
        metadata = _metadata;
    }

    /// @notice Updates the EthFS contract address
    /// @param _ethFs New EthFS contract address
    /// @dev Only callable by the contract owner
    function setEthFS(address _ethFs) external onlyOwner {
        ethFs = FileStore(_ethFs);
    }

    /// @notice Generates the full metadata JSON string
    /// @return Full metadata JSON as a string
    function renderMetadata() public view returns (string memory) {
        return string(abi.encodePacked("data:application/json;utf8,{", metadata, ', "image": "', renderImage(), '"}'));
    }

    /// @notice Generates the SVG image with IPFS pointer
    /// @return Base64 encoded SVG image as a data URI
    function renderImage() public view returns (string memory) {
        return string.concat(
            "data:image/svg+xml;base64,",
            Base64.encode(
                bytes(
                    string.concat(
                        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000" preserveAspectRatio="xMidYMid meet" fill="none" viewBox="0 0 1000 1000">',
                        "<defs>",
                        '<style type="text/css">',
                        "@font-face {",
                        "font-family: 'IBM Plex Mono';",
                        "src: url(data:font/woff2;charset=utf-8;base64,",
                        getFont(),
                        ") format('woff2');",
                        "font-weight: normal;",
                        "font-style: normal;",
                        "}",
                        "</style>",
                        "</defs>",
                        '<rect width="100%" height="100%" fill="#0a0a0a" />',
                        '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="\'IBM Plex Mono\'" font-size="22" fill="#fcfcfc">',
                        "ipfs://",
                        getIpfsPointer(),
                        "</text>",
                        "</svg>"
                    )
                )
            )
        );
    }

    /// @notice Retrieves the font data from EthFS
    /// @return Base64 encoded font data
    function getFont() public view returns (string memory) {
        return ethFs.readFile("IBMPlexMono-Regular.woff2");
    }

    /// @notice Generates the content to be stored on IPFS
    /// @return Bytes representation of the SVG content
    function getIpfsContent() public view returns (bytes memory) {
        return bytes(
            string.concat(
                '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000" preserveAspectRatio="xMidYMid meet" fill="none" viewBox="0 0 1000 1000">\n',
                "  <defs>\n",
                '    <style type="text/css">\n',
                "      @font-face {\n",
                "        font-family: 'IBM Plex Mono';\n",
                "        src: url(data:font/woff2;charset=utf-8;base64,",
                getFont(),
                ") format('woff2');\n",
                "        font-weight: normal;\n",
                "        font-style: normal;\n",
                "      }\n",
                "    </style>\n",
                "  </defs>\n",
                '\n  <rect width="100%" height="100%" fill="#0a0a0a" />\n',
                '  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="\'IBM Plex Mono\'" font-size="30"\n',
                '    fill="#fcfcfc">\n',
                "    0xCb337152b6181683010D07e3f00e7508cd348BC7\n",
                "  </text>\n",
                "  <!-- 9b37e899b50acbaf7bc2090a25a797c9 --></svg>"
            )
        );
    }

    /// @notice Generates the IPFS CIDv1 pointer for the content
    /// @return IPFS CIDv1 as a string
    function getIpfsPointer() public view returns (string memory) {
        return IPFSLibrary.generateCIDv1(getIpfsContent());
    }
}
