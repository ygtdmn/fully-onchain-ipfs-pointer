// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title IPFSLibrary
/// @notice A library for generating CIDv1 (Content Identifier version 1) for IPFS (InterPlanetary File System)
/// @dev This library provides functions to generate CIDv1 and perform base32 encoding
library IPFSLibrary {
    // Constants for CID and codec prefixes
    bytes constant CIDV1_PREFIX = hex"01";
    bytes constant RAW_CODEC = hex"55";
    bytes constant SHA256_CODEC = hex"12";
    uint8 constant DIGEST_LENGTH = 32;

    /// @notice Generates a CIDv1 for the given content
    /// @param content The bytes of the content to generate a CID for
    /// @return A string representation of the CIDv1 in base32 encoding
    function generateCIDv1(bytes memory content) public pure returns (string memory) {
        // Generate SHA-256 hash of the content
        bytes32 contentHash = sha256(content);

        // Construct multihash (hash function code + digest length + digest)
        bytes memory multihash = abi.encodePacked(SHA256_CODEC, bytes1(DIGEST_LENGTH), contentHash);

        // Construct CIDv1 (version + content type + multihash)
        bytes memory cid = abi.encodePacked(CIDV1_PREFIX, RAW_CODEC, multihash);

        // Encode the CID to Base32 and return
        return base32Encode(cid);
    }

    /// @notice Encodes the input bytes to base32
    /// @param input The bytes to encode
    /// @return A string representation of the input encoded in base32
    function base32Encode(bytes memory input) public pure returns (string memory) {
        bytes memory alphabet = "abcdefghijklmnopqrstuvwxyz234567";
        uint256 bitLen = input.length * 8;
        uint256 resultLen = (bitLen + 4) / 5;
        bytes memory result = new bytes(resultLen);

        uint256 bits;
        uint256 buffer;
        uint256 bufferLen;

        for (uint256 i = 0; i < resultLen; i++) {
            // Fill the buffer with 8 bits at a time until we have at least 5 bits
            while (bufferLen < 5) {
                if (bits / 8 < input.length) {
                    buffer = (buffer << 8) | uint8(input[bits / 8]);
                    bufferLen += 8;
                } else {
                    buffer <<= 8;
                    bufferLen += 8;
                }
                bits += 8;
            }

            // Extract 5 bits from the buffer and add the corresponding character to the result
            result[i] = alphabet[uint8(buffer >> (bufferLen - 5))];
            bufferLen -= 5;
            buffer &= (1 << bufferLen) - 1;
        }

        // Prepend 'b' to the result as per CIDv1 base32 encoding specification
        return string(abi.encodePacked("b", result));
    }
}
