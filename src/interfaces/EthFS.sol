// SPDX-License-Identifier: MIT
pragma solidity ^0.8.22;

// Credit: https://github.com/holic/ethfs

/**
 * @title EthFS File
 * @notice A representation of an onchain file, composed of slices of contract bytecode and utilities to construct the
 * file contents from those slices.
 * @dev For best gas efficiency, it's recommended using `File.read()` as close to the output returned by the contract
 * call as possible. Lots of gas is consumed every time a large data blob is passed between functions.
 */

/**
 * @dev Represents a reference to a slice of bytecode in a contract
 */
struct BytecodeSlice {
    address pointer;
    uint32 start;
    uint32 end;
}

/**
 * @dev Represents a file composed of one or more bytecode slices
 */
struct File {
    // Total length of file contents (sum of all slice sizes). Useful when you want to use DynamicBuffer to build the
    // file contents from the slices.
    uint256 size;
    BytecodeSlice[] slices;
}
// extend File struct with read functions

using { read } for File global;
using { readUnchecked } for File global;

/**
 * @dev Error thrown when a slice is out of the bounds of the contract's bytecode
 */
error SliceOutOfBounds(address pointer, uint32 codeSize, uint32 sliceStart, uint32 sliceEnd);

/**
 * @notice Reads the contents of a file by concatenating its slices
 * @param file The file to read
 * @return contents The concatenated contents of the file
 */
function read(File memory file) view returns (string memory contents) {
    BytecodeSlice[] memory slices = file.slices;
    bytes4 sliceOutOfBoundsSelector = SliceOutOfBounds.selector;

    assembly {
        let len := mload(slices)
        let size := 0x20
        contents := mload(0x40)
        let slice
        let pointer
        let start
        let end
        let codeSize

        for { let i := 0 } lt(i, len) { i := add(i, 1) } {
            slice := mload(add(slices, add(0x20, mul(i, 0x20))))
            pointer := mload(slice)
            start := mload(add(slice, 0x20))
            end := mload(add(slice, 0x40))

            codeSize := extcodesize(pointer)
            if gt(end, codeSize) {
                mstore(0x00, sliceOutOfBoundsSelector)
                mstore(0x04, pointer)
                mstore(0x24, codeSize)
                mstore(0x44, start)
                mstore(0x64, end)
                revert(0x00, 0x84)
            }

            extcodecopy(pointer, add(contents, size), start, sub(end, start))
            size := add(size, sub(end, start))
        }

        // update contents size
        mstore(contents, sub(size, 0x20))
        // store contents
        mstore(0x40, add(contents, and(add(size, 0x1f), not(0x1f))))
    }
}

/**
 * @notice Reads the contents of a file without reverting on unreadable/invalid slices. Skips any slices that are out of
 * bounds or invalid. Useful if you are composing contract bytecode where a contract can still selfdestruct (which would
 * result in an invalid slice) and want to avoid reverts but still output potentially "corrupted" file contents (due to
 * missing data).
 * @param file The file to read
 * @return contents The concatenated contents of the file, skipping invalid slices
 */
function readUnchecked(File memory file) view returns (string memory contents) {
    BytecodeSlice[] memory slices = file.slices;

    assembly {
        let len := mload(slices)
        let size := 0x20
        contents := mload(0x40)
        let slice
        let pointer
        let start
        let end
        let codeSize

        for { let i := 0 } lt(i, len) { i := add(i, 1) } {
            slice := mload(add(slices, add(0x20, mul(i, 0x20))))
            pointer := mload(slice)
            start := mload(add(slice, 0x20))
            end := mload(add(slice, 0x40))

            codeSize := extcodesize(pointer)
            if lt(end, codeSize) {
                extcodecopy(pointer, add(contents, size), start, sub(end, start))
                size := add(size, sub(end, start))
            }
        }

        // update contents size
        mstore(contents, sub(size, 0x20))
        // store contents
        mstore(0x40, add(contents, and(add(size, 0x1f), not(0x1f))))
    }
}
/// @title EthFS FileStore interface
/// @notice Specifies a content-addressable onchain file store

interface IFileStore {
    event Deployed();

    /**
     * @dev Emitted when a new file is created
     * @param indexedFilename The indexed filename for easier finding by filename in logs
     * @param pointer The pointer address of the file
     * @param filename The name of the file
     * @param size The total size of the file
     * @param metadata Additional metadata of the file, only emitted for use in offchain indexers
     */
    event FileCreated(
        string indexed indexedFilename, address indexed pointer, string filename, uint256 size, bytes metadata
    );

    /**
     * @dev Error thrown when a requested file is not found
     * @param filename The name of the file requested
     */
    error FileNotFound(string filename);

    /**
     * @dev Error thrown when a filename already exists
     * @param filename The name of the file attempted to be created
     */
    error FilenameExists(string filename);

    /**
     * @dev Error thrown when attempting to create an empty file
     */
    error FileEmpty();

    /**
     * @dev Error thrown when a provided slice for a file is empty
     * @param pointer The contract address where the bytecode lives
     * @param start The byte offset to start the slice (inclusive)
     * @param end The byte offset to end the slice (exclusive)
     */
    error SliceEmpty(address pointer, uint32 start, uint32 end);

    /**
     * @dev Error thrown when the provided pointer's bytecode does not have the expected STOP opcode prefix from SSTORE2
     * @param pointer The SSTORE2 pointer address
     */
    error InvalidPointer(address pointer);

    /**
     * @notice Returns the address of the CREATE2 deterministic deployer used by this FileStore
     * @return The address of the CREATE2 deterministic deployer
     */
    function deployer() external view returns (address);

    /**
     * @notice Retrieves the pointer address of a file by its filename
     * @param filename The name of the file
     * @return pointer The pointer address of the file
     */
    function files(string memory filename) external view returns (address pointer);

    /**
     * @notice Checks if a file exists for a given filename
     * @param filename The name of the file to check
     * @return True if the file exists, false otherwise
     */
    function fileExists(string memory filename) external view returns (bool);

    /**
     * @notice Retrieves the pointer address for a given filename
     * @param filename The name of the file
     * @return pointer The pointer address of the file
     */
    function getPointer(string memory filename) external view returns (address pointer);

    /**
     * @notice Retrieves a file by its filename
     * @param filename The name of the file
     * @return file The file associated with the filename
     */
    function getFile(string memory filename) external view returns (File memory file);

    /**
     * @notice Creates a new file with the provided file contents
     * @dev This is a convenience method to simplify small file uploads. It's recommended to use
     * `createFileFromPointers` or `createFileFromSlices` for larger files. This particular method splits `contents`
     * into 24575-byte chunks before storing them via SSTORE2.
     * @param filename The name of the new file
     * @param contents The contents of the file
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFile(
        string memory filename,
        string memory contents
    )
        external
        returns (address pointer, File memory file);

    /**
     * @notice Creates a new file with the provided file contents and file metadata
     * @dev This is a convenience method to simplify small file uploads. It's recommended to use
     * `createFileFromPointers` or `createFileFromSlices` for larger files. This particular method splits `contents`
     * into 24575-byte chunks before storing them via SSTORE2.
     * @param filename The name of the new file
     * @param contents The contents of the file
     * @param metadata Additional file metadata, usually a JSON-encoded string, for offchain indexers
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFile(
        string memory filename,
        string memory contents,
        bytes memory metadata
    )
        external
        returns (address pointer, File memory file);

    /**
     * @notice Creates a new file where its content is composed of the provided string chunks
     * @dev This is a convenience method to simplify small and nuanced file uploads. It's recommended to use
     * `createFileFromPointers` or `createFileFromSlices` for larger files. This particular will store each chunk
     * separately via SSTORE2. For best gas efficiency, each chunk should be as large as possible (up to the contract
     * size limit) and at least 32 bytes.
     * @param filename The name of the new file
     * @param chunks The string chunks composing the file
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFileFromChunks(
        string memory filename,
        string[] memory chunks
    )
        external
        returns (address pointer, File memory file);

    /**
     * @notice Creates a new file with the provided string chunks and file metadata
     * @dev This is a convenience method to simplify small and nuanced file uploads. It's recommended to use
     * `createFileFromPointers` or `createFileFromSlices` for larger files. This particular will store each chunk
     * separately via SSTORE2. For best gas efficiency, each chunk should be as large as possible (up to the contract
     * size limit) and at least 32 bytes.
     * @param filename The name of the new file
     * @param chunks The string chunks composing the file
     * @param metadata Additional file metadata, usually a JSON-encoded string, for offchain indexers
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFileFromChunks(
        string memory filename,
        string[] memory chunks,
        bytes memory metadata
    )
        external
        returns (address pointer, File memory file);

    /**
     * @notice Creates a new file where its content is composed of the provided SSTORE2 pointers
     * @param filename The name of the new file
     * @param pointers The SSTORE2 pointers composing the file
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFileFromPointers(
        string memory filename,
        address[] memory pointers
    )
        external
        returns (address pointer, File memory file);

    /**
     * @notice Creates a new file with the provided SSTORE2 pointers and file metadata
     * @param filename The name of the new file
     * @param pointers The SSTORE2 pointers composing the file
     * @param metadata Additional file metadata, usually a JSON-encoded string, for offchain indexers
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFileFromPointers(
        string memory filename,
        address[] memory pointers,
        bytes memory metadata
    )
        external
        returns (address pointer, File memory file);

    /**
     * @notice Creates a new file where its content is composed of the provided bytecode slices
     * @param filename The name of the new file
     * @param slices The bytecode slices composing the file
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFileFromSlices(
        string memory filename,
        BytecodeSlice[] memory slices
    )
        external
        returns (address pointer, File memory file);

    /**
     * @notice Creates a new file with the provided bytecode slices and file metadata
     * @param filename The name of the new file
     * @param slices The bytecode slices composing the file
     * @param metadata Additional file metadata, usually a JSON-encoded string, for offchain indexers
     * @return pointer The pointer address of the new file
     * @return file The newly created file
     */
    function createFileFromSlices(
        string memory filename,
        BytecodeSlice[] memory slices,
        bytes memory metadata
    )
        external
        returns (address pointer, File memory file);
}

interface FileStore {
    function readFile(string memory filename) external view returns (string memory contents);
}
