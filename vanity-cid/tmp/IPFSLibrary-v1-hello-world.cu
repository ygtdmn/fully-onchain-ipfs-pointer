#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// Constants
const unsigned char CIDV1_PREFIX[] = {0x01};
const unsigned char RAW_CODEC[] = {0x55};
const unsigned char SHA256_CODEC[] = {0x12};
const int DIGEST_LENGTH = 32;

// SHA-256 constants
__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 functions
__device__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t ep0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ uint32_t ep1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ uint32_t sig0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t sig1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// SHA-256 kernel
__global__ void sha256Kernel(const uint8_t* input, size_t inputLen, uint32_t* hash) {
    uint32_t w[64];
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    size_t paddedLen = ((inputLen + 8) / 64 + 1) * 64;
    uint8_t padded[256];  // Assuming max input length of 192 bytes (256 - 64)
    
    // Manual padding
    for (size_t i = 0; i < inputLen; i++) {
        padded[i] = input[i];
    }
    padded[inputLen] = 0x80;
    for (size_t i = inputLen + 1; i < paddedLen - 8; i++) {
        padded[i] = 0;
    }
    uint64_t bitLen = inputLen * 8;
    for (int i = 0; i < 8; i++) {
        padded[paddedLen - 8 + i] = (bitLen >> (56 - i * 8)) & 0xFF;
    }

    for (size_t chunk = 0; chunk < paddedLen; chunk += 64) {
        for (int i = 0; i < 16; i++) {
            w[i] = (padded[chunk + i*4] << 24) | (padded[chunk + i*4 + 1] << 16) |
                   (padded[chunk + i*4 + 2] << 8) | padded[chunk + i*4 + 3];
        }

        for (int i = 16; i < 64; i++) {
            w[i] = sig1(w[i-2]) + w[i-7] + sig0(w[i-15]) + w[i-16];
        }

        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + ep1(e) + ch(e, f, g) + k[i] + w[i];
            uint32_t t2 = ep0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    for (int i = 0; i < 8; i++) {
        hash[i] = state[i];
    }
}

// Base32 alphabet
const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz234567";

// CPU-based base32 encoding function
void base32Encode(const unsigned char* input, int inputLen, char* output) {
    int i = 0, j = 0;
    uint64_t buffer = 0;
    int bufferSize = 0;

    while (i < inputLen) {
        buffer = (buffer << 8) | input[i++];
        bufferSize += 8;

        while (bufferSize >= 5) {
            bufferSize -= 5;
            output[j++] = ALPHABET[(buffer >> bufferSize) & 0x1F];
        }
    }

    if (bufferSize > 0) {
        buffer <<= (5 - bufferSize);
        output[j++] = ALPHABET[buffer & 0x1F];
    }

    output[j] = '\0';
}

// Helper function to check CUDA errors
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(0); \
    } \
}

// Host function to generate CIDv1
void generateCIDv1(const unsigned char* content, int contentLen, char* output, size_t outputSize) {
    // Generate SHA-256 hash
    uint32_t* contentHash;
    cudaMallocManaged(&contentHash, 8 * sizeof(uint32_t));
    cudaCheckError();
    
    uint8_t* d_content;
    cudaMalloc(&d_content, contentLen);
    cudaCheckError();
    cudaMemcpy(d_content, content, contentLen, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    sha256Kernel<<<1, 1>>>(d_content, contentLen, contentHash);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    cudaFree(d_content);
    cudaCheckError();

    // Debugging: Print the hash
    printf("SHA-256 Hash: ");
    for (int i = 0; i < 8; i++) {
        printf("%08x", contentHash[i]);
    }
    printf("\n");

    // Construct multihash
    unsigned char multihash[34];
    multihash[0] = SHA256_CODEC[0];
    multihash[1] = DIGEST_LENGTH;
    for (int i = 0; i < 8; i++) {
        uint32_t word = contentHash[i];
        for (int j = 0; j < 4; j++) {
            multihash[2 + i*4 + j] = (word >> (24 - j*8)) & 0xFF;
        }
    }

    // Construct CIDv1
    unsigned char cid[36];
    cid[0] = CIDV1_PREFIX[0];
    cid[1] = RAW_CODEC[0];
    memcpy(cid + 2, multihash, 34);

    // Debugging: Print CID
    printf("CID: ");
    for (int i = 0; i < 36; i++) {
        printf("%02x", cid[i]);
    }
    printf("\n");

    // Perform base32 encoding
    char encoded[64];  // Temporary buffer for encoded result
    base32Encode(cid, 36, encoded);

    // Prepare final output
    size_t encodedLen = strlen(encoded);
    if (outputSize < encodedLen + 2) {  // +2 for 'b' prefix and null terminator
        fprintf(stderr, "Error: Output buffer too small\n");
        return;
    }
    output[0] = 'b';
    strncpy(output + 1, encoded, outputSize - 2);
    output[outputSize - 1] = '\0';

    // Debugging: Print result
    printf("Base32 encoded: %s\n", output);

    cudaFree(contentHash);
}

// Main function for testing
int main() {
    const char* testContent = "Hello, IPFS!";
    char output[64] = {0};  // Initialize to zero

    printf("Input content: %s\n", testContent);
    generateCIDv1((const unsigned char*)testContent, strlen(testContent), output, sizeof(output));
    printf("CIDv1: %s\n", output);

    return 0;
}
