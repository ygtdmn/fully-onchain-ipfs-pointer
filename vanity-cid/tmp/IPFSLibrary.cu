#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <openssl/sha.h>

// Constants for both host and device code
#define CIDV1_PREFIX 0x01
#define RAW_CODEC 0x55
#define SHA256_CODEC 0x12
#define DIGEST_LENGTH 32

// Define ALPHABET as a device constant
__device__ __constant__ const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz234567";

// Device function to calculate string length
__device__ int d_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// Device function for base32 encoding
__device__ void base32Encode(const uint8_t* input, int inputLen, char* output) {
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

// SHA-256 constants
__device__ __constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 helper functions
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

// Device function for SHA-256
__device__ void sha256Kernel(const uint8_t* input, size_t inputLen, uint32_t* hash) {
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

__global__ void findVanityCID(const uint8_t* charset, int charsetSize, uint8_t* result, int* found) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t state = tid;
    char content[60];
    int contentLength = (state % 50) + 10;  // Random length between 10 and 59

    // Generate random content
    for (int i = 0; i < contentLength; i++) {
        state = state * 1664525 + 1013904223;  // Linear congruential generator
        content[i] = charset[state % charsetSize];
    }

    uint32_t hash[8];
    sha256Kernel((const uint8_t*)content, contentLength, hash);

    // Construct CIDv1
    uint8_t cid[36];
    cid[0] = CIDV1_PREFIX;
    cid[1] = RAW_CODEC;
    cid[2] = SHA256_CODEC;
    cid[3] = DIGEST_LENGTH;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            cid[4 + i*4 + j] = (hash[i] >> (24 - j*8)) & 0xFF;
        }
    }

    // Base32 encode
    char encoded[64];
    base32Encode(cid, 36, encoded);

    // Check if it includes "chain"
    bool includes_chain = false;
    int encodedLen = d_strlen(encoded);
    for (int i = 0; i <= encodedLen - 5; i++) {
        if (encoded[i] == 'c' && encoded[i+1] == 'h' && encoded[i+2] == 'a' &&
            encoded[i+3] == 'i' && encoded[i+4] == 'n') {
            includes_chain = true;
            break;
        }
    }

    if (includes_chain) {
        *found = 1;
        memcpy(result, content, contentLength);
        result[contentLength] = '\0';
    }
}

// Host function for base32 encoding
void base32EncodeHost(const unsigned char* input, int inputLen, char* output) {
    int i = 0, j = 0;
    uint64_t buffer = 0;
    int bufferSize = 0;
    const char ALPHABET[] = "abcdefghijklmnopqrstuvwxyz234567";

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

// Modify generateCIDv1 to use CPU-based SHA-256 and base32 encoding
void generateCIDv1(const unsigned char* content, int contentLen, char* output, size_t outputSize) {
    // Use CPU-based SHA-256 implementation
    unsigned char hash[32];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, content, contentLen);
    SHA256_Final(hash, &sha256);

    // Construct multihash
    unsigned char multihash[34];
    multihash[0] = SHA256_CODEC;
    multihash[1] = DIGEST_LENGTH;
    memcpy(multihash + 2, hash, 32);

    // Construct CIDv1
    unsigned char cid[36];
    cid[0] = CIDV1_PREFIX;
    cid[1] = RAW_CODEC;
    memcpy(cid + 2, multihash, 34);

    // Perform base32 encoding
    char encoded[64];
    base32EncodeHost(cid, 36, encoded);

    // Prepare final output
    size_t encodedLen = strlen(encoded);
    if (outputSize < encodedLen + 2) {
        fprintf(stderr, "Error: Output buffer too small\n");
        return;
    }
    output[0] = 'b';
    strncpy(output + 1, encoded, outputSize - 2);
    output[outputSize - 1] = '\0';
}

// Helper function to check CUDA errors
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(0); \
    } \
}

// Function to generate random content
void generateRandomContent(char* buffer, int length) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int i = 0; i < length; i++) {
        buffer[i] = charset[rand() % (sizeof(charset) - 1)];
    }
    buffer[length] = '\0';
}

#define BLOCK_SIZE 256
#define GRID_SIZE 1024

int main() {
    uint8_t* d_charset;
    uint8_t* d_result;
    int* d_found;
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    int charsetSize = sizeof(charset) - 1;

    cudaMalloc(&d_charset, charsetSize);
    cudaMalloc(&d_result, 60);
    cudaMalloc(&d_found, sizeof(int));
    cudaMemcpy(d_charset, charset, charsetSize, cudaMemcpyHostToDevice);

    uint8_t h_result[60];
    int h_found = 0;
    uint64_t attempts = 0;
    uint64_t last_reported_attempts = 0;

    clock_t start = clock();

    while (!h_found) {
        cudaMemset(d_found, 0, sizeof(int));
        findVanityCID<<<GRID_SIZE, BLOCK_SIZE>>>(d_charset, charsetSize, d_result, d_found);
        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        attempts += GRID_SIZE * BLOCK_SIZE;

        if (attempts - last_reported_attempts >= 1000000) {
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            double rate = attempts / elapsed / 1000000.0;
            printf("Attempts: %lu million (%.2f M/s)\n", attempts / 1000000, rate);
            last_reported_attempts = attempts;
        }
    }

    cudaMemcpy(h_result, d_result, 60, cudaMemcpyDeviceToHost);

    char output[64];
    generateCIDv1(h_result, strlen((char*)h_result), output, sizeof(output));

    double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    double final_rate = attempts / total_time / 1000000.0;

    printf("Found CID including 'chain' after %lu attempts!\n", attempts);
    printf("Total time: %.2f seconds\n", total_time);
    printf("Final rate: %.2f M/s\n", final_rate);
    printf("Content: %s\n", h_result);
    printf("CIDv1: %s\n", output);

    cudaFree(d_charset);
    cudaFree(d_result);
    cudaFree(d_found);

    return 0;
}