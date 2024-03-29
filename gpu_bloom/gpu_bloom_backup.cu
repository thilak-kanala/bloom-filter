#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include <stddef.h>   /* uint32_t, NULL */
#include <stdint.h>   /* uint8_t, uint32_t, uint64_t */
#include <inttypes.h> /* print formatting */

#define BLOOM_FILTER_SIZE 33547705
#define N_HASHES 23
#define FILE_WORDS_TO_INSERT "words_insert.txt"
#define FILE_WORDS_TO_QUERY "words_query.txt"

uint32_t total_words = 0;

__device__ uint32_t djb2(char *words_to_insert, int len_words_to_insert);
__device__ uint32_t jenkins(char *words_to_insert, int len_words_to_insert);

__device__ __host__ uint64_t XXH64(void const *const input, uint32_t const length, uint64_t const seed);

__device__ __host__ void print_bloom_filter(uint32_t *bloom_filter);

/* === BIT VECTOR BEGIN ===*/

// sets the kth bit in bitvector 'bv'
__device__
void set_bit(uint32_t *bv, uint32_t k)
{
    bv[k / 32] |= (1 << (k % 32));
}

// clears the kth bit in bitvector 'bv'
__device__ __host__
void clear_bit(uint32_t *bv, uint32_t k)
{
    bv[k / 32] &= ~(1 << (k % 32));
}

__device__ __host__
int test_bit(uint32_t *bv, uint32_t k)
{
    return ((bv[k / 32] & (1 << (k % 32))) != 0);
}

/* === BIT VECTOR END ===*/

/* === XXHASH64 === */
// Reference: https://github.com/easyaspi314/xxhash-clean/blob/master/xxhash64-ref.c

__device__ __host__
uint64_t
XXH64(void const *const input, uint32_t const length, uint64_t const seed);

static uint64_t const PRIME64_1 = 0x9E3779B185EBCA87ULL; /* 0b1001111000110111011110011011000110000101111010111100101010000111 */
static uint64_t const PRIME64_2 = 0xC2B2AE3D27D4EB4FULL; /* 0b1100001010110010101011100011110100100111110101001110101101001111 */
static uint64_t const PRIME64_3 = 0x165667B19E3779F9ULL; /* 0b0001011001010110011001111011000110011110001101110111100111111001 */
static uint64_t const PRIME64_4 = 0x85EBCA77C2B2AE63ULL; /* 0b1000010111101011110010100111011111000010101100101010111001100011 */
static uint64_t const PRIME64_5 = 0x27D4EB2F165667C5ULL; /* 0b0010011111010100111010110010111100010110010101100110011111000101 */

/* Rotates value left by amt bits. */
__device__ __host__ static uint64_t XXH_rotl64(uint64_t const value, uint32_t const amt)
{
    return (value << (amt % 64)) | (value >> (64 - amt % 64));
}

/* Portably reads a 32-bit little endian integer from data at the given offset. */
__device__ __host__ static uint32_t XXH_read32(uint8_t const *const data, uint32_t const offset)
{
    return (uint32_t)data[offset + 0] | ((uint32_t)data[offset + 1] << 8) | ((uint32_t)data[offset + 2] << 16) | ((uint32_t)data[offset + 3] << 24);
}

/* Portably reads a 64-bit little endian integer from data at the given offset. */
__device__ __host__ static uint64_t XXH_read64(uint8_t const *const data, uint32_t const offset)
{
    return (uint64_t)data[offset + 0] | ((uint64_t)data[offset + 1] << 8) | ((uint64_t)data[offset + 2] << 16) | ((uint64_t)data[offset + 3] << 24) | ((uint64_t)data[offset + 4] << 32) | ((uint64_t)data[offset + 5] << 40) | ((uint64_t)data[offset + 6] << 48) | ((uint64_t)data[offset + 7] << 56);
}

/* Mixes input into acc, this is mostly used in the first loop. */
__device__ __host__ static uint64_t XXH64_round(uint64_t acc, uint64_t const input)
{
    acc += input * PRIME64_2;
    acc = XXH_rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

/* Merges acc into hash to finalize */
__device__ __host__ static uint64_t XXH64_mergeRound(uint64_t hash, uint64_t const acc)
{
    hash ^= XXH64_round(0, acc);
    hash *= PRIME64_1;
    hash += PRIME64_4;
    return hash;
}

/* Mixes all bits to finalize the hash. */
__device__ __host__ static uint64_t XXH64_avalanche(uint64_t hash)
{
    hash ^= hash >> 33;
    hash *= PRIME64_2;
    hash ^= hash >> 29;
    hash *= PRIME64_3;
    hash ^= hash >> 32;
    return hash;
}

/* The XXH64 hash function.
* input:   The data to hash.
* length:  The length of input. It is undefined behavior to have length larger than the
*          capacity of input.
* seed:    A 64-bit value to seed the hash with.
* returns: The 64-bit calculated hash value. */
__device__ __host__
    uint64_t
    XXH64(void const *const input, uint32_t const length, uint64_t const seed)
{
    uint8_t const *const data = (uint8_t const *)input;
    uint64_t hash = 0;
    uint32_t remaining = length;
    uint32_t offset = 0;

    /* Don't dereference a null pointer. The reference implementation notably doesn't
    * check for this by default. */
    if (input == NULL)
    {
        return XXH64_avalanche(seed + PRIME64_5);
    }

    if (remaining >= 32)
    {
        /* Initialize our accumulators */
        uint64_t acc1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t acc2 = seed + PRIME64_2;
        uint64_t acc3 = seed + 0;
        uint64_t acc4 = seed - PRIME64_1;

        while (remaining >= 32)
        {
            acc1 = XXH64_round(acc1, XXH_read64(data, offset));
            offset += 8;
            acc2 = XXH64_round(acc2, XXH_read64(data, offset));
            offset += 8;
            acc3 = XXH64_round(acc3, XXH_read64(data, offset));
            offset += 8;
            acc4 = XXH64_round(acc4, XXH_read64(data, offset));
            offset += 8;
            remaining -= 32;
        }

        hash = XXH_rotl64(acc1, 1) + XXH_rotl64(acc2, 7) + XXH_rotl64(acc3, 12) + XXH_rotl64(acc4, 18);

        hash = XXH64_mergeRound(hash, acc1);
        hash = XXH64_mergeRound(hash, acc2);
        hash = XXH64_mergeRound(hash, acc3);
        hash = XXH64_mergeRound(hash, acc4);
    }
    else
    {
        /* Not enough data for the main loop, put something in there instead. */
        hash = seed + PRIME64_5;
    }

    hash += (uint64_t)length;

    /* Process the remaining data. */
    while (remaining >= 8)
    {
        hash ^= XXH64_round(0, XXH_read64(data, offset));
        hash = XXH_rotl64(hash, 27);
        hash *= PRIME64_1;
        hash += PRIME64_4;
        offset += 8;
        remaining -= 8;
    }

    if (remaining >= 4)
    {
        hash ^= (uint64_t)XXH_read32(data, offset) * PRIME64_1;
        hash = XXH_rotl64(hash, 23);
        hash *= PRIME64_2;
        hash += PRIME64_3;
        offset += 4;
        remaining -= 4;
    }

    while (remaining != 0)
    {
        hash ^= (uint64_t)data[offset] * PRIME64_5;
        hash = XXH_rotl64(hash, 11);
        hash *= PRIME64_1;
        ++offset;
        --remaining;
    }

    return XXH64_avalanche(hash);
}

/* === XXH64 END === */

__device__ __host__ void print_binary_64(uint64_t n, const char *message)
{
    printf("%s", message);
    int binary[64];
    int bi = 63;
    for (int i = 0; i < 64; i++)
    {
        uint64_t mask = 1;
        if (n & (mask << i))
        {
            binary[bi--] = 1;
        }
        else
        {
            binary[bi--] = 0;
        }
    }

    for (int i = 0; i < 64; i++)
    {
        if (i % 4 == 0)
        {
            printf("\t");
        }
        printf("%d", binary[i]);
    }
    printf("\n");
}

__device__ __host__ void print_binary_32(uint64_t n, const char *message)
{
    printf("%s", message);
    int binary[32];
    int bi = 31;
    for (int i = 0; i < 32; i++)
    {
        uint64_t mask = 1;
        if (n & (mask << i))
        {
            binary[bi--] = 1;
        }
        else
        {
            binary[bi--] = 0;
        }
    }

    for (int i = 0; i < 32; i++)
    {
        if (i % 4 == 0)
        {
            printf("\t");
        }
        printf("%d", binary[i]);
    }
    printf("\n");
}

/* Splits the 64bit hash into 2 32 bit hashes h1 and h2 */
__device__ __host__ void split_hash_bits(uint64_t hash, uint64_t *h1, uint64_t *h2)
{
    uint64_t mask;
    uint64_t one_64bit = 1;

    // Clear all bits
    mask = 0;

    // Create mask to extract bottom 32 bits
    for (int i = 0; i < 32; i++)
    {
        mask |= (one_64bit << i);
    }

    // Extract bottom 32 bits
    *h1 = hash & mask;

    // Create mask to extract top 32 bits
    for (int i = 32; i < 64; i++)
    {
        mask |= (one_64bit << i);
    }

    // Extract top 32 bits
    *h2 = (hash & mask) >> 32;
}


/* Generate a hash based on The Combinatorial Approach (https://github.com/Claudenw/BloomFilter/wiki/Bloom-Filters----An-Overview) */
__device__ void bloom_insert(uint32_t *d_bloom_filter, char *word, uint32_t word_len, int tid)
{
    uint32_t index = 0;

    uint64_t hash = XXH64((uint8_t*) word, word_len, 0);
    uint64_t h1;
    uint64_t h2;

    split_hash_bits(hash, &h1, &h2);

    for (uint32_t i = 0; i < N_HASHES; i++)
    {
        h1 += (h2 * i);
        h1 = h1 % BLOOM_FILTER_SIZE;
        index = (uint32_t)h1;

        // set index bit
        atomicOr(&d_bloom_filter[index / 32], (1 << (index % 32)));
    }
}

/* Generate a hash based on The Combinatorial Approach (https://github.com/Claudenw/BloomFilter/wiki/Bloom-Filters----An-Overview) */
__device__ void bloom_query(uint32_t *d_bloom_filter, char *word, uint32_t word_len, uint32_t *d_query_results, int tid)
{
    int is_present = 1;
    uint32_t index;
    uint32_t bloom_filter_partial;

    uint64_t hash = XXH64((uint8_t*) word, word_len, 0);
    uint64_t h1;
    uint64_t h2;

    split_hash_bits(hash, &h1, &h2);

    for (uint32_t i = 0; i < N_HASHES; i++)
    {
        h1 += (h2 * i);
        h1 = h1 % BLOOM_FILTER_SIZE;
        index = (uint32_t)h1;

        // extract the relevant part (32 bits) of bloom filter
        bloom_filter_partial = d_bloom_filter[index / 32];

        if ((bloom_filter_partial & (1 << (index % 32))) == 0)
        {
            is_present = 0;
            break;
        }
    }

    if (is_present)
    {
        // set the tid bit to indicate the word processed by tid is present
        atomicOr(&d_query_results[tid / 32], (1 << (tid % 32)));
    }
}

__global__ void map_bloom_kernel(char *d_words_to_insert, int len_words_to_insert, uint32_t *d_word_indices, uint32_t *d_bloom_filter, uint32_t total_words)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= total_words)
    {
        return;
    }

    // Find word start and end indices
    uint32_t word_len = d_word_indices[tid + 1] - d_word_indices[tid] - 1;
    uint32_t si = d_word_indices[tid] + 1;
    uint32_t ei = d_word_indices[tid + 1] - 1;

    // Copy string to private memory?
    char *word = (char *)malloc((ei - si + 1) * sizeof(char));
    for (uint32_t i = si; si <= ei; si++)
    {
        word[si - i] = d_words_to_insert[si];
    }

    // Add word to bloom filter
    bloom_insert(d_bloom_filter, word, word_len, tid);
}

__global__ void query_bloom_kernel(char *d_words_to_query, int len_words_to_insert, uint32_t *d_word_indices, uint32_t *d_bloom_filter, uint32_t *d_query_results, uint32_t total_words)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // printf("%d: query kernel\n", tid);

    if (tid >= total_words)
    {
        return;
    }

    // Find word start and end indices
    uint32_t word_len = d_word_indices[tid + 1] - d_word_indices[tid] - 1;
    uint32_t si = d_word_indices[tid] + 1;
    uint32_t ei = d_word_indices[tid + 1] - 1;

    // Copy string to private memory?
    char *word = (char *)malloc((ei - si + 1) * sizeof(char));

    for (uint32_t i = si; si <= ei; si++)
    {
        word[si - i] = d_words_to_query[si];
    }

    // Add word to bloom filter
    bloom_query(d_bloom_filter, word, word_len, d_query_results, tid);
}

// Hash Functions
__device__
uint32_t
djb2(char *words_to_insert, int len_words_to_insert)
{
    uint32_t hash = 5381;
    char c;
    uint32_t i = 0;
    while (i++ < len_words_to_insert)
    {
        c = *words_to_insert++;
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

__device__
uint32_t
jenkins(char *words_to_insert, int len_words_to_insert)
{
    uint32_t hash = 5381;
    uint32_t i = 0;
    while (i < len_words_to_insert)
    {
        hash += *words_to_insert++;
        hash += (hash << 10);
        hash ^= (hash >> 6);
        i++;
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

__device__ __host__ void print_bloom_filter(uint32_t *bloom_filter)
{
    // printf("Bloom Filter: ");
    // for (int i = 0; i < BLOOM_FILTER_SIZE; i++)
    // {
    //     if (test_bit(bloom_filter, i))
    //     {
    //         printf("1, ");
    //     }
    //     else
    //     {
    //         printf("0, ");
    //     }
    // }
    // printf("\n");

    // printf("The follwing bits are set in the bloom filter: ");
    // for (int i = 0; i < BLOOM_FILTER_SIZE; i++)
    // {
    //     if (test_bit(bloom_filter, i))
    //     {
    //         printf("%d, ", i);
    //     }
    // }
    // printf("\n\n");

    uint32_t count = 0;
    for (int i = 0; i < BLOOM_FILTER_SIZE; i++)
    {
        if (test_bit(bloom_filter, i))
        {
            count++;
        }
    }
    printf("%d bits are set in the bloom filter\n\n", count);
}

void print_query_results(const char *words_to_insert, int len_strin_in, uint32_t *word_indices, uint32_t *query_results, int total_words)
{
    printf("\n");

    int present = 0;

    for (int i = 0; i < total_words; i++)
    {
        // Find word start and end indices
        int word_len = word_indices[i + 1] - word_indices[i] - 1;
        int si = word_indices[i] + 1;
        int ei = word_indices[i + 1] - 1;

        char *word = (char *)malloc((ei - si + 2) * sizeof(char));
        for (int j = si; si <= ei; si++)
        {
            word[si - j] = words_to_insert[si];
        }
        word[word_len] = '\0';

        if (test_bit(query_results, i))
        {
            // printf("%s: present\n", word);
            present += 1;
        }
        else
        {
            // printf("%s: absent\n", word);
        }
    }

    printf("Query Result: %d / %d are present\n\n", present, total_words);
}

char *read_from_file(char *file_location)
{
    FILE *fp;
    uint32_t lSize;
    char *buffer;

    fp = fopen(file_location, "r");
    if( !fp ) 
    {
        perror(file_location);
        exit(1);
    }

    fseek(fp, 0L, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    /* allocate memory for entire content */
    buffer = (char *)calloc(1, lSize + 1);
    if (!buffer)
        fclose(fp), fputs("memory alloc fails", stderr), exit(1);

    /* copy the file into the buffer */
    if (1 != fread(buffer, lSize, 1, fp))
        fclose(fp), free(buffer), fputs("entire read fails", stderr), exit(1);

    fclose(fp);

    return buffer;
}

int main(void)
{
    cudaFree(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* === Intialize profiling related variables === */
    // cudaEvent_t start, stop;
    // float time;

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    /* === Map Words to Bloom Filter === */

    char* words_to_insert = read_from_file(FILE_WORDS_TO_INSERT);
    uint32_t len_words_to_insert = strlen(words_to_insert);

    uint32_t *word_indices;
    int wi = -1;
    total_words = 0;

    uint32_t *bloom_filter = (uint32_t *)calloc(ceil(BLOOM_FILTER_SIZE / 32.0), sizeof(uint32_t));

    for (uint32_t i = 0; i < len_words_to_insert; i++)
    {
        if (words_to_insert[i] == ' ')
        {
            total_words++;
        }
    }
    // To account for the space at the beginning and the end
    total_words -= 1;

    printf("\n---\n");
    printf("Number of items to insert: %d\n", total_words);
    printf("Number of bits in the Bloom Filter: %d\n", BLOOM_FILTER_SIZE);
    printf("Number of hash functions: %d\n", N_HASHES);
    printf("---\n");

    word_indices = (uint32_t *)calloc((total_words + 1), sizeof(uint32_t));

    for (uint32_t i = 0; i < len_words_to_insert; i++)
    {
        if (words_to_insert[i] == ' ')
        {
            word_indices[++wi] = i;
        }
    }

    printf("\n == Before Inserting Words ==\n");
    print_bloom_filter(bloom_filter);

    char *d_words_to_insert;
    uint32_t *d_word_indices;
    uint32_t *d_bloom_filter;

    cudaEventRecord(start);
    cudaMalloc((void **)&d_words_to_insert, len_words_to_insert * sizeof(char));
    cudaMalloc((void **)&d_word_indices, (total_words + 1) * sizeof(uint32_t));
    cudaMalloc((void **)&d_bloom_filter, ceil(BLOOM_FILTER_SIZE / 32.0) * sizeof(uint32_t));

    cudaMemcpy(d_words_to_insert, words_to_insert, len_words_to_insert * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_indices, word_indices, (total_words + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom_filter, bloom_filter, ceil(BLOOM_FILTER_SIZE / 32.0) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Transferring strings from Host to GPU Global Memory took %f ms\n", milliseconds);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("1.1.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    // cudaEventRecord( start, 0 );

    cudaEventRecord(start);
    map_bloom_kernel<<<((total_words + 511) / 512), 512>>>(d_words_to_insert, len_words_to_insert, d_word_indices, d_bloom_filter, total_words);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Inserting words to Bloom Filter (kernel) took:  %f ms\n", milliseconds);

    err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("1.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    cudaEventRecord(start);
    cudaMemcpy(bloom_filter, d_bloom_filter, ceil(BLOOM_FILTER_SIZE / 32.0) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Transferring Bloom Filter from GPU Global Memory to Host took %f ms\n", milliseconds);


    err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("2.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    printf("\n == After Inserting Words to Bloom Filter ==\n");
    print_bloom_filter(bloom_filter);

    // printf("== Inserting words using gpu took %f ms ==\n", time);

    cudaFree(d_words_to_insert);
    cudaFree(d_word_indices);

    err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("3.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    /* === Query Bloom Filter === */

    // char *words_to_query = read_from_file(FILE_WORDS_TO_QUERY);
    // /* char *words_to_query = buffer; */
    // uint32_t len_words_to_query = strlen(words_to_query);

    // total_words = 0;
    // for (uint32_t i = 0; i < len_words_to_query; i++)
    // {
    //     if (words_to_query[i] == ' ')
    //     {
    //         total_words++;
    //     }
    // }
    // // To account for the space at the beginning and the end
    // total_words -= 1;

    // word_indices = (uint32_t *)calloc((total_words + 1), sizeof(int));

    // wi = -1;
    // for (uint32_t i = 0; i < len_words_to_query; i++)
    // {
    //     if (words_to_query[i] == ' ')
    //     {
    //         word_indices[++wi] = i;
    //     }
    // }

    // uint32_t *query_results = (uint32_t *)calloc(total_words, sizeof(uint32_t));
    // uint32_t *d_query_results;
    // char *d_words_to_query;


    // cudaMalloc((void **)&d_words_to_query, len_words_to_query * sizeof(char));
    // cudaMalloc((void **)&d_word_indices, (total_words + 1) * sizeof(int));
    // cudaMalloc((void **)&d_query_results, (ceil(total_words / 32.0) * sizeof(uint32_t)));


    // err = cudaGetLastError();

    // if ( err != cudaSuccess )
    // {
    //    printf("4.CUDA Error: %s\n", cudaGetErrorString(err));       

    //    // Possibly: exit(-1) if program cannot continue....
    // }

    // cudaMemcpy(d_words_to_query, words_to_query, len_words_to_query * sizeof(char), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_word_indices, word_indices, (total_words + 1) * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_query_results, query_results, (ceil(total_words / 32.0) * sizeof(uint32_t)), cudaMemcpyHostToDevice);

    // err = cudaGetLastError();

    // if ( err != cudaSuccess )
    // {
    //    printf("5.CUDA Error: %s\n", cudaGetErrorString(err));       

    //    // Possibly: exit(-1) if program cannot continue....
    // }

    // // printf("Before Query Kernel: %d, %d, %d\n", total_words, ((total_words + 1023) / 1024), 1024);

    // query_bloom_kernel<<<((total_words + 1023) / 1024), 1024>>>(d_words_to_query, len_words_to_query, d_word_indices, d_bloom_filter, d_query_results, total_words);

    // err = cudaGetLastError();

    // if ( err != cudaSuccess )
    // {
    //    printf("6.CUDA Error: %s\n", cudaGetErrorString(err));       

    //    // Possibly: exit(-1) if program cannot continue....
    // }

    // cudaMemcpy(query_results, d_query_results, (ceil(total_words / 32.0) * sizeof(uint32_t)), cudaMemcpyDeviceToHost);

    // printf("\n== After Querying Words from Bloom Filter ==\n");

    // print_query_results(words_to_query, len_words_to_query, word_indices, query_results, total_words);

    // cudaFree(d_words_to_query);
    // cudaFree(d_word_indices);
    // cudaFree(d_query_results);
    // cudaFree(d_bloom_filter);

    return 0;
}