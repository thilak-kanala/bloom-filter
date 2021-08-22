#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

#include <stddef.h>   /* uint32_t, NULL */
#include <stdint.h>   /* uint8_t, uint32_t, uint64_t */
#include <inttypes.h> /* print formatting */

#include "xxhash64.cuh"
#include "hash.cuh"
#include "utility.cuh"
#include "defines.cuh"

uint32_t total_words = 0;

/* Generate a hash based on The Combinatorial Approach (https://github.com/Claudenw/BloomFilter/wiki/Bloom-Filters----An-Overview) */
__device__ void bloom_insert(uint32_t *d_bloom_filter, char *word, uint32_t word_len, int tid)
{
    uint32_t index = 0;

    // uint64_t hash = XXH64((char*) word, word_len, 0);
    // uint64_t h1;
    // uint64_t h2;

    // split_hash_bits(hash, &h1, &h2);
    // split_hash_bits_32(hash, &h1, &h2);

    for (uint32_t i = 0; i < N_HASHES; i++)
    {
        // h1 += (h2 * i);
        // h1 = h1 % BLOOM_FILTER_SIZE;
        // index = h1;

        index = hash(word, word_len, 1, i);

        printf("%d: bloom_query: %d\n", tid, index);

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

    // uint64_t hash = XXH64((char*) word, word_len, 0);
    // uint64_t h1;
    // uint64_t h2;

    // split_hash_bits(hash, &h1, &h2);

    for (uint32_t i = 0; i < N_HASHES; i++)
    {
        // h1 += (h2 * i);
        // h1 = h1 % BLOOM_FILTER_SIZE;
        // index = h1;

        index = hash(word, word_len, 1, i);

        printf("%d: bloom_query: %d\n", tid, index);

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

    // char *word = (char *)malloc((ei - si + 1) * sizeof(char));
    char word[128];

    for (uint32_t i = si; si <= ei; si++)
    {
        word[si - i] = d_words_to_insert[si];
    }

    // Add word to bloom filter
    bloom_insert(d_bloom_filter, word, word_len, tid);
    // free(word);
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

    // char *word = (char *)malloc((ei - si + 1) * sizeof(char));
    char word[128];

    for (uint32_t i = si; si <= ei; si++)
    {
        // printf("tid, bid, si: %d, %d, %d\n", threadIdx.x, blockIdx.x, si);
        word[si - i] = d_words_to_query[si];
    }

    // Add word to bloom filter
    bloom_query(d_bloom_filter, word, word_len, d_query_results, tid);
    // free(word);
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
    printf("Transferring strings and bloom filter from Host to GPU Global Memory took %f ms\n", milliseconds);

    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("1.1.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    // cudaEventRecord( start, 0 );

    cudaEventRecord(start);
    map_bloom_kernel<<<ceil(total_words / 256.0), 256>>>(d_words_to_insert, len_words_to_insert, d_word_indices, d_bloom_filter, total_words);
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

    /* === Query Bloom Filter === */

    char *words_to_query = read_from_file(FILE_WORDS_TO_QUERY);
    uint32_t len_words_to_query = strlen(words_to_query);

    total_words = 0;
    for (uint32_t i = 0; i < len_words_to_query; i++)
    {
        if (words_to_query[i] == ' ')
        {
            total_words++;
        }
    }
    // To account for the space at the beginning and the end
    total_words -= 1;

    word_indices = (uint32_t *)calloc((total_words + 1), sizeof(int));

    wi = -1;
    for (uint32_t i = 0; i < len_words_to_query; i++)
    {
        if (words_to_query[i] == ' ')
        {
            word_indices[++wi] = i;
        }
    }

    uint32_t *query_results = (uint32_t *)calloc(total_words, sizeof(uint32_t));
    uint32_t *d_query_results;
    char *d_words_to_query;

    cudaEventRecord(start);

    cudaMalloc((void **)&d_words_to_query, len_words_to_query * sizeof(char));
    cudaMalloc((void **)&d_word_indices, (total_words + 1) * sizeof(int));
    cudaMalloc((void **)&d_query_results, (ceil(total_words / 32.0) * sizeof(uint32_t)));


    err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("4.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    cudaMemcpy(d_words_to_query, words_to_query, len_words_to_query * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_indices, word_indices, (total_words + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_results, query_results, (ceil(total_words / 32.0) * sizeof(uint32_t)), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Transerring Query Words to Global Memory took:  %f ms\n", milliseconds);

    err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("5.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    cudaEventRecord(start);

    query_bloom_kernel<<<ceil(total_words / 256.0), 256>>>(d_words_to_query, len_words_to_query, d_word_indices, d_bloom_filter, d_query_results, total_words);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Querying words to Bloom Filter (kernel) took:  %f ms\n", milliseconds);

    err = cudaGetLastError();

    if ( err != cudaSuccess )
    {
       printf("6.CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }

    cudaEventRecord(start);
    cudaMemcpy(query_results, d_query_results, (ceil(total_words / 32.0) * sizeof(uint32_t)), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Transferring query results to host took:  %f ms\n", milliseconds);

    printf("\n== After Querying Words from Bloom Filter ==\n");

    print_query_results(words_to_query, len_words_to_query, word_indices, query_results, total_words);

    cudaFree(d_words_to_query);
    cudaFree(d_word_indices);
    cudaFree(d_query_results);
    cudaFree(d_bloom_filter);

    return 0;
}