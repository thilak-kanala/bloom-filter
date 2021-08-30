#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "xxhash-ref.h"
#include "bit_vector.h"

#define BLOOM_FILTER_SIZE 335477044
#define N_HASHES 23
// #define N_WORDS_TO_INSERT 10
// #define N_WORDS_TO_QUERY 10
#define FILE_WORDS_TO_INSERT "random_strings.txt"
#define FILE_WORDS_TO_QUERY "random_strings.txt"

#define MILLION 1000000.0

int N_WORDS_TO_INSERT = 0;
int N_WORDS_TO_QUERY = 0;

// uint32_t query_results[N_WORDS_TO_QUERY];

void print_binary_64(uint64_t n, const char *message)
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

void print_binary_32(uint64_t n, const char *message)
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
void split_hash_bits(uint64_t hash, uint64_t *h1, uint64_t *h2)
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

void print_bloom_filter(uint32_t *bloom_filter)
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

void print_query_results(char **words_to_query, uint32_t *query_results)
{
    int contains = 0;
    printf("\n");
    for (int i = 0; i < N_WORDS_TO_QUERY; i++)
    {
        // Find word start and end indices
        if (test_bit(query_results, i))
        {
            contains++;
            // printf("%s: present\n", words_to_query[i]);
        }
        else
        {
            // printf("%s: absent\n", words_to_query[i]);
        }
    }

    printf("Query Result: %d / %d are present\n\n", contains, N_WORDS_TO_QUERY);
}

void generate_words_to_insert(char **words_to_insert)
{
    FILE *fh;
    fh = fopen(FILE_WORDS_TO_INSERT, "r");

    for (int i = 0; i < N_WORDS_TO_INSERT; i++)
    {
        fscanf(fh, "%s", words_to_insert[i]);
    }
}

void generate_words_to_query(char *words_to_query[256])
{
    FILE *fh;
    fh = fopen(FILE_WORDS_TO_QUERY, "r");

    for (int i = 0; i < N_WORDS_TO_QUERY; i++)
    {
        fscanf(fh, "%s", words_to_query[i]);
    }
}

void print_info()
{
    printf("\n---\n");
    printf("Number of items to insert: %d\n", N_WORDS_TO_INSERT);
    printf("Number of bits in the Bloom Filter: %d\n", BLOOM_FILTER_SIZE);
    printf("Number of hash functions: %d\n", N_HASHES);
    printf("---\n");
}

void insert_words_bloom_filter(char **words_to_insert, uint32_t *bloom_filter)
{
    for (int i = 0; i < N_WORDS_TO_INSERT; i++)
    {
        uint32_t index = 0;

        // uint64_t hash = XXH64((uint8_t *)words_to_insert[i], strlen(words_to_insert[i]), 0);
        uint64_t hash = XXH32((uint8_t *)words_to_insert[i], strlen(words_to_insert[i]), 0);

        uint64_t h1;
        uint64_t h2;

        split_hash_bits(hash, &h1, &h2);

        for (uint32_t j = 0; j < N_HASHES; j++)
        {
            h1 += (h2 * j);
            h1 = h1 % BLOOM_FILTER_SIZE;
            index = (uint32_t)h1;

            // set index bit
            set_bit(bloom_filter, index);
        }
    }
}

void query_words_bloom_filter(char **words_to_query, uint32_t *bloom_filter, uint32_t *query_results)
{
    for (int i = 0; i < N_WORDS_TO_QUERY; i++)
    {
        int is_present = 1;
        uint32_t index = 0;

        // uint64_t hash = XXH64((uint8_t *)words_to_query[i], strlen(words_to_query[i]), 0);
        uint64_t hash = XXH32((uint8_t *)words_to_query[i], strlen(words_to_query[i]), 0);

        uint64_t h1;
        uint64_t h2;

        split_hash_bits(hash, &h1, &h2);

        for (uint32_t j = 0; j < N_HASHES; j++)
        {
            h1 += (h2 * j);
            h1 = h1 % BLOOM_FILTER_SIZE;
            index = (uint32_t)h1;

            // set index bit
            if (test_bit(bloom_filter, index) == 0)
            {
                is_present = 0;
                break;
            }
        }

        if (is_present == 1)
        {
            set_bit(query_results, i);
        }
    }
}

// uint64_t hash = XXH64(data, n_bytes, 0);
int main(int argc, char const *argv[])
{
    N_WORDS_TO_INSERT = atoi(argv[1]);
    N_WORDS_TO_QUERY = atoi(argv[2]);

    char **words_to_insert = (char **)malloc(N_WORDS_TO_INSERT * sizeof(char *));
    for (int i = 0; i < N_WORDS_TO_INSERT; i++)
        words_to_insert[i] = (char *)malloc(256 * sizeof(char));

    char **words_to_query = (char **)malloc(N_WORDS_TO_QUERY * sizeof(char *));
    for (int i = 0; i < N_WORDS_TO_QUERY; i++)
        words_to_query[i] = (char *)malloc(256 * sizeof(char));


    /* === INSERT === */
    generate_words_to_insert(words_to_insert);

    uint32_t *bloom_filter = (uint32_t *)calloc(ceil(BLOOM_FILTER_SIZE / 32.0), sizeof(uint32_t));

    print_info();

    // printf("sample of words to insert: \n");
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("%s\n", words_to_insert[i]);
    // }

    // printf("\n == Before Inserting Words ==\n");
    // print_bloom_filter(bloom_filter);

    struct timespec begin, end;
    clock_gettime(CLOCK_REALTIME, &begin);

    insert_words_bloom_filter(words_to_insert, bloom_filter);

    clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds * 1e9 + nanoseconds;

    // printf("\n == Inserting words using cpu took %.3f nanoseconds ===", elapsed);

    // printf("\n == After Mapping Words to Bloom Filter ==\n");
    // print_bloom_filter(bloom_filter);

    /* === QUERY === */

    generate_words_to_query(words_to_query);

    uint32_t *query_results = (uint32_t *)calloc(N_WORDS_TO_QUERY, sizeof(uint32_t));

    clock_gettime(CLOCK_REALTIME, &begin);

    query_words_bloom_filter(words_to_query, bloom_filter, query_results);

    clock_gettime(CLOCK_REALTIME, &end);
    seconds = end.tv_sec - begin.tv_sec;
    nanoseconds = end.tv_nsec - begin.tv_nsec;
    elapsed = seconds*1e9 + nanoseconds;

    // printf("\n == Querying words using cpu took %.3f nanoseconds ===", elapsed);

    // printf("\n== After Querying Words from Bloom Filter ==\n");

    print_query_results(words_to_query, query_results);

    return 0;
}
