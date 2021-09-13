/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "xxhash-ref.h"
#include "bit_vector.h"

/* Defines and Globals */
#define BLOOM_FILTER_SIZE 335477044
// #define N_HASHES 23
#define MAX_WORD_BYTES 128

// #define N_WORDS_TO_INSERT 10
// #define N_WORDS_TO_QUERY 10
#define FILE_WORDS_TO_INSERT "random_strings.txt"
#define FILE_WORDS_TO_QUERY "random_strings.txt"

#define BLOOM_FILTER_FILE "bloom_filter.txt"

#define FALSE_PROBABILITY 1.0E-4
#define MILLION 1000000.0

int N_WORDS_TO_INSERT = 0;
int N_WORDS_TO_QUERY = 0;

/* Miscellaneous */
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

// Splits the 32bit hash into 2 16 bit hashes h1 and h2 
void split_hash_bits_32(uint32_t hash, uint32_t *h1, uint32_t *h2)
{
    uint32_t mask;
    uint32_t one_32bit = 1;

    // Clear all bits
    mask = 0;

    // Create mask to extract bottom 16 bits
    for (int i = 0; i < 16; i++)
    {
        mask |= (one_32bit << i);
    }

    // Extract bottom 16 bits
    *h1 = hash & mask;

    // Create mask to extract top 16 bits
    for (int i = 16; i < 32; i++)
    {
        mask |= (one_32bit << i);
    }

    // Extract top 16 bits
    *h2 = (hash & mask) >> 16;
}

/* Input Data */
void prepare_input_data(char **words, int n_words, const char *file_location)
{
    FILE *fh;
    fh = fopen(file_location, "r");

    if (fh == NULL)
    {
        printf("Error opening file %s\n", file_location);
        return;
    }

    for (int i = 0; i < n_words; i++)
    {
        fscanf(fh, "%s", words[i]);
    }
}

// void generate_words_to_insert(char **words_to_insert)
// {
//     FILE *fh;
//     fh = fopen(FILE_WORDS_TO_INSERT, "r");

//     if (fh == NULL)
//     {
//         printf("Error opening file %s\n", FILE_WORDS_TO_INSERT);
//         return;
//     }

//     for (int i = 0; i < N_WORDS_TO_INSERT; i++)
//     {
//         fscanf(fh, "%s", words_to_insert[i]);
//     }
// }

// void generate_words_to_query(char **words_to_query)
// {
//     FILE *fh;
//     fh = fopen(FILE_WORDS_TO_QUERY, "r");

//     for (int i = 0; i < N_WORDS_TO_QUERY; i++)
//     {
//         fscanf(fh, "%s", words_to_query[i]);
//     }
// }

/* Query */
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

/* Bloom Filter */
struct BLOOM_FILTER
{
    uint32_t *bf; // bloom filter
    int m; // number of bits in the bloom filter
    int n; // number of items in the bloom filter
    float p; // probability of false positives
    int k; // number of hash functions

    int bits_set; // number of bits set
};

void prepare_bloom_filter(struct BLOOM_FILTER *bf, int n, float p)
{
    bf->n = n;
    bf->p = p;
    bf->m = ceil((bf->n * log(bf->p)) / log(1 / pow(2, log(2))));
    bf->k = round((bf->m / bf->n) * log(2));

    bf->bf = (uint32_t *)calloc(ceil(bf->m / 32.0), sizeof(uint32_t));

    bf->bits_set = 0;
}

void print_bloom_filter(struct BLOOM_FILTER *bf)
{
    printf("=== Bloom Filter ===\n");
    printf("Number of items: %d\n", bf->n);
    printf("False Proabability: %E\n", bf->p);
    printf("Number of bits: %d\n", bf->m);
    printf("Number of hash functions: %d\n", bf->k);

    uint32_t bits_set = 0;
    for (int i = 0; i < bf->n; i++)
    {
        if (test_bit(bf->bf, i))
        {
            bits_set++;
        }
    }

    bf->bits_set = bits_set;

    printf("Number of bits set: %d\n", bf->bits_set);

    printf("\n");
}

void insert_words_bloom_filter(char **words_to_insert, struct BLOOM_FILTER *bf)
{
    for (int i = 0; i < bf->n; i++)
    {
        uint32_t index = 0;

        // uint64_t hash = XXH64((uint8_t *)words_to_insert[i], strlen(words_to_insert[i]), 0);
        uint32_t hash = XXH32((uint8_t *)words_to_insert[i], strlen(words_to_insert[i]), 0);

        uint32_t h1;
        uint32_t h2;

        split_hash_bits_32(hash, &h1, &h2);

        for (uint32_t j = 0; j < bf->k; j++)
        {
            h1 += (h2 * j);
            h1 = h1 % bf->m;
            index = (uint32_t)h1;

            // set index bit
            set_bit(bf->bf, index);
        }
    }
}

void query_words_bloom_filter(char **words_to_query, struct BLOOM_FILTER *bf, uint32_t *query_results)
{
    for (int i = 0; i < bf->n; i++)
    {
        int is_present = 1;
        uint32_t index = 0;

        // uint64_t hash = XXH64((uint8_t *)words_to_query[i], strlen(words_to_query[i]), 0);
        uint32_t hash = XXH32((uint8_t *)words_to_query[i], strlen(words_to_query[i]), 0);

        uint32_t h1;
        uint32_t h2;

        split_hash_bits_32(hash, &h1, &h2);

        for (uint32_t j = 0; j < bf->k; j++)
        {
            h1 += (h2 * j);
            h1 = h1 % bf->m;
            index = (uint32_t)h1;

            // set index bit
            if (test_bit(bf->bf, index) == 0)
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

void write_bloom_filter(struct BLOOM_FILTER *bf, const char *bloom_filter_file)
{
    FILE *fptr;
    fptr = fopen(bloom_filter_file, "w");

    if (fptr == NULL)
    {
        printf("Error opening file %s\n", bloom_filter_file);
        return;
    }

    fprintf(fptr, "Number Of Bits:%d\n", bf->m);
    fprintf(fptr, "Number Of Items:%d\n", bf->n);
    fprintf(fptr, "False Positive Probability:%E\n", bf->p);
    fprintf(fptr, "Number Of Hash Functions:%d\n", bf->k);
    fprintf(fptr, "Number Of Bits Set:%d\n", bf->bits_set);

    for (int i = 0; i < bf->m; i++)
    {
        int bit = test_bit(bf->bf, i);
        fprintf(fptr, "%d", bit);
    }
    fprintf(fptr, "\n");
}

void read_bloom_filter(struct BLOOM_FILTER *bf, const char *bloom_filter_file)
{
    FILE *fptr;
    fptr = fopen(bloom_filter_file, "r");

    if (fptr == NULL)
    {
        printf("Error opening file %s\n", bloom_filter_file);
        return;
    }

    char line[256];
    fscanf(fptr, "%s", line);
    printf("%s\n", line);

    // fprintf(fptr, "Number Of Bits:%d\n", bf->m);
    // fprintf(fptr, "Number Of Items:%d\n", bf->n);
    // fprintf(fptr, "False Positive Probability:%E\n", bf->p);
    // fprintf(fptr, "Number Of Hash Functions:%d\n", bf->k);
    // fprintf(fptr, "Number Of Bits Set:%d\n", bf->bits_set);

    // for (int i = 0; i < bf->m; i++)
    // {
    //     int bit = test_bit(bf->bf, i);
    //     fprintf(fptr, "%d", bit);
    // }
    // fprintf(fptr, "\n");
}
