/* Includes */
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include "kernels.cuh"

/* Globals and Defines*/
#define INPUT_WORDS_FILE "random_strings_2.txt"
#define QUERY_WORDS_FILE "random_strings_2.txt"
#define BLOOM_FILTER_FILE "bloom_filter.txt"
// #define WORDS_FILE "../input/bloom-filter-t/random_strings_2.txt"

#define MAX_WORD_BYTES 128
#define FALSE_PROBABILITY 1.0E-4

int N_WORDS_TO_INSERT;

/* Miscellaneous */
void print_gpu_info()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
}

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

// Splits the 32bit hash into 2 16 bit hashes h1 and h2 
__device__ __host__ void split_hash_bits_32(uint32_t hash, uint32_t *h1, uint32_t *h2)
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

/* Miscellaneous - END*/

/* Input and Query Data*/
struct INPUT_DATA
{
    char *words;
    int *spaces;
};

void prepare_input_data(struct INPUT_DATA input_data, const char* FILE_LOCATION)
{
    FILE *fptr;
    fptr = fopen(FILE_LOCATION, "r");

    if (fptr == NULL)
    {
        printf("Error opening file %s\n", FILE_LOCATION);
        return;
    }

    // strcat(words, " ");
    char word[MAX_WORD_BYTES];

    int spaces_index = -1;
    input_data.spaces[++spaces_index] = 0;

    for (int i = 0; i < N_WORDS_TO_INSERT; i++)
    {
        fscanf(fptr, "%s", word);

        strcat(input_data.words, word);
        strcat(input_data.words, " ");

        input_data.spaces[spaces_index + 1] = input_data.spaces[spaces_index] + strlen(word) + 1;
        spaces_index++;
    }
    strcat(input_data.words, "\0");
}

void print_input_data(struct INPUT_DATA input_data)
{
    printf("=== Input Data ===\n");
    printf("Words: \n");
    printf("%s\n", input_data.words);

    printf("Spaces: \n");
    for (int i = 0; i < N_WORDS_TO_INSERT + 1; i++)
    {
        printf("%d, ", input_data.spaces[i]);
    }
    printf("\n\n");
}
/* Input -- END */

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
    for (int i = 0; i < bf.n; i++)
    {
        if (test_bit(bf.bf, i))
        {
            bits_set++;
        }
    }

    bf->bits_set = bits_set;

    printf("Number of bits set: %d\n", bf->bits_set);

    printf("\n");
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
    fprintf(fptr, "False Positive Probability:%f\n", bf->p);
    fprintf(fptr, "Number Of Hash Functions:%d\n", bf->k);
    fprintf(fptr, "Number Of Bits Set:%d\n", bf->bits_set);

    for (int i = 0; i < bf->m; i++)
    {
        int bit = test_bit(bf->bf, i);
        fprintf(fptr, "%d", bit);
    }
    fprintf(fptr, "\n");
}

// TODO: define function read_bloom_filter(struct BLOOM_FILTER bf, const char *BLOOM_FILTER_FILE)

/* Bloom Filter - END*/

/* Hash Functions */
__device__
uint32_t
djb2(char *string, uint32_t string_len)
{
    uint32_t hash = 5381;
    char c;
    uint32_t i = 0;
    while (i++ < string_len)
    {
        c = *string++;
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

/* Hash Functions - END */

