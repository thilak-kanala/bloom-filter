/* Includes */
#include <cuda.h>
#include <stdio.h>
#include <math.h>

/* Globals and Defines*/
// #define WORDS_FILE "random_strings_2.txt"
#define WORDS_FILE "../input/bloom-filter-t/random_strings_2.txt"

#define FALSE_PROBABILITY 1.0E-7

int N_WORDS_TO_INSERT;

/* Miscellaneous */
void print_gpu_info()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
}

/* Input */
struct INPUT_DATA
{
    char *words;
    int *spaces;
};

void prepare_input_data(struct INPUT_DATA *input_data)
{
    FILE *fptr;
    fptr = fopen(WORDS_FILE, "r");

    if (fptr == NULL)
    {
        printf("Error opening file %s\n", WORDS_FILE);
        return;
    }

    // strcat(words, " ");
    char word[256];

    int spaces_index = -1;
    input_data->spaces[++spaces_index] = 0;

    for (int i = 0; i < N_WORDS_TO_INSERT; i++)
    {
        fscanf(fptr, "%s", word);

        strcat(input_data->words, word);
        strcat(input_data->words, " ");

        input_data->spaces[spaces_index + 1] = input_data->spaces[spaces_index] + strlen(word) + 1;
        spaces_index++;
    }
    strcat(input_data->words, "\0");
}

void print_input_data(struct INPUT_DATA *input_data)
{
    printf("=== Input Data ===\n");
    printf("Words: \n");
    printf("%s\n", input_data->words);

    printf("Spaces: \n");
    for (int i = 0; i < N_WORDS_TO_INSERT + 1; i++)
    {
        printf("%d, ", input_data->spaces[i]);
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
};

void prepare_bloom_filter(struct BLOOM_FILTER *bf, int n, float p)
{
    bf->n = n;
    bf->p = p;
    bf->m = ceil((bf->n * log(bf->p)) / log(1 / pow(2, log(2))));
    bf->k = round((bf->m / bf->n) * log(2));

    bf->bf = (uint32_t *)calloc(ceil(bf->m / 32.0), sizeof(uint32_t));
}

void print_bloom_filter(struct BLOOM_FILTER *bf)
{
    printf("=== Bloom Filter ===\n");
    printf("Number of items: %d\n", bf->n);
    printf("False Proabability: %E\n", bf->p);
    printf("Number of bits: %d\n", bf->m);
    printf("Number of hash functions: %d\n", bf->k);

    printf("\n");
    
    // TODO: print #bits set in bloom filter
}

/* Bloom Filter - END*/
