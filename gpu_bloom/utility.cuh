/* === DEFINES === */
#define BLOOM_FILTER_SIZE 335477044
#define N_HASHES 23
#define FILE_WORDS_TO_INSERT "words_insert.txt"
#define FILE_WORDS_TO_QUERY "words_query.txt"


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

char *read_from_file(const char *file_location)
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