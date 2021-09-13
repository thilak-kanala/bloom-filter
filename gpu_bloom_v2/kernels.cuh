__global__ void insert_words_kernel(struct INPUT_DATA input_data, struct BLOOM_FILTER bf)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= bf.n) { return; }

    // Find word start and end indices
    uint32_t word_len = input_data.spaces[tid + 1] - input_data.spaces[tid] - 1;
    uint32_t si = input_data.spaces[tid] + 1;
    uint32_t ei = input_data.spaces[tid + 1] - 1;

    if (tid == 0)
    {
        printf("MAX_WORD_BYTES: %d\n", MAX_WORD_BYTES);
    }

    char word[MAX_WORD_BYTES];
    for (uint32_t i = si; si <= ei; si++)
    {
        word[si - i] = input_data.words[si];
    }

    // Add word to bloom filter
    uint32_t index = 0;

    uint32_t hash_value;
    // TODO: replace djb2() with hash()
    hash_value = djb2(word, word_len);
    // hash_value = hash(word, word_len, HASH_FUNCTION);

    uint32_t h1;
    uint32_t h2;

    split_hash_bits_32(hash_value, &h1, &h2);

    int k = bf.k; // number of hash functions
    int n = bf.n; // number of bits in the bloom filter
    for (uint32_t i = 0; i < k; i++)
    {
        h1 += (h2 * i);
        h1 = h1 % n;
        index = h1;

        // set index bit
        atomicOr(&bf.bf[index / 32], (1 << (index % 32)));
    }
}

// __global__ void query_words_kernel(struct INPUT_DATA input_data, struct BLOOM_FILTER bf)
// {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;

//     if (tid >= bf.n) { return; }

//     // Find word start and end indices
//     uint32_t word_len = input_data.spaces[tid + 1] - input_data.spaces[tid] - 1;
//     uint32_t si = input_data.spaces[tid] + 1;
//     uint32_t ei = input_data.spaces[tid + 1] - 1;

//     if (tid == 0)
//     {
//         printf("MAX_WORD_BYTES: %d\n", MAX_WORD_BYTES);
//     }

//     char word[MAX_WORD_BYTES];
//     for (uint32_t i = si; si <= ei; si++)
//     {
//         word[si - i] = input_data.words[si];
//     }

//     // Add word to bloom filter
//     uint32_t index = 0;

//     uint32_t hash_value;
//     // TODO: replace djb2() with hash()
//     hash_value = djb2(word, word_len);
//     // hash_value = hash(word, word_len, HASH_FUNCTION);

//     uint32_t h1;
//     uint32_t h2;

//     split_hash_bits_32(hash_value, &h1, &h2);

//     int k = bf.k; // number of hash functions
//     int n = bf.n; // number of bits in the bloom filter
//     for (uint32_t i = 0; i < k; i++)
//     {
//         h1 += (h2 * i);
//         h1 = h1 % n;
//         index = h1;

//         // set index bit
//         atomicOr(&bf.bf[index / 32], (1 << (index % 32)));
//     }
// }