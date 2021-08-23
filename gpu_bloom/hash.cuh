#include "defines.cuh"

/*
Hash lookup table
1: XXHASH64
/*

/* Splits the 64bit hash into 2 32 bit hashes h1 and h2 */
__device__ __host__ void split_hash_bits(uint64_t hash, uint32_t *h1, uint32_t *h2)
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

/* Splits the 32bit hash into 2 16 bit hashes h1 and h2 */
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

/* === General Purpose Hash Functions BEGIN ===*/
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

__device__
uint32_t
jenkins(char *string, uint32_t string_len)
{
    uint32_t hash = 5381;
    uint32_t i = 0;
    while (i < string_len)
    {
        hash += *string++;
        hash += (hash << 10);
        hash ^= (hash >> 6);
        i++;
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

/* === General Purpose Hash Functions END ===*/


/* 
    value of hash changes based on k
*/
__device__
uint32_t hash(char *string, uint32_t string_len, int hash, int k)
{
    if (hash == 1)
    {
        uint64_t hash = XXH64(string, string_len, 0);
        uint32_t h1;
        uint32_t h2;
    
        split_hash_bits(hash, &h1, &h2);

        h1 += (h2 * k);
        h1 = h1 % BLOOM_FILTER_SIZE;

        return h1;
    }
    else if (hash == 2)
    {
        uint32_t hash = djb2(string, string_len);
        uint32_t h1;
        uint32_t h2;
    
        split_hash_bits_32(hash, &h1, &h2);

        h1 += (h2 * k);
        h1 = h1 % BLOOM_FILTER_SIZE;

        return h1;
    }
    else if (hash == 3)
    {
        uint32_t hash = jenkins(string, string_len);
        uint32_t h1;
        uint32_t h2;
    
        split_hash_bits_32(hash, &h1, &h2);

        h1 += (h2 * k);
        h1 = h1 % BLOOM_FILTER_SIZE;

        return h1;
    }
    else 
    {
        printf("Not a valid hash: %d\n", hash);
    }

    return 0;
}