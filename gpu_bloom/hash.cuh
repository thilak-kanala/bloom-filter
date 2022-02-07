#include "defines.cuh"

#include "xxhash32.cuh"
#include "md5.cu"
#include "sha1.cu"
#include "normal_hash.cu"

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

/* Some more hash functions */
__device__
uint32_t
FNVHash(char* str, uint32_t length) {
	const unsigned int fnv_prime = 0x811C9DC5;
	unsigned int hash = 0;
	unsigned int i = 0;

	for (i = 0; i < length; str++, i++)
	{
		hash *= fnv_prime;
		hash ^= (*str);
	}

	return hash;
}

__device__
uint32_t
JSHash(char* str, uint32_t length) {
	unsigned int hash = 1315423911;
	unsigned int i = 0;

	for (i = 0; i < length; str++, i++)
	{
		hash ^= ((hash << 5) + (*str) + (hash >> 2));
	}

	return hash;
}

__device__
uint32_t
BPHash(char* str, uint32_t length) {
	unsigned int hash = 0;
	unsigned int i = 0;

	for (i = 0; i < length; str++, i++)
	{
		hash = hash << 7 ^ (*str);
	}

	return hash;
}

__device__
uint32_t
SDBMHash(char* str, uint32_t length) {
	unsigned int hash = 0;
	unsigned int i = 0;

	for (i = 0; i < length; str++, i++)
	{
		hash = (*str) + (hash << 6) + (hash << 16) - hash;
	}

	return hash;
}

__device__
uint32_t
ELFHash(char* str, uint32_t length) {
	unsigned int hash = 0;
	unsigned int x = 0;
	unsigned int i = 0;

	for (i = 0; i < length; str++, i++)
	{
		hash = (hash << 4) + (*str);

		if ((x = hash & 0xF0000000L) != 0)
		{
			hash ^= (x >> 24);
		}

		hash &= ~x;
	}

	return hash;
}

__device__
uint32_t
PJWHash(char* str, uint32_t length) {
	const unsigned int BitsInUnsignedInt = (unsigned int)(sizeof(unsigned int) * 8);
	const unsigned int ThreeQuarters = (unsigned int)((BitsInUnsignedInt * 3) / 4);
	const unsigned int OneEighth = (unsigned int)(BitsInUnsignedInt / 8);
	const unsigned int HighBits = (unsigned int)(0xFFFFFFFF) << (BitsInUnsignedInt - OneEighth);
	unsigned int hash = 0;
	unsigned int test = 0;
	unsigned int i = 0;

	for (i = 0; i < length; str++, i++)
	{
		hash = (hash << OneEighth) + (*str);

		if ((test = hash & HighBits) != 0)
		{
			hash = ((hash ^ (test >> ThreeQuarters)) & (~HighBits));
		}
	}

	return hash;
}

__device__
uint32_t
BKDRHash(char* str, uint32_t length) {
	unsigned int seed = 131;
	unsigned int hash = 0;
	unsigned int i = 0;

	for (i = 0; i < length; str++, i++)
	{
		hash = (hash * seed) + (*str);
	}

	return hash;
}

/* === Murmur hash BEGIN === */
/* Reference - https://github.com/jwerle/murmurhash.c/blob/master/murmurhash.c */
__device__
uint32_t
murmurhash (const char *key, uint32_t len, uint32_t seed) {
  uint32_t c1 = 0xcc9e2d51;
  uint32_t c2 = 0x1b873593;
  uint32_t r1 = 15;
  uint32_t r2 = 13;
  uint32_t m = 5;
  uint32_t n = 0xe6546b64;
  uint32_t h = 0;
  uint32_t k = 0;
  uint8_t *d = (uint8_t *) key; // 32 bit extract from `key'
  const uint32_t *chunks = NULL;
  const uint8_t *tail = NULL; // tail - last 8 bytes
  int i = 0;
  int l = len / 4; // chunk length

  h = seed;

  chunks = (const uint32_t *) (d + l * 4); // body
  tail = (const uint8_t *) (d + l * 4); // last 8 byte chunk of `key'

  // for each 4 byte chunk of `key'
  for (i = -l; i != 0; ++i) {
    // next 4 byte chunk of `key'
    k = chunks[i];

    // encode next 4 byte chunk of `key'
    k *= c1;
    k = (k << r1) | (k >> (32 - r1));
    k *= c2;

    // append to hash
    h ^= k;
    h = (h << r2) | (h >> (32 - r2));
    h = h * m + n;
  }

  k = 0;

  // remainder
  switch (len & 3) { // `len % 4'
    case 3: k ^= (tail[2] << 16);
    case 2: k ^= (tail[1] << 8);

    case 1:
      k ^= tail[0];
      k *= c1;
      k = (k << r1) | (k >> (32 - r1));
      k *= c2;
      h ^= k;
  }

  h ^= len;

  h ^= (h >> 16);
  h *= 0x85ebca6b;
  h ^= (h >> 13);
  h *= 0xc2b2ae35;
  h ^= (h >> 16);

  return h;
}

/* === Murmur hash END === */

/* 
    value of hash changes based on k
*/
__device__
uint32_t hash(char *string, uint32_t string_len, int hash_function)
{
    if (hash_function == XXHASH32)
    {
        uint32_t hash = XXH32(string, string_len, 0);

        return hash;
    }
    else if (hash_function == DJB2)
    {
        uint32_t hash = djb2(string, string_len);

        return hash;
    }
    else if (hash_function == JENKINS)
    {
        uint32_t hash = jenkins(string, string_len);

        return hash;
    }
    else if (hash_function == APHASH)
    {
        uint32_t hash = Normal_APHash((unsigned char*) string, string_len);

        return hash;
    }
    else if (hash_function == SHA1)
    {
    	uint32_t shaHash[HASHSIZE_SHA];
        sha1((unsigned char*) string, string_len, (unsigned char*) shaHash);
        return shaHash[0];
    }
    else if (hash_function == MD5)
    {
        uint32_t md5Hash[HASHSIZE_MD5];
        md5((unsigned char*) string, string_len, (unsigned char*) md5Hash);
        return md5Hash[0];
    }
    else if (hash_function == MURMUR)
    {
        // TODO: 
        uint32_t hash = murmurhash((const char*) string, string_len, 0);

        return hash;
    }
    else if (hash_function == FNV)
    {
        // TODO: 
        uint32_t hash = FNVHash(string, string_len);

        return hash;
    }
    else if (hash_function == JS)
    {
        // TODO: 
        uint32_t hash = JSHash(string, string_len);

        return hash;
    }
    else if (hash_function == BP)
    {
        // TODO: 
        uint32_t hash = BPHash(string, string_len);

        return hash;
    }
    else if (hash_function == SDBM)
    {
        // TODO: 
        uint32_t hash = SDBMHash(string, string_len);

        return hash;
    }
    else if (hash_function == ELF)
    {
        // TODO: 
        uint32_t hash = ELFHash(string, string_len);

        return hash;
    }
    else if (hash_function == PJW)
    {
        // TODO: 
        uint32_t hash = PJWHash(string, string_len);

        return hash;
    }
    else if (hash_function == BKDR)
    {
        // TODO: 
        uint32_t hash = BKDRHash(string, string_len);

        return hash;
    }    
    {
        printf("Not a valid hash: %d\n", hash_function);
        return 0;
    }
}