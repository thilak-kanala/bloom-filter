/*  Reference - https://drewdevault.com/2016/04/12/How-to-write-a-better-bloom-filter-in-C.html 
    How to write a better bloom filter in C
*/

#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "bloom.h"

struct bloom_filter
{
    void *bits;
    size_t size;
};

/* Hash Functions */

unsigned int djb2(const void *_str)
{
    const char *str = _str;
    unsigned int hash = 5381;
    char c;
    while ((c = *str++))
    {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

unsigned int jenkins(const void *_str)
{
    const char *key = _str;
    unsigned int hash, i;
    while (*key)
    {
        hash += *key;
        hash += (hash << 10);
        hash ^= (hash >> 6);
        key++;
    }
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
}

/* Bloom Filter */

bloom_t bloom_create(size_t size)
{
    bloom_t res = calloc(1, sizeof(struct bloom_filter));
    res->size = size;
    res->bits = malloc(size);
    return res;
}

void bloom_free(bloom_t filter)
{
    if (filter)
    {
        free(filter->bits);
        free(filter);
    }
}

void bloom_add(bloom_t filter, const void *item)
{
    uint8_t *bits = filter->bits;
    unsigned int hash;

    // djb2
    hash = djb2(item);
    hash %= filter->size * 8;
    bits[hash / 8] |= 1 << hash % 8;

    // jenkins
    hash = jenkins(item);
    hash %= filter->size * 8;
    bits[hash / 8] |= 1 << hash % 8;
}

bool bloom_test(bloom_t filter, const void *item)
{
    uint8_t *bits = filter->bits;
    uint8_t result = 0;
    unsigned int hash;

    // djb2
    hash = djb2(item);
    hash %= filter->size * 8;
    result = (bits[hash / 8] & 1 << hash % 8);

    // jenkins
    hash = djb2(item);
    hash %= filter->size * 8;
    result &= (bits[hash / 8] & 1 << hash % 8);

    return result;
}
