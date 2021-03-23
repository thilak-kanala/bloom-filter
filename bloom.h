/*  Reference - https://drewdevault.com/2016/04/12/How-to-write-a-better-bloom-filter-in-C.html 
    How to write a better bloom filter in C
*/

#ifndef _BLOOM_H
#define _BLOOM_H
#include <stddef.h>
#include <stdbool.h>

typedef unsigned int (*hash_function)(const void *data);
typedef struct bloom_filter *bloom_t;

/* Creates a new bloom filter with size = (size * 8) bits. */
bloom_t bloom_create(size_t size);

/* Frees a bloom filter. */
void bloom_free(bloom_t filter);

/* Adds an item to the bloom filter. */
void bloom_add(bloom_t filter, const void *item);

/* Tests if an item is in the bloom filter.
 *
 * Returns false if the item has definitely not been added before. Returns true
 * if the item was probably added before. */
bool bloom_test(bloom_t filter, const void *item);

#endif