#include <math.h>

/* === BIT VECTOR BEGIN ===*/

// sets the kth bit in bitvector 'bv'
void set_bit(uint32_t *bv, uint32_t k)
{
    bv[k / 32] |= (1 << (k % 32));
}

// clears the kth bit in bitvector 'bv'
void clear_bit(uint32_t *bv, uint32_t k)
{
    bv[k / 32] &= ~(1 << (k % 32));
}

int test_bit(uint32_t *bv, uint32_t k)
{
    return ((bv[k / 32] & (1 << (k % 32))) != 0);
}

/* === BIT VECTOR END ===*/
