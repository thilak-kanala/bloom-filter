#include "bloom.h"
#include <stdio.h>

char test_data[5][100] = {
    "ramesh",
    "suresh",
    "thilak",
    "vaibhav",
    "sujith",
};

int main(int argc, char const *argv[])
{
    bloom_t bloom = bloom_create(2);

    for (int i = 0; i < 5; i++)
    {
        bloom_add(bloom, test_data[i]);
    }

    bool result[5];

    for (int i = 0; i < 5; i++)
    {
        result[i] = bloom_test(bloom, test_data[i]);
    }

    printf("results: ");
    for (int i = 0; i < 5; i++)
    {
        printf("%d, ", result[i]);
    }
    printf("\n");

    bloom_print(bloom);

    return 0;
}
