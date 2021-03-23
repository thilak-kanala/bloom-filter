#include "bloom.h"
#include <stdio.h>

int main(int argc, char const *argv[])
{
    int result[10];

    bloom_t bloom = bloom_create(8);

    result[0] = bloom_test(bloom, "ramesh");
    result[1] = bloom_test(bloom, "suresh");
    result[2] = bloom_test(bloom, "thilak");
    result[3] = bloom_test(bloom, "vaibhav");

    for (int i = 0; i < 4; i++)
    {
        printf("%d, ", result[i]);
    }
    printf("\n");

    bloom_add(bloom, "ramesh");
    bloom_add(bloom, "suresh");
    bloom_add(bloom, "thilak");
    bloom_add(bloom, "vaibhav");


    result[0] = bloom_test(bloom, "ramesh");
    result[1] = bloom_test(bloom, "suresh");
    result[2] = bloom_test(bloom, "thilak");
    result[3] = bloom_test(bloom, "vaibhav");

    for (int i = 0; i < 4; i++)
    {
        printf("%d, ", result[i]);
    }
    printf("\n");


    return 0;
}
