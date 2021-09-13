#include "utility.h"

int main(int argc, char const *argv[])
{
    N_WORDS_TO_INSERT = atoi(argv[1]);
    // N_WORDS_TO_QUERY = atoi(argv[2]);

    char **words_to_insert = (char **)malloc(N_WORDS_TO_INSERT * sizeof(char *));
    for (int i = 0; i < N_WORDS_TO_INSERT; i++)
        words_to_insert[i] = (char *)malloc(MAX_WORD_BYTES * sizeof(char));

    // char **words_to_query = (char **)malloc(N_WORDS_TO_QUERY * sizeof(char *));
    // for (int i = 0; i < N_WORDS_TO_QUERY; i++)
    //     words_to_query[i] = (char *)malloc(MAX_WORD_BYTES * sizeof(char));


    /* === INSERT === */
    // generate_words_to_insert(words_to_insert);
    prepare_input_data(words_to_insert, N_WORDS_TO_INSERT, FILE_WORDS_TO_INSERT);

    // uint32_t *bloom_filter = (uint32_t *)calloc(ceil(BLOOM_FILTER_SIZE / 32.0), sizeof(uint32_t));
    /* Bloom Filter */
    struct BLOOM_FILTER bf;
    prepare_bloom_filter(&bf, N_WORDS_TO_INSERT, FALSE_PROBABILITY);
    print_bloom_filter(&bf);

    // print_info();

    // printf("sample of words to insert: \n");
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("%s\n", words_to_insert[i]);
    // }

    // printf("\n == Before Inserting Words ==\n");
    // print_bloom_filter(bloom_filter);

    // struct timespec begin, end;
    // clock_gettime(CLOCK_REALTIME, &begin);

    insert_words_bloom_filter(words_to_insert, &bf);

    print_bloom_filter(&bf);

    write_bloom_filter(&bf, BLOOM_FILTER_FILE);

    // clock_gettime(CLOCK_REALTIME, &end);
    // long seconds = end.tv_sec - begin.tv_sec;
    // long nanoseconds = end.tv_nsec - begin.tv_nsec;
    // double elapsed = seconds * 1e9 + nanoseconds;

    // printf("\n == Inserting words using cpu took %.3f nanoseconds ===", elapsed);

    // printf("\n == After Mapping Words to Bloom Filter ==\n");
    // print_bloom_filter(bloom_filter);

    // /* === QUERY === */

    // generate_words_to_query(words_to_query);

    // uint32_t *query_results = (uint32_t *)calloc(N_WORDS_TO_QUERY, sizeof(uint32_t));

    // // clock_gettime(CLOCK_REALTIME, &begin);

    // query_words_bloom_filter(words_to_query, bloom_filter, query_results);

    // // clock_gettime(CLOCK_REALTIME, &end);
    // // seconds = end.tv_sec - begin.tv_sec;
    // // nanoseconds = end.tv_nsec - begin.tv_nsec;
    // // elapsed = seconds*1e9 + nanoseconds;

    // // printf("\n == Querying words using cpu took %.3f nanoseconds ===", elapsed);

    // // printf("\n== After Querying Words from Bloom Filter ==\n");

    // print_query_results(words_to_query, query_results);

    return 0;
}
