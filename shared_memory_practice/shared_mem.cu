#include "utility.cuh"

__global__ void kernel(struct INPUT_DATA *input_data)
{
    int tid = threadIdx.x;
    printf("%d:%d\n", tid, input_data->spaces[0]);
}

int main(int argc, char const *argv[])
{
    cudaFree(0);

    print_gpu_info();

    N_WORDS_TO_INSERT = atoi(argv[1]);

    /* Input */
    INPUT_DATA *input_data = (struct INPUT_DATA *)malloc(sizeof(struct INPUT_DATA));
    input_data->words = (char *)malloc((N_WORDS_TO_INSERT + 1) * 256 * sizeof(char));
    input_data->words[0] = ' ';
    input_data->words[1] = '\0';
    input_data->spaces = (int *)malloc((N_WORDS_TO_INSERT + 2) * sizeof(int));

    prepare_input_data(input_data);
    print_input_data(input_data);

    /* Bloom Filter */
    BLOOM_FILTER bf;
    prepare_bloom_filter(&bf, N_WORDS_TO_INSERT, FALSE_PROBABILITY);
    print_bloom_filter(&bf);

    /* GPU */
    INPUT_DATA *d_input_data = NULL;

    cudaMalloc((void **)d_input_data, sizeof(input_data));
    cudaMemcpy(d_input_data, input_data, sizeof(input_data), cudaMemcpyHostToDevice);

    kernel<<<2, 32>>>(d_input_data);

    cudaMemcpy(input_data, d_input_data, sizeof(input_data), cudaMemcpyDeviceToHost);

    cudaFree(d_input_data);

    return 0;
}