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

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("1.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMalloc((void **)&d_input_data, sizeof(input_data));

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("2.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(d_input_data, input_data, sizeof(input_data), cudaMemcpyHostToDevice);

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("3.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    kernel<<<2, 32>>>(d_input_data);

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("4.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    cudaMemcpy(input_data, d_input_data, sizeof(input_data), cudaMemcpyDeviceToHost);

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("5.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }    

    cudaFree(d_input_data);

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("6.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    

    return 0;
}