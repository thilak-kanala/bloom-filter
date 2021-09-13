#include "utility.cuh"

int main(int argc, char const *argv[])
{
    cudaFree(0);

    print_gpu_info();

    N_WORDS_TO_INSERT = atoi(argv[1]);

    /* Input */
    INPUT_DATA input_data;
    input_data.words = (char *)malloc((N_WORDS_TO_INSERT + 1) * MAX_WORD_BYTES * sizeof(char));
    input_data.words[0] = ' ';
    input_data.words[1] = '\0';
    input_data.spaces = (int *)malloc((N_WORDS_TO_INSERT + 2) * sizeof(int));

    prepare_input_data(input_data, INPUT_WORDS_FILE);
    print_input_data(input_data);

    /* Bloom Filter */
    struct BLOOM_FILTER bf;
    prepare_bloom_filter(&bf, N_WORDS_TO_INSERT, FALSE_PROBABILITY);
    print_bloom_filter(&bf);

    /* GPU */
    INPUT_DATA d_input_data;
    BLOOM_FILTER d_bf;

    // Testing
    // int *h_arr = (int*) malloc(10 * sizeof(int));
    // int *d_arr;

    // cudaMalloc((void**)&d_arr, 10 * sizeof(int));
    // cudaMemcpy(d_arr, h_arr, 10*sizeof(int), cudaMemcpyHostToDevice);

    // kernel<<<1, 32>>>(d_arr);

    // cudaFree(d_arr);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("1.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMalloc((void **)&d_input_data.words, sizeof(input_data.words));
    cudaMalloc((void **)&d_input_data.spaces, sizeof(input_data.spaces));
    cudaMalloc((void **)&d_bf.bf, sizeof(bf.bf));
    cudaMalloc((void **)&d_bf.m, sizeof(bf.m));
    cudaMalloc((void **)&d_bf.n, sizeof(bf.n));
    cudaMalloc((void **)&d_bf.p, sizeof(bf.p));
    cudaMalloc((void **)&d_bf.k, sizeof(bf.k));

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("2.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(d_input_data.words, input_data.words, sizeof(input_data.words), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_data.spaces, input_data.spaces, sizeof(input_data.spaces), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bf.bf, bf.bf, sizeof(bf.bf), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_bf.m, &bf.m, sizeof(bf.m), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_bf.n, &bf.n, sizeof(bf.n), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_bf.p, &bf.p, sizeof(bf.p), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_bf.k, &bf.k, sizeof(bf.k), cudaMemcpyHostToDevice);


    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("3.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    // TODO: change dimensions
    insert_words_kernel<<<ceil(bf.n / 256.0), 256>>>(d_input_data, d_bf);

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("4.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Copy the bloom filter from GPU to CPU
    cudaMemcpy(bf.bf, d_bf.bf, sizeof(d_bf.bf), cudaMemcpyDeviceToHost);

    write_bloom_filter(&bf, BLOOM_FILTER_FILE);

    // check for error
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("5.CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }    

    cudaFree(d_input_data.words);
    cudaFree(d_input_data.spaces);
    cudaFree(d_bf.bf);
    cudaFree(d_bf.m);
    cudaFree(d_bf.n);
    cudaFree(d_bf.p);
    cudaFree(d_bf.k);

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