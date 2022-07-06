//-----------------------------------------------
//       Created by Siddhant Kadwe
//       -Generates an array of size 64
//        and squares the elements and
//        saves it in another array
//-----------------------------------------------

#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void square(float *d_out, float *d_in) {
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f;
}

int main(int argc, char ** argv) {
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    // declare GPU memory pointers
    float * d_in;
    float * d_out;

    // allocate GPU memory
    gpuErrchk(cudaMalloc((void **) &d_in, ARRAY_BYTES));
    gpuErrchk(cudaMalloc((void **) &d_out, ARRAY_BYTES));

    // transfer the array to the GPU
    gpuErrchk(cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice));

    // launch the kernel
    (square<<<1, ARRAY_SIZE>>>(d_out, d_in));

    // copy back the result array to the CPU
    gpuErrchk(cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost));

    // print out the resulting array
    for(int i = 0; i < ARRAY_SIZE; i++) {
        printf("%f", h_out[i]);
        printf(((i%4)!=3) ? "\t" : "\n");
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}