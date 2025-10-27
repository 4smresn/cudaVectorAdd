#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <omp.h>
#include <math.h>

#define START_GPU                                 \
    {                                             \
        cudaEvent_t start, stop;                  \
        float elapsedTime;                        \
        checkCudaErrors(cudaEventCreate(&start)); \
        checkCudaErrors(cudaEventCreate(&stop));  \
        checkCudaErrors(cudaEventRecord(start, 0));

#define END_GPU                                                       \
    checkCudaErrors(cudaEventRecord(stop, 0));                        \
    checkCudaErrors(cudaEventSynchronize(stop));                      \
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
    printf("GPU Time used:  %3.1f ms", elapsedTime);                \
    checkCudaErrors(cudaEventDestroy(start));                         \
    checkCudaErrors(cudaEventDestroy(stop));                          \
    }

#define START_CPU \
    {             \
        double start = omp_get_wtime();

#define END_CPU                                           \
    double end = omp_get_wtime();                         \
    double duration = end - start;                        \
    printf("CPU Time used: %3.1f ms", duration * 1000); \
    }

#define N (1024 * 1024)
#define FULL_DATA_SIZE (N * 120)

__global__ void
vectorAdd(const double *A, const double *B, double *C)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        C[i] = cos(A[i]) / sin(B[i]);
    }
}

void cpuVectorAdd(const double *A, const double *B, double *C){
    START_CPU
    for (int i = 0; i < FULL_DATA_SIZE; ++i)
    {
        C[i] = cos(A[i]) / sin(B[i]);
    }
    END_CPU
}

void gpuVectorAddOnDefaultStream(const double *A, const double *B, double *C){
    double *devA, *devB, *devC;

    cudaMalloc((void **)&devA, N*sizeof(double));
    cudaMalloc((void **)&devB, N*sizeof(double));
    cudaMalloc((void **)&devC, N*sizeof(double));

    START_GPU
    for(int i=0;i<FULL_DATA_SIZE;i+=N){
        cudaMemcpy(devA, A+i, N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(devB, B+i, N*sizeof(double), cudaMemcpyHostToDevice);

        vectorAdd<<<N/256, 256, 0>>>(devA, devB, devC);

        cudaMemcpy(C+i, devC, N*sizeof(double), cudaMemcpyDeviceToHost);
    }
    END_GPU

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

void gpuVectorAddOnOneStream(const double *A, const double *B, double *C){
    cudaStream_t stream0;
    cudaStreamCreate(&stream0);

    double *devA, *devB, *devC;
    cudaMalloc((void **)&devA, N*sizeof(double));
    cudaMalloc((void **)&devB, N*sizeof(double));
    cudaMalloc((void **)&devC, N*sizeof(double));

    START_GPU
    for(int i=0;i<FULL_DATA_SIZE;i+=N){
        cudaMemcpyAsync(devA, A+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(devB, B+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaStreamSynchronize(stream0);
        vectorAdd<<<N/256, 256, 0, stream0>>>(devA, devB, devC);
        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(C+i, devC, N*sizeof(double), cudaMemcpyDeviceToHost, stream0);
    }
    cudaStreamSynchronize(stream0);
    END_GPU

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    cudaStreamDestroy(stream0);
}

void gpuVectorAddOnTwoStreamUnefficient(const double *A, const double *B, double *C){
    cudaStream_t stream0,stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    double *devA, *devB, *devC;
    cudaMalloc((void **)&devA, N*sizeof(double));
    cudaMalloc((void **)&devB, N*sizeof(double));
    cudaMalloc((void **)&devC, N*sizeof(double));

    START_GPU
    for(int i=0;i<FULL_DATA_SIZE;i+=N*2){
        cudaMemcpyAsync(devA, A+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(devB, B+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaStreamSynchronize(stream0);
        vectorAdd<<<N/256, 256, 0, stream0>>>(devA, devB, devC);
        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(C+i, devC, N*sizeof(double), cudaMemcpyDeviceToHost, stream0);

        cudaMemcpyAsync(devA, A+i+N, N*sizeof(double), cudaMemcpyHostToDevice, stream1);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(devB, B+i+N, N*sizeof(double), cudaMemcpyHostToDevice, stream1);
        cudaStreamSynchronize(stream1);
        vectorAdd<<<N/256, 256, 0, stream1>>>(devA, devB, devC);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(C+i+N, devC, N*sizeof(double), cudaMemcpyDeviceToHost, stream1);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    END_GPU

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}

void gpuVectorAddOnTwoStream(const double *A, const double *B, double *C){
    cudaStream_t stream0,stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    double *devA1, *devB1, *devC1;
    cudaMalloc((void **)&devA1, N*sizeof(double));
    cudaMalloc((void **)&devB1, N*sizeof(double));
    cudaMalloc((void **)&devC1, N*sizeof(double));
    double *devA2, *devB2, *devC2;
    cudaMalloc((void **)&devA2, N*sizeof(double));
    cudaMalloc((void **)&devB2, N*sizeof(double));
    cudaMalloc((void **)&devC2, N*sizeof(double));

    START_GPU
    for(int i=0;i<FULL_DATA_SIZE;i+=N*2){
        cudaMemcpyAsync(devA1, A+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(devA2, A+i+N, N*sizeof(double), cudaMemcpyHostToDevice, stream1);

        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(devB1, B+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(devB2, B+i+N, N*sizeof(double), cudaMemcpyHostToDevice, stream1);

        cudaStreamSynchronize(stream0);
        vectorAdd<<<N/256, 256, 0, stream0>>>(devA1, devB1, devC1);
        cudaStreamSynchronize(stream1);
        vectorAdd<<<N/256, 256, 0, stream1>>>(devA2, devB2, devC2);

        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(C+i, devC1, N*sizeof(double), cudaMemcpyDeviceToHost, stream0);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(C+i+N, devC2, N*sizeof(double), cudaMemcpyDeviceToHost, stream1);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    END_GPU

    cudaFree(devA1);
    cudaFree(devB1);
    cudaFree(devC1);

    cudaFree(devA2);
    cudaFree(devB2);
    cudaFree(devC2);

        
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}

void gpuVectorAddOnThreeStream(const double *A, const double *B, double *C){
    cudaStream_t stream0,stream1,stream2;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double *devA1, *devB1, *devC1;
    cudaMalloc((void **)&devA1, N*sizeof(double));
    cudaMalloc((void **)&devB1, N*sizeof(double));
    cudaMalloc((void **)&devC1, N*sizeof(double));

    double *devA2, *devB2, *devC2;
    cudaMalloc((void **)&devA2, N*sizeof(double));
    cudaMalloc((void **)&devB2, N*sizeof(double));
    cudaMalloc((void **)&devC2, N*sizeof(double));

    double *devA3, *devB3, *devC3;
    cudaMalloc((void **)&devA3, N*sizeof(double));
    cudaMalloc((void **)&devB3, N*sizeof(double));
    cudaMalloc((void **)&devC3, N*sizeof(double));

    START_GPU
    for(int i=0;i<FULL_DATA_SIZE;i+=N*3){
        cudaMemcpyAsync(devA1, A+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(devA2, A+i+N, N*sizeof(double), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(devA3, A+i+N*2, N*sizeof(double), cudaMemcpyHostToDevice, stream2);
        
        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(devB1, B+i, N*sizeof(double), cudaMemcpyHostToDevice, stream0);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(devB2, B+i+N, N*sizeof(double), cudaMemcpyHostToDevice, stream1);
        cudaStreamSynchronize(stream2);
        cudaMemcpyAsync(devB3, B+i+N*2, N*sizeof(double), cudaMemcpyHostToDevice, stream2);

        cudaStreamSynchronize(stream0);
        vectorAdd<<<N/256, 256, 0, stream0>>>(devA1, devB1, devC1);
        cudaStreamSynchronize(stream1);
        vectorAdd<<<N/256, 256, 0, stream1>>>(devA2, devB2, devC2);
        cudaStreamSynchronize(stream2);
        vectorAdd<<<N/256, 256, 0, stream2>>>(devA3, devB3, devC3);
        
        cudaStreamSynchronize(stream0);
        cudaMemcpyAsync(C+i, devC1, N*sizeof(double), cudaMemcpyDeviceToHost, stream0);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(C+i+N, devC2, N*sizeof(double), cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream2);
        cudaMemcpyAsync(C+i+N*2, devC3, N*sizeof(double), cudaMemcpyDeviceToHost, stream2);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    END_GPU

    cudaFree(devA1);
    cudaFree(devB1);
    cudaFree(devC1);

    cudaFree(devA2);
    cudaFree(devB2);
    cudaFree(devC2);

    cudaFree(devA3);
    cudaFree(devB3);
    cudaFree(devC3);

        
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}



int main(void)
{
    double *hostA, *hostB, *hostC;

    cudaHostAlloc((void **)&hostA, FULL_DATA_SIZE*sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **)&hostB, FULL_DATA_SIZE*sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void **)&hostC, FULL_DATA_SIZE*sizeof(double), cudaHostAllocDefault);

    for (int i = 0; i < FULL_DATA_SIZE; ++i)
    {
        hostA[i] = rand() / (double)RAND_MAX;
        hostB[i] = rand() / (double)RAND_MAX;
    }

    cpuVectorAdd(hostA, hostB, hostC);
    printf("\n");

    gpuVectorAddOnDefaultStream(hostA, hostB, hostC);
    printf("(with default stream)\n");
    
    gpuVectorAddOnOneStream(hostA, hostB, hostC);
    printf("(with 1 stream)\n");

    gpuVectorAddOnTwoStreamUnefficient(hostA, hostB, hostC);
    printf("(with 2 stream but unefficient)\n");

    gpuVectorAddOnTwoStream(hostA, hostB, hostC);
    printf("(with 2 stream)\n");

    gpuVectorAddOnThreeStream(hostA, hostB, hostC);
    printf("(with 3 stream)\n");

    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
}