nvcc -I ./cuda-samples/Common -O2 -Xcompiler -fopenmp vectorAdd.cu -o build/vectorAdd
./build/vectorAdd