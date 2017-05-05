# K-Means Image Segmentation in CUDA

## Requirements:
- Install [NVIDIA CUDA Toolkit](docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install [Cilk Plus](https://www.cilkplus.org/download)

## Clone repository:
```sh
$ git clone https://github.com/dmadariaga/gpu-programming-research.git
```

## Run GPU algorithm:
```sh
$ nvcc -o kmeans_gpu kmeans_gpu.cu
$ kmeans_gpu input.ppm k numIter output.ppm
```
where:
- `input.ppm`: Image to apply segmentation. Must be a raw PPM file (file type = `P6`).
- `k`: Number of clusters
- `numIter`: Number of iterations
- `output.ppm`: Name of the image file created to save the results.

## Run CPU algorithm:
```sh
$ gcc -std=gnu99 -o kmeans_cpu kmeans_cpu.c -fcilkplus -lcilkrts -lm
$ kmeans_cpu input.ppm k numIter output.ppm
```
Params are the same that GPU version

## Run tests:
```sh
$ ./test_gpu.sh >> gpu_results.csv
$ ./test_cpu.sh >> cpu_results.csv
```
csv format is:
