# K-Means Image Segmentation in CUDA
![](http://users.dcc.uchile.cl/~dmadaria/images/1600.png "Description goes here")
![](http://users.dcc.uchile.cl/~dmadaria/images/1.png "Description goes here")
 <br />K-Means example (K=3, Iterations=1000)

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
Where:
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

## Run GPU tests:
```sh
$ ./test_gpu.sh >> gpu_results.csv
```
`gpu_results.csv` format is:
```sh
unit,size,k,iter,t
gpu,160000,3,100,0.001152
gpu,160000,3,1000,0.627432
gpu,160000,3,5000,3.854647
...
```
Where:
- `unit`: Unit of execution (GPU or CPU)
- `size`: Number of pixels in image
- `k`: Number of clusters
- `iter`: Number of iterations
- `t`: Elapsed time int seconds

## Run CPU tests:
```sh
$ ./test_cpu.sh >> cpu_results.csv
```
`cpu_results.csv` format is:
```sh
unit,size,k,iter,t
cpu,160000,3,100,0.570475
cpu,160000,3,1000,5.699365
cpu,160000,3,5000,3.793827
...
```
Column headers are the same that GPU version
