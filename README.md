# K-Means Image Segmentation in CUDA

## Requirements:
- Install [NVIDIA CUDA Toolkit](docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install [Cilk Plus](https://www.cilkplus.org/download)

## Clone repository:
```sh
$ git clone https://github.com/dmadariaga/gpu-programming-research.git`
```

## Run GPU algorithm:
```sh
$ nvcc kmeans_gpu.cu -o kmeans_gpu
$ kmeans_gpu input.ppm k numIter output.ppm
```
