#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <ctype.h>
#include <sys/time.h>
#include <time.h> 

#define BLOCK_SIZE 16
#define GRID_SIZE 256

#define uchar unsigned char

__constant__ int d_k;
__constant__ int d_pixelCount;

__global__ void assignClusters(uchar *d_imageR, uchar *d_imageG, uchar *d_imageB, int *d_assignedClusters, 
				uchar *d_clusterR, uchar *d_clusterG, uchar *d_clusterB){
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	if (threadID < d_pixelCount){
		double dist, min = 0;
		int index;
		for (int i=0; i<d_k; i++){
			dist = sqrtf(powf(d_imageR[threadID] - d_clusterR[i], 2) +
					powf(d_imageG[threadID] - d_clusterG[i], 2) +
					powf(d_imageB[threadID] - d_clusterB[i], 2) );
			if (dist < min || i == 0){
				min = dist;	
				index = i;
			}
		}
		d_assignedClusters[threadID] = index;
	}
}

__global__ void sumClusters(uchar *d_imageR, uchar *d_imageG, uchar *d_imageB, int *d_assignedClusters,
				int *d_sumR, int *d_sumG, int *d_sumB, int *d_clusterSize){
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;

	if(threadID < d_pixelCount) {
		int cluster = d_assignedClusters[threadID];
		int R = d_imageR[threadID];
		int G = d_imageG[threadID];
		int B = d_imageB[threadID];

		atomicAdd(&d_sumR[cluster], R);
		atomicAdd(&d_sumG[cluster], G);
		atomicAdd(&d_sumB[cluster], B);
		atomicAdd(&d_clusterSize[cluster], 1);
	}	
}

__global__ void clearClusterInfo(int *d_sumR, int *d_sumG, int *d_sumB, int *d_clusterSize){
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;
	if(threadID < d_k) {
		d_sumR[threadID] = 0;
		d_sumG[threadID] = 0;
		d_sumB[threadID] = 0;
		d_clusterSize[threadID] = 0;
	}
}

__global__ void calculateCentroids(uchar *d_clusterR, uchar *d_clusterG, uchar *d_clusterB,
					int *d_sumR, int *d_sumG, int *d_sumB, int *d_clusterSize){
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;
	if(threadID < d_k) {
		int clusterSize = d_clusterSize[threadID];
		d_clusterR[threadID] = d_sumR[threadID] / clusterSize;
		d_clusterG[threadID] = d_sumG[threadID] / clusterSize;
		d_clusterB[threadID] = d_sumB[threadID] / clusterSize;
	}
}

void error(char const *message){
  	fprintf(stderr, "Error: %s\n", message);
  	exit(1);
}
void readPPMHeader(FILE *fp, int *width, int *height){
  	char ch;
  	int  maxval;

  	if (fscanf(fp, "P%c\n", &ch) != 1 || ch != '6')
    	error("file is not in ppm raw format (P6)");

  	/* skip comments */
  	ch = getc(fp);
  	while (ch == '#'){
    	do {
			ch = getc(fp);
      	} while (ch != '\n');	/* read to the end of the line */
      	ch = getc(fp);            
    }

  	if (!isdigit(ch)) error("cannot read header information from ppm file");

  	ungetc(ch, fp);		/* put that digit back */

  	/* read the width, height, and maximum value for a pixel */
  	fscanf(fp, "%d%d%d\n", width, height, &maxval);

  	if (maxval != 255) error("image is not true-color (24 bit); read failed");
}

void writePPMImage(uchar *imageR, uchar *imageG, uchar *imageB, int width, int height, char const *filename){
    int pixelCount = width*height;

    FILE *fp = fopen(filename, "w");

    if (!fp) error("cannot open file for writing");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    for (int i=0; i<pixelCount; i++){
    	fwrite(&imageR[i], sizeof(uchar), 1, fp);
    	fwrite(&imageG[i], sizeof(uchar), 1, fp);
    	fwrite(&imageB[i], sizeof(uchar), 1, fp);
  	}

    fclose(fp);
} 

void uploadImage(uchar *image, int size, uchar *imageR, uchar *imageG, uchar *imageB){
	for (int i=0; i<size; i+=3){
		int index = (int)i/3;
		imageR[index] = image[i];
		imageG[index] = image[i+1];
		imageB[index] = image[i+2];	
	}
}

int main(int argc, char *argv[]) {
	char* inputFile = argv[1];
	int k = atoi(argv[2]);
	int numIter = atoi(argv[3]);
	char* outputFile;
	if (argc ==5)
		outputFile = argv[4];

	int width, height;

	FILE  *fp    = fopen(inputFile, "r");
	readPPMHeader(fp, &width, &height);
	int pixelCount = width*height;
	uchar *image = (uchar*)malloc(pixelCount*3);
	fread(image, 1, pixelCount*3, fp);
	fclose(fp);

	uchar *imageR, *imageG, *imageB, *clusterR, *clusterG, *clusterB;
	int *assignedClusters;
	uchar *d_imageR, *d_imageG, *d_imageB, *d_clusterR, *d_clusterG, *d_clusterB;
	int *d_assignedClusters, *d_sumR, *d_sumG, *d_sumB, *d_clusterSize;

	int imageSize = sizeof(uchar)*pixelCount;
	int centroidsSize = sizeof(int)*k;

	imageR = (uchar*)malloc(imageSize);
	imageG = (uchar*)malloc(imageSize);
	imageB = (uchar*)malloc(imageSize);

	uploadImage(image, pixelCount*3, imageR, imageG, imageB);
	free(image);

	clusterR = (uchar*)calloc(sizeof(uchar), k);
	clusterG = (uchar*)calloc(sizeof(uchar), k);
	clusterB = (uchar*)calloc(sizeof(uchar), k);
	assignedClusters = (int*)malloc(sizeof(int)*pixelCount);

	/*initial random centroids*/
	srand (time(NULL));
	for (int i=0; i<k; i++){
		clusterR[i] = rand() % 256;
		clusterG[i] = rand() % 256;
		clusterB[i] = rand() % 256;
	}
	
	cudaMalloc(&d_imageR, imageSize);
	cudaMalloc(&d_imageG, imageSize);	
	cudaMalloc(&d_imageB, imageSize);
	cudaMalloc(&d_assignedClusters, sizeof(int)*pixelCount);
	cudaMalloc(&d_clusterR, sizeof(uchar)*k);
	cudaMalloc(&d_clusterG, sizeof(uchar)*k);
	cudaMalloc(&d_clusterB, sizeof(uchar)*k);
	cudaMalloc(&d_sumR, centroidsSize);
	cudaMalloc(&d_sumG, centroidsSize);
	cudaMalloc(&d_sumB, centroidsSize);
	cudaMalloc(&d_clusterSize, centroidsSize);

	cudaMemcpy(d_imageR, imageR, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_imageG, imageG, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_imageB, imageB, imageSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_clusterR, clusterR, sizeof(uchar)*k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_clusterG, clusterG, sizeof(uchar)*k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_clusterB, clusterB, sizeof(uchar)*k, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(d_k, &k, sizeof(int));
	cudaMemcpyToSymbol(d_pixelCount, &pixelCount, sizeof(int));


	int BLOCK_X, BLOCK_Y;
	BLOCK_X = ceil(width/BLOCK_SIZE);
	BLOCK_Y = ceil(height/BLOCK_SIZE);
	if(BLOCK_X > GRID_SIZE)
		BLOCK_X = GRID_SIZE;
	if(BLOCK_Y > GRID_SIZE)
		BLOCK_Y = GRID_SIZE;
	//2D Grid
	//Minimum number of threads that can handle width¡height pixels
 	dim3 dimGRID(BLOCK_X,BLOCK_Y);
 	//2D Block
	//Each dimension is fixed
	dim3 dimBLOCK(BLOCK_SIZE,BLOCK_SIZE);
	
	struct timespec stime, etime;
	double t;
	if (clock_gettime(CLOCK_THREAD_CPUTIME_ID , &stime)) {
	    fprintf(stderr, "clock_gettime failed");
	    exit(-1);
	}
	for (int i=0; i<numIter; i++){
		assignClusters<<< dimGRID, dimBLOCK >>> (d_imageR, d_imageG, d_imageB, d_assignedClusters,
								d_clusterR, d_clusterG, d_clusterB);
		clearClusterInfo<<< 1, dimBLOCK >>> (d_sumR, d_sumG, d_sumB, d_clusterSize);		
		sumClusters<<< dimGRID, dimBLOCK >>> (d_imageR, d_imageG, d_imageB, d_assignedClusters,
								d_sumR, d_sumG, d_sumB, d_clusterSize);
		calculateCentroids<<< 1, dimBLOCK >>> (d_clusterR, d_clusterG, d_clusterB,
								d_sumR, d_sumG, d_sumB, d_clusterSize);
	}
	if (clock_gettime(CLOCK_THREAD_CPUTIME_ID , &etime)) {
    		fprintf(stderr, "clock_gettime failed");
   		 exit(-1);
  	}
  
  	t = (etime.tv_sec - stime.tv_sec) + (etime.tv_nsec - stime.tv_nsec) / 1000000000.0;
  	printf("%d,%d,%d,%lf\n", pixelCount, k, numIter, t);

	int *clusterSize = (int*)malloc(sizeof(int)*k);
	cudaMemcpy(clusterSize, d_clusterSize, centroidsSize, cudaMemcpyDeviceToHost);

	cudaMemcpy(clusterR, d_clusterR, sizeof(uchar)*k, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusterG, d_clusterG, sizeof(uchar)*k, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusterB, d_clusterB, sizeof(uchar)*k, cudaMemcpyDeviceToHost);
	cudaMemcpy(imageR, d_imageR, imageSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(imageG, d_imageR, imageSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(imageB, d_imageR, imageSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(assignedClusters, d_assignedClusters, sizeof(int)*pixelCount, cudaMemcpyDeviceToHost);

	for (int i=0; i<pixelCount; i++){
		int cluster = assignedClusters[i];
		imageR[i] = clusterR[cluster];
		imageG[i] = clusterG[cluster];
		imageB[i] = clusterB[cluster];
	}
	if (argc == 5)
		writePPMImage(imageR, imageG, imageB, width, height, outputFile);
	
	free(imageR);
	free(imageG);
	free(imageB);

	free(clusterR);
	free(clusterG);
	free(clusterB);

	free(assignedClusters);
	free(clusterSize);

	cudaFree(d_imageR);
	cudaFree(d_imageG);	
	cudaFree(d_imageB);
	cudaFree(d_assignedClusters);
	cudaFree(d_clusterR);
	cudaFree(d_clusterG);
	cudaFree(d_clusterB);
	cudaFree(d_sumR);
	cudaFree(d_sumG);
	cudaFree(d_sumB);
	cudaFree(d_clusterSize);
}
