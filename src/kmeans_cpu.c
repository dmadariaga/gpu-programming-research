// Compile: gcc -std=gnu99 -o kmeans_cpu kmeans_cpu.c -fcilkplus -lcilkrts -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>

#ifdef NOPARALLEL
#define __cilkrts_get_nworkers() 1
#define cilk_for for
#define cilk_spawn 
#define cilk_sync 
#else
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/common.h>
#endif

#define num_threads __cilkrts_get_nworkers()
#define uchar unsigned char

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
        } while (ch != '\n'); /* read to the end of the line */
        ch = getc(fp);            
    }

    if (!isdigit(ch)) error("cannot read header information from ppm file");

    ungetc(ch, fp);   /* put that digit back */

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

void assignClusters(int pixelCount, int k, uchar *imageR, uchar *imageG, uchar *imageB, int *assignedClusters, 
        uchar *clusterR, uchar *clusterG, uchar *clusterB){

  cilk_for(int j = 0; j < pixelCount; j++) {
    double dist, min = 0;
    int index;
    for (int i=0; i<k; i++){
      dist = sqrt(pow(imageR[j] - clusterR[i], 2) +
          pow(imageG[j] - clusterG[i], 2) +
          pow(imageB[j] - clusterB[i], 2) );
      if (dist < min || i == 0){
        min = dist; 
        index = i;
      }
    }
    assignedClusters[j] = index;
  }
}

void clearClusterInfo(int k, int *sumR, int *sumG, int *sumB, int *clusterSize){
  cilk_for(int j = 0; j < k; j++) {
    sumR[j] = 0;
    sumG[j] = 0;
    sumB[j] = 0;
    clusterSize[j] = 0;
  }
}

void sumClusters(int pixelCount, int k, uchar *imageR, uchar *imageG, uchar *imageB, int *assignedClusters,
        int *sumR, int *sumG, int *sumB, int *clusterSize){
  int* partialR = malloc(num_threads*k*sizeof(int*));
  int* partialG = malloc(num_threads*k*sizeof(int*));
  int* partialB = malloc(num_threads*k*sizeof(int*));
  int* partialSize = malloc(num_threads*k*sizeof(int));

  int chk = pixelCount/num_threads;
  
  /* Each thread will write in its own local variable */
  cilk_for(int i = 0; i < num_threads; i++) {
    uint  ll = i*chk, ul = ll + chk;
    if(i == num_threads-1)
      ul = pixelCount;
    int* tmpR = calloc(k, sizeof(int));
    int* tmpG = calloc(k, sizeof(int));
    int* tmpB = calloc(k, sizeof(int));
    int* tmpSize = calloc(k, sizeof(int));
    
    for(uint j = ll; j < ul; j++) {
      int cluster = assignedClusters[j];
      tmpR[cluster] += imageR[j];
      tmpG[cluster] += imageG[j];
      tmpB[cluster] += imageB[j];
      tmpSize[cluster] += 1;
    }
    for (int j=0; j<k; j++){
      partialR[i*k+j] = tmpR[j];
      partialG[i*k+j] = tmpG[j];
      partialB[i*k+j] = tmpB[j];
      partialSize[i*k+j] = tmpSize[j];
    }
    free(tmpR);
    free(tmpG);
    free(tmpB);
    free(tmpSize);
  }
  
  /* The total is the sum of the partial results*/
  for(int i = 0; i < num_threads; i++){
    for (int j=0; j<k; j++){
      sumR[j] += partialR[i*k+j];
      sumG[j] += partialG[i*k+j];
      sumB[j] += partialB[i*k+j];
      clusterSize[j] += partialSize[i*k+j];
    }
  }

  free(partialR);
  free(partialG);
  free(partialB);
  free(partialSize);
}

void calculateCentroids(int k, uchar *clusterR, uchar *clusterG, uchar *clusterB,
          int *sumR, int *sumG, int *sumB, int *clusterSize){
  cilk_for(int j = 0; j < k; j++) {
    int size = clusterSize[j];
    if (size==0)
      size = 1;
    clusterR[j] = sumR[j] / size;
    clusterG[j] = sumG[j] / size;
    clusterB[j] = sumB[j] / size;
  }
}

int main(int argc, char* argv[]) {
  char* inputFile = argv[1];
  int k = atoi(argv[2]);
  int numIter = atoi(argv[3]);
  char* outputFile;
  if (argc == 5)
    outputFile = argv[4];

  int width, height;

  FILE  *fp    = fopen(inputFile, "r");
  readPPMHeader(fp, &width, &height);
  int pixelCount = width*height;
  uchar *image = (uchar*)malloc(pixelCount*3);
  fread(image, 1, pixelCount*3, fp);
  fclose(fp);

  uchar *imageR, *imageG, *imageB, *clusterR, *clusterG, *clusterB;
  int *assignedClusters, *sumR, *sumG, *sumB, *clusterSize;

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
  sumR = (int*)malloc(centroidsSize);
  sumG = (int*)malloc(centroidsSize);
  sumB = (int*)malloc(centroidsSize);
  clusterSize = (int*)malloc(centroidsSize);

  /*initial random centroids*/
  srand (time(NULL));
  for (int i=0; i<k; i++){
    clusterR[i] = rand() % 256;
    clusterG[i] = rand() % 256;
    clusterB[i] = rand() % 256;
  }

  struct timespec stime, etime;
  double t;
  
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID , &stime)) {
    fprintf(stderr, "clock_gettime failed");
    exit(-1);
  }
  
  /* YOUR PARALLEL CODE*/
  for (int i=0; i<numIter; i++){
    assignClusters(pixelCount, k, imageR, imageG, imageB, assignedClusters, clusterR, clusterG, clusterB);
    clearClusterInfo(k, sumR, sumG, sumB, clusterSize);    
    sumClusters(pixelCount, k, imageR, imageG, imageB, assignedClusters, sumR, sumG, sumB, clusterSize);
    calculateCentroids(k, clusterR, clusterG, clusterB, sumR, sumG, sumB, clusterSize);
  }
  
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID , &etime)) {
    fprintf(stderr, "clock_gettime failed");
    exit(-1);
  }
  
  for (int i=0; i<pixelCount; i++){
    int cluster = assignedClusters[i];
    imageR[i] = clusterR[cluster];
    imageG[i] = clusterG[cluster];
    imageB[i] = clusterB[cluster];
  }

  t = (etime.tv_sec - stime.tv_sec) + (etime.tv_nsec - stime.tv_nsec) / 1000000000.0;
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
  
  printf("cpu,%d,%d,%d,%lf\n", pixelCount, k, numIter, t);
  
  return EXIT_SUCCESS;
}
