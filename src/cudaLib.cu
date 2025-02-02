
#include "cudaLib.cuh"
#include <curand_kernel.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if( index < size){
		y[index] = x[index] * scale + y[index];
	}
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	int Thread_per_block = 512;

	// Init two arrays
	float* h_x = (float*)malloc(vectorSize * sizeof(float));
	float* h_y = (float*)malloc(vectorSize * sizeof(float));

	// create a random number generator
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<float> distribution(-1e3, 1e3);

	//feed two arrays values and INIT A
	for (int i = 0; i < vectorSize; i++){
		// h_x[i] = distribution(generator);
		// h_y[i] = distribution(generator);
		h_x[i] = (float)(rand() % 100);
		h_y[i] = (float)(rand() % 100);
	}
	float a = distribution(generator);
	a = 1.923;

	//allocate the memory for GPU device and copy the arrays to device
	float* d_x;
	cudaMalloc(&d_x, vectorSize * sizeof(float));

	float* d_y;
	cudaMalloc(&d_y, vectorSize * sizeof(float));

	cudaMemcpy(d_x, h_x, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

	//calculate the number of blocks and thread/block needed
	int numBlocks;
	numBlocks = (vectorSize + Thread_per_block -1 ) / Thread_per_block;

	//invoke kernal
	saxpy_gpu<<<numBlocks, Thread_per_block>>>(d_x, d_y, a, vectorSize);

	//copy memory back to the host
	float* device_result = (float*)malloc(vectorSize * sizeof(float));
	cudaMemcpy(device_result, d_y, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	//verify the result by using the CPU model
	saxpy_cpu(h_x,h_y,a,vectorSize);
	int error_count = 0;
	float diff = 0;
	for (int i = 0; i < vectorSize; i++){
		diff = device_result[i] - h_y[i];
		if( h_y[i] != device_result[i] && (diff > 0.001 || diff < -0.001)){
			error_count ++;
			printf("device_result: %f, host result: %f  ", device_result[i], h_y[i]);
			printf("Error index: %d\n", i);
		}
	}

	//free host memory
	free(h_x);
	free(h_y);
	free(device_result);
	//free the device memeory
	cudaFree(d_x);
	cudaFree(d_y);

	std::cout << " Error result: " << error_count << "\n";

	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here

	uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

	// get the calculations needed by one kernel
	uint64_t iter = (sampleSize + pSumSize - 1) / pSumSize;

	// set the random generator
	curandState_t rng;
	curand_init(clock64(), index, 0, &rng);

	uint64_t starting_point = index * iter;

	uint64_t hitCount = 0;
	float x;
	float y;

	for(uint64_t i = 0; i < iter; i++){
		if( (starting_point + i) < sampleSize){
			x = curand_uniform(&rng);
			y = curand_uniform(&rng);
			if ( int(x * x + y * y) == 0 ) {
				hitCount++;
			}
		}
	}
	pSums[index] = hitCount;

	// printf("Thread ID: %llu, starting_point: %llu , iter: %llu, hit count: %llu\n", index, starting_point, iter, hitCount);


}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	uint64_t Thread_ID = blockIdx.x * blockDim.x + threadIdx.x;

	//use a loop to calculate the sum of "reduceSize" elements
	uint64_t starting_point = Thread_ID * reduceSize;
	uint64_t partial_sum = 0;
	for (uint64_t i = 0; i < reduceSize; i++){
		if( (starting_point + i) < pSumSize){
			partial_sum = partial_sum + pSums[starting_point + i];
		}
	}
	if (Thread_ID < ((pSumSize + reduceSize -1) / reduceSize)){
		totals[Thread_ID] = partial_sum;
	}
	// printf("Thread_ID: %llu, Partial_sum: %llu, Starting_point: %llu  \n", Thread_ID, partial_sum, starting_point);
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";

	// // Way One: Try to generate all the points at host side and then move the data to the device memory //////////////////////
	// std::random_device random_device;
	// std::uniform_real_distribution<float> dist(0.0, 1.0);

	// // Allocate the space for points on the host side
	// float* x_h = (float *)malloc(sampleSize * sizeof(float*));
	// float* y_h = (float *)malloc(sampleSize * sizeof(float*));

	// // Init the x_h and y_h arrays
	// for (int i = 0; j < sampleSize; i++){
	// 	x_h[i] = dist(random_device);
	// 	y_h[i] = dist(random_device); 
	// }

	// //allocate the memory for GPU device and copy the arrays to device
	// float* x_d;
	// cudaMalloc(&x_d, sampleSize * sizeof(float));

	// float* y_d;
	// cudaMalloc(&y_d, sampleSize * sizeof(float));

	// cudaMemcpy(d_x, h_x, sampleSize * sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_y, h_y, sampleSize * sizeof(float), cudaMemcpyHostToDevice);  ////////////////////////////////////////////







	// Way Two: just generate the point within each kernel and then do the calculation. ////////////////////////////////////////////
	// calculate how many calculations each threads need to perform
	uint64_t Thread_per_block     = 1024;
	uint64_t numBlocks            = generateThreadCount; // If there are too many threads, huge overhead of context switch (API calls).  
	// uint64_t numBlocks            = 160;
	uint64_t reduced_result_count = (Thread_per_block * numBlocks + reduceSize - 1) / reduceSize;
	printf("reduced_result_count: %llu\n", reduced_result_count);

	// allocate the memory for host and device
	uint64_t* result_host         = (uint64_t*)malloc((Thread_per_block * numBlocks) * sizeof(uint64_t));
	uint64_t* reduced_result_host = (uint64_t*)malloc(reduced_result_count * sizeof(uint64_t));

	uint64_t* result_device;
	cudaMalloc(&result_device, (Thread_per_block * numBlocks) * sizeof(uint64_t));

	uint64_t* reduced_result_device;
	cudaMalloc(&reduced_result_device, reduced_result_count * sizeof(uint64_t));

	//lanuch the kernel
	generatePoints<<<numBlocks, Thread_per_block>>>(result_device, (Thread_per_block * numBlocks), sampleSize);
	cudaDeviceSynchronize();

	reduceCounts<<<(numBlocks/reduceSize), Thread_per_block>>>(result_device, reduced_result_device, (Thread_per_block * numBlocks), reduceSize);
	cudaDeviceSynchronize();

	//Copy the memory back to the host
	cudaMemcpy(result_host, result_device, (Thread_per_block * numBlocks) * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(reduced_result_host, reduced_result_device, reduced_result_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	uint64_t total_count = 0;
	for(uint64_t i = 0; i < (Thread_per_block * numBlocks); i++){
		total_count = total_count + result_host[i];
	}
	approxPi = ((double)total_count / sampleSize);
	approxPi = approxPi * 4.0f;

	// uint64_t total_count = 0;
	// for(uint64_t j = 0; j < (reduced_result_count); j++){
	// 	total_count = total_count + reduced_result_host[j];
	// 	// printf("reduced partial: %llu\n", reduced_result_host[j]);
	// }
	// approxPi = ((double)total_count / sampleSize);
	// approxPi = approxPi * 4.0f;

	//free the memory
	free(result_host);
	free(reduced_result_host);
	cudaFree(result_device);
	cudaFree(reduced_result_device);


	return approxPi;
}
