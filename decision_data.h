#ifndef DECISION_DATA_H
#define DECISION_DATA_H

#include "cudpp/cudpp.h"
#include "misc_kernel.cu"

class DecisionData
{
	public:
		DecisionData();
		~DecisionData();

		float *d_primary_ray_normal;
		float *d_primary_ray_t_value;
		float *d_primary_ray_direction;

		int *d_is_shadowed;
		int *d_intersect_id;

		int *h_intersect_id;
		float *h_t;
		float *h_t2;
		int *h_tt;
		size_t *h_numCudaBlocks;

		unsigned int *d_map;
		unsigned int *d_valid;
		unsigned int *d_scratchValid;
		unsigned int *d_segScanArr;
		unsigned int *d_scanArray;
		unsigned int *d_prefixMap;
		size_t *d_numCudaBlocks;

		float *d_angle;

		float *d_x_list;
		float *d_y_list;
		
		float *h_x_list;
		float *h_y_list;

		void printSomething();
		void get_pseudo_valid_array();
		void sort_by_block_indices();
		void segment_pseudo_valid_array();
		void get_cuda_block_array();
		void copy_thread_id();
		void stream_compact();
	private:
		void init_cudpp_plans();

		CUDPPConfiguration config;
		CUDPPHandle sortPlan;
		CUDPPHandle segScanPlan;
		CUDPPHandle compactPlan;
		CUDPPHandle sortDSPlan;
		CUDPPHandle scanDSPlan;
		CUDPPHandle scanN1N2N3Plan;
		CUDPPResult result;


};

DecisionData::DecisionData()
{
	cutilSafeCall(cudaMalloc((void **) &d_primary_ray_normal, sizeof(float) * 3 * IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **) &d_primary_ray_t_value, sizeof(float) * IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **) &d_primary_ray_direction, sizeof(float) * 3 * IMAGE_SIZE));

	cutilSafeCall(cudaMalloc((void **) &d_is_shadowed, sizeof(int) * IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **) &d_intersect_id, sizeof(int) * IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **) &d_map, sizeof(unsigned int) * IMAGE_SIZE * 2));

	cutilSafeCall(cudaMalloc((void **)&d_valid, sizeof(unsigned int)*IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **)&d_scratchValid, sizeof(unsigned int)*IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **)&d_segScanArr, sizeof(unsigned int)*IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **)&d_scanArray, sizeof(unsigned int)*IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **) &d_prefixMap, sizeof(unsigned int)*(NUM_BLOCKS_X*NUM_BLOCKS_Y + 1)));
	cutilSafeCall(cudaMalloc((void **)&d_numCudaBlocks, sizeof(size_t)));
	
	cutilSafeCall(cudaMalloc((void **)&d_x_list, sizeof(float) * IMAGE_SIZE));
	cutilSafeCall(cudaMalloc((void **)&d_y_list, sizeof(float) * IMAGE_SIZE));

	h_t = (float *) malloc(sizeof(float) * IMAGE_SIZE * 3);
	h_t2 = (float *) malloc(sizeof(float) * IMAGE_SIZE * 3);
	h_tt = (int *) malloc(sizeof(int) * IMAGE_SIZE);
	h_numCudaBlocks = (size_t *) malloc(sizeof(size_t));

	h_x_list = (float *) malloc(sizeof(float) * IMAGE_SIZE);
	h_y_list = (float *) malloc(sizeof(float) * IMAGE_SIZE);

	init_cudpp_plans();
}

DecisionData::~DecisionData()
{
	cutilSafeCall(cudaFree(d_primary_ray_normal));
	cutilSafeCall(cudaFree(d_primary_ray_t_value));
	cutilSafeCall(cudaFree(d_primary_ray_direction));

	cutilSafeCall(cudaFree(d_is_shadowed));
	cutilSafeCall(cudaFree(d_intersect_id));
	cutilSafeCall(cudaFree(d_map));
	cutilSafeCall(cudaFree(d_valid));
	cutilSafeCall(cudaFree(d_scratchValid));
	cutilSafeCall(cudaFree(d_segScanArr));
	cutilSafeCall(cudaFree(d_scanArray));
	cutilSafeCall(cudaFree(d_prefixMap));
	cutilSafeCall(cudaFree(d_numCudaBlocks));
	cutilSafeCall(cudaFree(d_x_list));
	cutilSafeCall(cudaFree(d_y_list));
}

void DecisionData::printSomething()
{
	cutilSafeCall(cudaMemcpy(h_t, d_primary_ray_t_value, sizeof(float) * IMAGE_SIZE, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_t2, d_primary_ray_normal, sizeof(float) * IMAGE_SIZE * 3, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_tt, d_intersect_id, sizeof(int) * IMAGE_SIZE, cudaMemcpyDeviceToHost));

	for (int i=0; i<IMAGE_SIZE; i++)
	{
		if (h_t[i] > 0)
			printf("%f\t%f %f %f\t%d\n", h_t[i], h_t2[i*3+0], h_t2[i*3+1], h_t2[i*3+2], h_tt[i]);
	}

	return;
}

void DecisionData::init_cudpp_plans()
{
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SORT_RADIX;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	result = cudppPlan(&sortPlan, config, IMAGE_SIZE, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		printf("error creating CUDPP plan for sorting\n");
		exit(-1);
	}

	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SEGMENTED_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	result = cudppPlan(&segScanPlan, config, IMAGE_SIZE, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		printf("error creating CUDPP plan for segmented scan\n");
		exit(-1);
	}

	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_COMPACT;

	result = cudppPlan(&compactPlan, config, IMAGE_SIZE, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		printf("error creating CUDPP plan for stream compaction\n");
		exit(-1);
	}

	return;
}

void DecisionData::sort_by_block_indices()
{
	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer(&timer));
	CUT_SAFE_CALL( cutStartTimer(timer));
	cudaThreadSynchronize();
	cudppSort(sortPlan, &d_map[IMAGE_SIZE], (void *) &d_map[0], 15, IMAGE_SIZE);
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));

	printf("time taken to sort : %f (msec)\n", cutGetTimerValue(timer));
}

void DecisionData::get_pseudo_valid_array()
{
	unsigned int timer = 0;
	dim3 bsgrid(128, 128, 1);
	dim3 bsthreads(8, 8, 1);

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	cudaThreadSynchronize();
	blockScan<<< bsgrid, bsthreads, 0 >>>(&d_map[IMAGE_SIZE], d_valid, d_scratchValid, IMAGE_SIZE);
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));

	printf("time taken to get pseudo valid array : %f (msec)\n", cutGetTimerValue(timer));

	return;
}

void DecisionData::segment_pseudo_valid_array()
{
	unsigned int timer = 0;

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	cudaThreadSynchronize();
	cudppSegmentedScan(segScanPlan, d_segScanArr, d_scratchValid, d_valid, IMAGE_SIZE);
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));

	printf("time taken to do a segmented scan : %f (msec)\n", cutGetTimerValue(timer));

	return;
}

void DecisionData::get_cuda_block_array()
{
	dim3 bsgrid(128, 128, 1);
	dim3 bsthreads(8, 8, 1);
	unsigned int timer = 0;

	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));
	cudaThreadSynchronize();
	preStreamCompaction<<< bsgrid, bsthreads, 0 >>>(d_segScanArr, d_valid, MAX_RAYS_PER_BLOCK);    // right now threshold = 128 => log(128) = 7
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));

	printf("time get cuda block valid array : %f (msec)\n", cutGetTimerValue(timer));

	return;
}

void DecisionData::copy_thread_id()
{
	dim3 bsgrid(128, 128, 1);
	dim3 bsthreads(8, 8, 1);
	unsigned int timer = 0;

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	cudaThreadSynchronize();
	tag_thread<<< bsgrid, bsthreads, 0 >>>(d_scanArray);
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));

	printf("time copy threadId to thread : %f (msec)\n", cutGetTimerValue(timer));

	return;
}

void DecisionData::stream_compact()
{
	unsigned int timer = 0;

	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));
	cudaThreadSynchronize();
	cudppCompact(compactPlan, d_prefixMap, d_numCudaBlocks, d_scanArray, d_valid, IMAGE_SIZE);
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));

	cutilSafeCall(cudaMemcpy(h_numCudaBlocks, d_numCudaBlocks, sizeof(size_t), cudaMemcpyDeviceToHost));

	printf("time to finally get the prefix array : %f (msec)\n", cutGetTimerValue(timer));
	printf("number of CUDA blocks to be assigned : %d\n", (int) *h_numCudaBlocks);

	return;
}

#endif
