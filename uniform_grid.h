#ifndef UNIFORM_GRID_H
#define UNIFORM_GRID_H

#include <stdio.h>
#include <stdlib.h>

#include "cudpp/cudpp.h"
#include "grid_kernel.cu"
#include "misc_kernel.cu"

class UniformGrid
{
	public:
		UniformGrid(int TRI_COUNT);
		~UniformGrid();

		void buildGrid(int *d_facelist, float *d_vertlist);

		unsigned int *d_offset;
		unsigned int *d_span;
	private:
		// all the private variables here
		unsigned int *d_sizeList;
		unsigned int *d_scannedSizeList;

		unsigned int *d_triangle_value_list;
		unsigned int *d_triangle_key_list;

		unsigned int *d_zSlabs;

		unsigned int *d_cell_position;
		unsigned int *d_cell_boundaries;

		unsigned int *d_compacted_list;

		float *d_projCoordZ;
		float *h_projCoordZ;
		
		size_t *d_num_actual_cells;
		size_t *h_num_actual_cells;

		unsigned int total_triangles;
		int TRI_COUNT;
		int *dd_modelParams;

		// cudpp stuff
		CUDPPConfiguration config;
		CUDPPHandle sortPlan;
		CUDPPHandle segScanPlan;
		CUDPPHandle compactPlan;
		CUDPPHandle sortDSPlan;
		CUDPPHandle scanDSPlan;
		CUDPPHandle scanN1N2N3Plan;
		CUDPPResult result;

		// private functions
		int getMinMax_host(float *min, float *max);
		void init_cudpp_plans();
};

UniformGrid::UniformGrid(int TRICOUNT)
{
	// allocate memory here
	TRI_COUNT = TRICOUNT;

	cutilSafeCall(cudaMalloc((void **) &d_span, sizeof(unsigned int) * NUM_BLOCKS_X * NUM_BLOCKS_Y * NUM_SLABS));
	cutilSafeCall(cudaMalloc((void **) &d_offset, sizeof(unsigned int) * NUM_BLOCKS_X * NUM_BLOCKS_Y * NUM_SLABS));

	cutilSafeCall(cudaMalloc((void **) &dd_modelParams, sizeof(int)));
	cutilSafeCall(cudaMemcpy(dd_modelParams, &TRI_COUNT, sizeof(int), cudaMemcpyHostToDevice));
	
	cutilSafeCall(cudaMalloc((void **) &d_sizeList, sizeof(unsigned int) * TRI_COUNT));
	cutilSafeCall(cudaMalloc((void **) &d_scannedSizeList, sizeof(unsigned int) * TRI_COUNT));
	cutilSafeCall(cudaMalloc((void **) &d_projCoordZ, sizeof(float) * TRI_COUNT));
	cutilSafeCall(cudaMalloc((void **) &d_zSlabs, sizeof(unsigned int) * TRI_COUNT));
	cutilSafeCall(cudaMalloc((void **) &d_num_actual_cells, sizeof(size_t)));
	cutilSafeCall(cudaMalloc((void **) &d_compacted_list, sizeof(unsigned int) * NUM_BLOCKS_X * NUM_BLOCKS_Y * NUM_SLABS));

	h_projCoordZ = (float *) malloc(sizeof(float) * TRI_COUNT);
	h_num_actual_cells = (size_t *) malloc(sizeof(size_t));

	init_cudpp_plans();
}

UniformGrid::~UniformGrid()
{
	// free all the memory allocated
	cutilSafeCall(cudaFree(d_sizeList));
	cutilSafeCall(cudaFree(d_projCoordZ));
	cutilSafeCall(cudaFree(d_scannedSizeList));
	cutilSafeCall(cudaFree(d_zSlabs));
	cutilSafeCall(cudaFree(d_num_actual_cells));
	cutilSafeCall(cudaFree(d_compacted_list));

	cutilSafeCall(cudaFree(dd_modelParams));

	cutilSafeCall(cudaFree(d_offset));
	cutilSafeCall(cudaFree(d_span));

	free(h_projCoordZ);
	free(h_num_actual_cells);
}

void UniformGrid::init_cudpp_plans()
{
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SORT_RADIX;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	result = cudppPlan(&sortPlan, config, IMAGE_SIZE, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		fprintf(stderr, "error creating CUDPP plan for sorting\n");
		exit(-1);
	}

	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SEGMENTED_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	result = cudppPlan(&segScanPlan, config, IMAGE_SIZE, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		fprintf(stderr, "error creating CUDPP plan for segmented scan\n");
		exit(-1);
	}

	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_COMPACT;

	result = cudppPlan(&compactPlan, config, IMAGE_SIZE, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		fprintf(stderr, "error creating CUDPP plan for stream compaction\n");
		exit(-1);
	}

	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

	result = cudppPlan(&scanDSPlan, config, TRI_COUNT, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		fprintf(stderr, "error creating CUDPP plan for scan\n");
		exit(-1);
	}

	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SCAN;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

	result = cudppPlan(&scanN1N2N3Plan, config, NUM_BLOCKS_X * NUM_BLOCKS_Y * NUM_SLABS, 1, 0);

	if (CUDPP_SUCCESS != result)
	{
		printf("error creating CUDPP plan for N1N2N3 scan\n");
		exit(-1);
	}

	return;
}

int UniformGrid::getMinMax_host(float *zmin, float *zmax)
{
	cutilSafeCall(cudaMemcpy(h_projCoordZ, d_projCoordZ, sizeof(float) * TRI_COUNT, cudaMemcpyDeviceToHost));

	int elems = 0;
	*zmin = +2.0f;
	*zmax = -2.0f;
	for (int i=0; i<TRI_COUNT; i++)
	{
		if (*zmin > h_projCoordZ[i] && h_projCoordZ[i] >= 0.0f)
			*zmin = h_projCoordZ[i];
		if (*zmax < h_projCoordZ[i])
			*zmax = h_projCoordZ[i];

		if (h_projCoordZ[i] >= 0.0f)
			elems++;

		if (h_projCoordZ[i] >= 0.0f)
			printf("elems : %d \t zmin : %f \t zmax : %f \t h_projCoordZ[%d] : %f\n", elems, *zmin, *zmax, i, h_projCoordZ[i]);
	}

	return elems;
}

void UniformGrid::buildGrid(int *d_facelist, float *d_vertlist)
{
	printf("blocks : %d\tthreads : %d\n", NUMBLOCKSDS, NUMTHREADSDS);
	dim3 dsGrid(NUMBLOCKSDS, 1, 1);
	dim3 dsThreads(NUMTHREADSDS, 1, 1);

	// call the DS kernel here
	cudaThreadSynchronize();
	DSKernel<<< dsGrid, dsThreads >>>(d_sizeList, d_projCoordZ, d_facelist, d_vertlist, dd_modelParams);
	cudaThreadSynchronize();

	cutilSafeCall(cudaMemcpy(h_projCoordZ, d_projCoordZ, sizeof(float) * TRI_COUNT, cudaMemcpyDeviceToHost));

	// do the reduction here
	printf("NUMELEMENTS : %d\n", TRI_COUNT);
	float zMin = +2.0f, zMax = -2.0f;
	int elems = 0;
	
	for (int i=0; i<TRI_COUNT; i++)
	{
		if (zMin > h_projCoordZ[i] && h_projCoordZ[i] >= 0.0f)
			zMin = h_projCoordZ[i];

		if (zMax < h_projCoordZ[i])
			zMax = h_projCoordZ[i];

		if (h_projCoordZ[i] >= 0.0f)
		{
			elems++;
			//printf("elems : %d \t zmin : %f \t zmax : %f \t h_projCoordZ[%d] : %f\n", elems, zMin, zMax, i, h_projCoordZ[i]);
		}
	}

	printf("zMin :: %f\tzMax:: %f\n",zMin, zMax);
	printf("Number of elements compared: %d\n", elems);

	// get the running count of
	// the number of elements
	cudaThreadSynchronize();
	cudppScan(scanDSPlan, d_scannedSizeList, d_sizeList, TRI_COUNT);
	cudaThreadSynchronize();

	// get the total triangle count to CPU
	cudaThreadSynchronize();
	cutilSafeCall(cudaMemcpy(&total_triangles, &d_scannedSizeList[TRI_COUNT - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cudaThreadSynchronize();

	printf("total triangles are %d\n", total_triangles);

	// free memory if already allocated
	cutilSafeCall(cudaMalloc((void**)&d_triangle_key_list, sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&d_triangle_value_list, sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&d_cell_position, sizeof(unsigned int)));
	cutilSafeCall(cudaMalloc((void**)&d_cell_boundaries, sizeof(unsigned int)));
	cutilSafeCall(cudaFree(d_triangle_key_list));
	cutilSafeCall(cudaFree(d_triangle_value_list));
	cutilSafeCall(cudaFree(d_cell_position));
	cutilSafeCall(cudaFree(d_cell_boundaries));

	// now allocate the memory
	cutilSafeCall(cudaMalloc((void **) &d_triangle_key_list, sizeof(unsigned int) * total_triangles));
	cutilSafeCall(cudaMalloc((void **) &d_triangle_value_list, sizeof(unsigned int) * total_triangles));
	cutilSafeCall(cudaMalloc((void **) &d_cell_position, sizeof(unsigned int) * total_triangles));
	cutilSafeCall(cudaMalloc((void **) &d_cell_boundaries, sizeof(unsigned int) * total_triangles));

	// now run the slab kernel
	cudaThreadSynchronize();
	SlabKernel<<< dsGrid, dsThreads >>>(d_zSlabs, d_projCoordZ, dd_modelParams, zMin, zMax);
	cudaThreadSynchronize();

	int sharedmem = NUMTHREADSDS + 1;
	cudaThreadSynchronize();
	DSFillkernel<<< dsGrid, dsThreads, sharedmem >>>(d_triangle_key_list, d_triangle_value_list, d_scannedSizeList, d_zSlabs, d_facelist, d_vertlist, dd_modelParams);
	cudaThreadSynchronize();

	// init the cudpp sort configuration
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_SORT_RADIX;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	result = cudppPlan(&sortDSPlan, config, total_triangles, 1, 0);

	// now sort the arrays with key as key_list
	unsigned int sort_time = 0;
	
	CUT_SAFE_CALL(cutCreateTimer(&sort_time));
	CUT_SAFE_CALL(cutStartTimer(sort_time));
	cudppSort(sortDSPlan, d_triangle_key_list, (void *) d_triangle_value_list, 32, total_triangles);
	CUT_SAFE_CALL(cutStopTimer(sort_time));

	printf("time taken to sort the %d pairs : %f (ms)\n",total_triangles, cutGetTimerValue(sort_time));

	int numThreads = 256;
	int numBlocks = (total_triangles / numThreads) + 1;

	dim3 singlegrid(numBlocks, 1, 1);
	dim3 singlethreads(numThreads, 1, 1);

	// first get the boundaries of cells
	// also get the position ID of each cell
	cudaThreadSynchronize();
	do_scan_dump<<< singlegrid, singlethreads, 0 >>>(d_triangle_key_list, d_cell_position, d_cell_boundaries, total_triangles );
	cudaThreadSynchronize();

	// now do a compact using these two arrays
	CUDPPConfiguration conf;
	CUDPPHandle cellCompactPlan = 0;
	CUDPPResult res;

	conf.datatype = CUDPP_UINT;
	conf.algorithm = CUDPP_COMPACT;
	conf.options = CUDPP_OPTION_FORWARD;

	res = cudppPlan(&cellCompactPlan, conf, total_triangles, 1, 0);

	if (res != CUDPP_SUCCESS)
	{
		fprintf(stderr, "error creating cell-triangle compaction plan!\n");
		exit(-1);
	}

	cudaThreadSynchronize();
	cudppCompact(cellCompactPlan, d_compacted_list, d_num_actual_cells, d_cell_position, d_cell_boundaries, total_triangles);
	cudaThreadSynchronize();

	cutilSafeCall(cudaMemcpy(h_num_actual_cells, d_num_actual_cells, sizeof(size_t), cudaMemcpyDeviceToHost));
	//printf("Number of actual cells : %d\n", (int) *h_num_actual_cells);

	numThreads = 256;
	numBlocks = (NUM_BLOCKS_X * NUM_BLOCKS_Y * NUM_SLABS / numThreads) + 1;
	dim3 gridSize(numBlocks, 1, 1);
	dim3 threadSize(numThreads, 1, 1);

	cudaThreadSynchronize();
	set_as_zero<<< gridSize, threadSize >>>(d_span);
	cudaThreadSynchronize();

	//printf("number of distinct cells : %d\tnumber of total triangles : %d\n", (int) *h_num_actual_cells, total_triangles);

	numBlocks = ( (int) *h_num_actual_cells / numThreads ) + 1;
	dim3 histGridSize(numBlocks, 1, 1);
	dim3 histThreadSize(numThreads, 1, 1);

	// now generate the scan array
	cudaThreadSynchronize();
	create_histogram<<< histGridSize, histThreadSize, 0 >>>(d_compacted_list, (int) *h_num_actual_cells, total_triangles, d_triangle_key_list, d_span);
	cudaThreadSynchronize();

	cudaThreadSynchronize();
	cudppScan(scanN1N2N3Plan, d_offset, d_span, NUM_BLOCKS_X * NUM_BLOCKS_Y * NUM_SLABS);
	cudaThreadSynchronize();

	return;
}

#endif
