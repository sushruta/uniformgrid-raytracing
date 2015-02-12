#ifndef SHADER_H
#define SHADER_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda_gl_interop.h>
#include <cutil.h>
#include <cutil_inline.h>

#include "cudpp/cudpp.h"
#include "shader_kernel.cu"

class Shader
{
	public:
		Shader();
		~Shader();

		void add_shadows(unsigned char *d_img, int *dd_shadowed);

		void simpleShade(unsigned char *d_img, float *dd_normal, float *dd_t_value, float *dd_dir, int *dd_intersect_id, float *d_cam_pos, int *mat_idx, float *mat_list, int num_mat);
		void spotlight_shade(unsigned char *d_img, float *dd_normal, float *dd_t_value, float *dd_dir, int *dd_intersect_id, float *d_cam_pos, int *mat_idx, float *mat_list, int num_mat);
		void perlinShade(unsigned char *d_img, float *t_value_list, float *ray_dir, float *camPt, int *dd_intersect_id);

	private:
		CUDPPConfiguration config;
		CUDPPResult result;
		CUDPPHandle randPlan;

		unsigned int *d_out;
};

Shader::Shader()
{
	/*cutilSafeCall(cudaMalloc((void **) &d_out, sizeof(unsigned int) * IMAGE_SIZE));

	config.datatype = CUDPP_UINT;
	config.algorithm = CUDPP_RAND_MD5;
	result = cudppPlan(&randPlan, config, IMAGE_SIZE, 1, 0);

	if (result != CUDPP_SUCCESS)
	{
		fprintf(stderr, "error creating plan\n");
		exit(-1);
	}

	cudaThreadSynchronize();
	cudppRand(randPlan, (void *) d_out, IMAGE_SIZE);
	cudaThreadSynchronize();*/
}

Shader::~Shader()
{
	cutilSafeCall(cudaFree(d_out));
}

void Shader::add_shadows(unsigned char *d_img, int *dd_shadows)
{
	dim3 shadeGridSize (NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 shadeThreadSize (NUM_THREADS_X, NUM_THREADS_Y, 1);
	
	cudaThreadSynchronize();
	shadow_kernel<<< shadeGridSize, shadeThreadSize >>>(d_img, dd_shadows);
	cudaThreadSynchronize();

	return;
}

void Shader::simpleShade(unsigned char *d_img, float *dd_normal, float *dd_t_value, float *dd_dir, int *dd_intersect_id, float *d_cam_pos, int *mat_idx, float *mat_list, int num_mat)
{
	unsigned int shader_time = 0;

	dim3 shadeGridSize (NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 shadeThreadSize (NUM_THREADS_X, NUM_THREADS_Y, 1);

	CUT_SAFE_CALL(cutCreateTimer(&shader_time));
	CUT_SAFE_CALL(cutStartTimer(shader_time));

	cudaThreadSynchronize();
	lambertian_shade<<< shadeGridSize, shadeThreadSize >>>(d_img, dd_normal, dd_t_value, dd_dir, dd_intersect_id, d_cam_pos, mat_idx, mat_list, num_mat);
	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(shader_time));
	printf("time taken to shade the scene : %f (ms)\n", cutGetTimerValue(shader_time));
}

void Shader::spotlight_shade(unsigned char *d_img, float *dd_normal, float *dd_t_value, float *dd_dir, int *dd_intersect_id, float *d_cam_pos, int *mat_idx, float *mat_list, int num_mat)
{
	unsigned int shader_time = 0;

	dim3 shadeGridSize (NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 shadeThreadSize (NUM_THREADS_X, NUM_THREADS_Y, 1);

	float *h_dump = (float *) malloc(sizeof(float) * IMAGE_SIZE * 2);
	float *d_dump;
	cutilSafeCall(cudaMalloc((void **) &d_dump, sizeof(float) * IMAGE_SIZE * 2));

	CUT_SAFE_CALL(cutCreateTimer(&shader_time));
	CUT_SAFE_CALL(cutStartTimer(shader_time));

	cudaThreadSynchronize();
	spot_shade<<< shadeGridSize, shadeThreadSize >>>(d_img, dd_normal, dd_t_value, dd_dir, dd_intersect_id, d_cam_pos, mat_idx, mat_list, num_mat, d_dump);
	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(shader_time));

	cutilSafeCall(cudaMemcpy(h_dump, d_dump, sizeof(float) * IMAGE_SIZE * 2, cudaMemcpyDeviceToHost));
	for (int i=0; i<IMAGE_SIZE; i++)
	{
		printf("x : %f\ty : %f\n", h_dump[i*2+0], h_dump[i*2+1]);
	}
	printf("time taken to shade the scene : %f (ms)\n", cutGetTimerValue(shader_time));
}

void Shader::perlinShade(unsigned char *d_img, float *t_value_list, float *ray_dir, float *camPt, int *dd_intersect_id)
{
	unsigned int shader_time = 0;

	dim3 gridSize(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 threadSize(NUM_THREADS_X, NUM_THREADS_Y, 1);

	CUT_SAFE_CALL(cutCreateTimer(&shader_time));
	CUT_SAFE_CALL(cutStartTimer(shader_time));

	cudaThreadSynchronize();
	perlin_noise_shade<<< gridSize, threadSize >>>(d_img, t_value_list, ray_dir, camPt, dd_intersect_id);
	cudaThreadSynchronize();

	CUT_SAFE_CALL(cutStopTimer(shader_time));
}

#endif
