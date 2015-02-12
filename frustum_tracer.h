#ifndef FRUSTUM_TRACER_H
#define FRUSTUM_TRACER_H

#include <stdio.h>
#include <stdlib.h>

#include <cuda_gl_interop.h>
#include <cutil.h>
#include <cutil_inline.h>

#include "cudpp/cudpp.h"
#include "trace_kernel.cu"

class FrustumTracer
{
	public:
		FrustumTracer();
		~FrustumTracer();

		void trace(unsigned int *d_value_list, unsigned int *dd_span, unsigned int *dd_offset, 
				float *dd_normal, float *dd_t_value, float *dd_ray_dir, 
				int *dd_shadowed, int *dd_intersect_id, 
				float *dd_vertlist, int *dd_trilist);
	private:
};

FrustumTracer::FrustumTracer()
{
}

FrustumTracer::~FrustumTracer()
{
}

void FrustumTracer::trace(unsigned int *d_value_list, unsigned int *dd_span, unsigned int *dd_offset, 
		float *dd_normal, float *dd_t_value, float *dd_ray_dir, 
		int *dd_shadowed, int *dd_intersect_id, 
		float *dd_vertlist, int *dd_trilist)
{
	unsigned int trace_time = 0;

	int sharedmem = (sizeof(int) * 16);
	sharedmem += sizeof(float) * MAX_TRIANGLES * 12;
	dim3 rcgrid (NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 rcthreads (NUM_THREADS_X, NUM_THREADS_Y, 1);
	
	CUT_SAFE_CALL( cutCreateTimer(&trace_time));
	CUT_SAFE_CALL( cutStartTimer(trace_time));

	cudaThreadSynchronize();
	rckernel_alpha<<< rcgrid, rcthreads, sharedmem >>>(d_value_list, dd_span, dd_offset, dd_normal, dd_t_value, dd_ray_dir, dd_shadowed, dd_intersect_id, dd_vertlist, dd_trilist);
	cudaThreadSynchronize();
	
	CUT_SAFE_CALL(cutStopTimer(trace_time));
	printf("time taken to do 1st level raytrace : %f (ms)\n", cutGetTimerValue(trace_time));

	return;
}

#endif
