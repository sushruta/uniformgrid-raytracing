#ifndef TRACE_KERNEL_CU
#define TRACE_KERNEL_CU

__device__ float intersectTriUV(volatile float tvec[3], volatile float edge1[3], volatile float edge2[3], float dir[3], float oldt, float *_u, float *_v)
{
	float u=0.0f, v=0.0f, t=0.0f;
	float pvec[3], qvec[3];
	float det, inv_det;

	// cross 1
	CROSS(pvec, dir, edge2);
	
	// if determinant is near zero, ray lies in plane of triangle
	det = DOT(edge1, pvec);
	if (det > -EPSILON && det < EPSILON)
		return 0;

	inv_det = 1.0 / det;

	// calculate U parameter and test bounds
	u = DOT(tvec, pvec) * inv_det;
	if (u < 0.0 || u > 1.0)
		return 0;

	// prepare to test V parameter
	CROSS(qvec, tvec, edge1);

	// calculate V parameter and test bounds
	v = DOT(dir, qvec) * inv_det;
	if (v < 0.0 || u + v > 1.0)
		return 0;

	// calculate t, ray intersects triangle
	t = DOT(edge2, qvec) * inv_det;
	(t < 0) ? t *= -1 : t = t;
	float retValue=0;
	
	if (t<oldt && t>0)
		retValue = t;

	*_u = u;
	*_v = v;

	return retValue;
}

__device__ void
mulMatrixVector_d(float *result, float *mat, float *vec)
{
	result[0] =  mat[0] * vec[0] + mat[4] * vec[1] + mat[8] * vec[2] + mat[12] * vec[3];
	result[1] =  mat[1] * vec[0] + mat[5] * vec[1] + mat[9] * vec[2] + mat[13] * vec[3];
	result[2] =  mat[2] * vec[0] + mat[6] * vec[1] + mat[10] * vec[2] + mat[14] * vec[3];
	result[3] =  mat[3] * vec[0] + mat[7] * vec[1] + mat[11] * vec[2] + mat[15] * vec[3];
}

__device__ int
isWithin(int rayDone, float t, float *ray_dir, int slab_idx)
{
	int ret_value;
	if (rayDone == 0 || rayDone == 2)
	{
		ret_value = 0;
	}
	else
	{
		float point[4], tmp[4];
		point[0] = dd_camcoords[0] + t * ray_dir[0];
		point[1] = dd_camcoords[1] + t * ray_dir[1];
		point[2] = dd_camcoords[2] + t * ray_dir[2];
		point[3] = 1.0f;

		mulMatrixVector_d(tmp, &dd_camcoords[48], point);
		tmp[2] /= tmp[3];
		int z_value = floor(tmp[2] * NUM_SLABS);
		if (z_value == slab_idx)
			ret_value = 2;
		else
			ret_value = 1;
	}

	return ret_value;
}

__global__ void
rckernel_alpha(unsigned *d_value_list, unsigned int *dd_span, unsigned int *dd_offset, float *dd_normal, float *dd_t_value, float *dd_ray_dir, int *dd_shadowed, int *dd_intersect_id, float *dd_vertlist, int *dd_trilist)
{
	float4 rayDir;
	float ray_direction[3];
	int rayDone = 0;

	int pixelID = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);
	float *d_normals = &dd_normal[pixelID*3];
	float *d_direction = &dd_ray_dir[pixelID*3];

	// direction computation
	float ftx = (float)(blockIdx.x * NUM_THREADS_X + threadIdx.x) / (float)(NUM_BLOCKS_X * NUM_THREADS_X);
	float fty = (float)(blockIdx.y * NUM_THREADS_Y + threadIdx.y) / (float)(NUM_BLOCKS_Y * NUM_THREADS_Y);

	ftx = 1 - ftx;

	// from texture 5x5
	ftx = ftx * (0.9 - 0.1) + 0.1;
	fty = fty * (0.9 - 0.1) + 0.1;

	rayDir = tex2D(texdir, ftx, fty);

	rayDir.x -= dd_camcoords[0];
	rayDir.y -= dd_camcoords[1];
	rayDir.z -= dd_camcoords[2];
	
	ray_direction[0] = rayDir.x;
	ray_direction[1] = rayDir.y;
	ray_direction[2] = rayDir.z;
	NORMALIZE(ray_direction);
	
	extern __shared__ int sharedmem[];
	volatile int *metadata = (int *) &sharedmem;					// size is 4
	volatile float *vertexData = (float *)&metadata[4];				// size is 64 * 9
	volatile float *normals = (float *)&vertexData[MAX_TRIANGLES * 9];		// size is 64 * 3

	__syncthreads();

	int curface, batchCount = 0;
	int curthread = (threadIdx.y*NUM_THREADS_X+threadIdx.x);
	float oldt = 99999999.9f, value;

	float e1[3], e2[3];
	float u, v;
	int tri_intersected;

	metadata[3] = 0;
	for (int slab=0; slab<NUM_SLABS; slab++)
	{
		if (!metadata[3])
		{
			if (curthread == 0)
			{
				metadata[0] = dd_span[(blockIdx.x * gridDim.y + blockIdx.y)*NUM_SLABS + slab];
				metadata[1] = dd_offset[(blockIdx.x * gridDim.y + blockIdx.y)*NUM_SLABS + slab];

				metadata[2] = (int) (metadata[0] / MAX_TRIANGLES) + 1;
				metadata[3] = 0;
			}

			__syncthreads();

			for (int b=0; b<metadata[2]; b++)
			{
				(metadata[0] > MAX_TRIANGLES) ? batchCount = MAX_TRIANGLES: batchCount = metadata[0];

				if (curthread < batchCount)
				{
					curface = d_value_list[metadata[1] + curthread];

					int face1, face2, face3;
					face1 = 3 * dd_trilist[curface * 3 + 0];
					face2 = 3 * dd_trilist[curface * 3 + 1];
					face3 = 3 * dd_trilist[curface * 3 + 2];

					vertexData[curthread*9 + 0] = dd_vertlist[face1 + 0];
					vertexData[curthread*9 + 1] = dd_vertlist[face1 + 1];
					vertexData[curthread*9 + 2] = dd_vertlist[face1 + 2];

					vertexData[curthread*9 + 3] = dd_vertlist[face2 + 0] - vertexData[curthread*9 + 0];
					vertexData[curthread*9 + 4] = dd_vertlist[face2 + 1] - vertexData[curthread*9 + 1];
					vertexData[curthread*9 + 5] = dd_vertlist[face2 + 2] - vertexData[curthread*9 + 2];

					vertexData[curthread*9 + 6] = dd_vertlist[face3 + 0] - vertexData[curthread*9 + 0];
					vertexData[curthread*9 + 7] = dd_vertlist[face3 + 1] - vertexData[curthread*9 + 1];
					vertexData[curthread*9 + 8] = dd_vertlist[face3 + 2] - vertexData[curthread*9 + 2];

					vertexData[curthread*9 + 0] = dd_camcoords[0] - vertexData[curthread*9 + 0];
					vertexData[curthread*9 + 1] = dd_camcoords[1] - vertexData[curthread*9 + 1];
					vertexData[curthread*9 + 2] = dd_camcoords[2] - vertexData[curthread*9 + 2];
				}

				__syncthreads();

				if (rayDone != 2)
				{
					for (int tri=0; tri<batchCount; tri++)
					{
						float tmp_u, tmp_v;
						value = intersectTriUV(&vertexData[tri*9+0], &vertexData[tri*9+3], &vertexData[tri*9+6], ray_direction, oldt, &tmp_u, &tmp_v);
						if (value)
						{
							u = tmp_u;
							v = tmp_v;

							oldt = value;
							rayDone = 1;

							tri_intersected = metadata[1] + tri;

							e1[0] = vertexData[tri*9+3];
							e1[1] = vertexData[tri*9+4];
							e1[2] = vertexData[tri*9+5];

							e2[0] = vertexData[tri*9+6];
							e2[1] = vertexData[tri*9+7];
							e2[2] = vertexData[tri*9+8];
						}
					}
				}

				__syncthreads();

				if (curthread == 0)
				{
					metadata[0] -= MAX_TRIANGLES;
					metadata[1] += MAX_TRIANGLES;
				}

				__syncthreads();
			}

			rayDone = isWithin(rayDone, oldt, ray_direction, slab);

			// check if the beam is done?
			if (threadIdx.x == 0)
				metadata[3] = 1;

			__syncthreads();

			if (rayDone != 2)
				metadata[3] = 0;

			__syncthreads();
		}
	}

	if (rayDone == 2)
	{
		NORMALIZE(e1);
		NORMALIZE(e2);

		volatile float *normal = &normals[curthread*3+0];
		CROSS(normal, e1, e2);
		NORMALIZE(normal);

		(normal[0] < 0) ? normal[0]*=-1:normal[0]=normal[0];
		(normal[1] < 0) ? normal[1]*=-1:normal[1]=normal[1];
		(normal[2] < 0) ? normal[2]*=-1:normal[2]=normal[2];

		dd_t_value[pixelID] = oldt;

		dd_shadowed[pixelID] = 0;
		dd_intersect_id[pixelID] = d_value_list[tri_intersected];

		d_normals[0] = normals[curthread*3 + 0];
		d_normals[1] = normals[curthread*3 + 1];
		d_normals[2] = normals[curthread*3 + 2];
	}
	else
	{
		dd_t_value[pixelID] = -1.0f;
		dd_shadowed[pixelID] = 0;
		dd_intersect_id[pixelID] = -2;
		
		d_normals[0] = -1.0f; 
		d_normals[1] = -1.0f; 
		d_normals[2] = -1.0f; 
	}

	d_direction[0] = ray_direction[0];
	d_direction[1] = ray_direction[1];
	d_direction[2] = ray_direction[2];

	__syncthreads();
}

#endif
