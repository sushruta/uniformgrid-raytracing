__device__ bool
isSmaller(float *a, float *b, float *ref)
{
	float epsilon = 1e-03;
	float distance_a = sqrt((a[0]-ref[0])*(a[0]-ref[0]) + (a[1]-ref[1])*(a[1]-ref[1]) + (a[2]-ref[2])*(a[2]-ref[2]));
	float distance_b = sqrt((b[0]-ref[0])*(b[0]-ref[0]) + (b[1]-ref[1])*(b[1]-ref[1]) + (b[2]-ref[2])*(b[2]-ref[2]));

	if (distance_a + epsilon < distance_b)
		return true;
	return false;
}

__device__ float intersectTri(volatile float tvec[3], volatile float edge1[3], volatile float edge2[3], float dir[3], float oldt)
{
	float   u=0.0,v=0.0,t=0.0;
	float   pvec[3], qvec[3];
	float   det,inv_det;

	// begin calculating determinant - also used to calculate U parameter
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
	float retValue = 0;

	if (t < oldt)
		retValue = t;

	return retValue;
}

__global__ void
mod_light_rckernel(unsigned int *curflist,float *dd_vertlist,int *dd_trilist, unsigned int *blockcnt,unsigned int *blockcntscan,float *t_value_list,float *ray_direction_list,int *is_shadowed,unsigned int *d_map,unsigned int *d_prefixmap,float *cmPt,int size)
{
	extern __shared__ int sharedmem[];
	// size of metadata will be 7
	// 0 -> the starting point of its set of blocks
	// 1 -> number of blocks it has to read
	// 2 -> number of batches
	// 3 -> whether the beam is done
	// 4 -> count of the number of triangles in the bin
	// 5 -> pitch of the triangle count
	// 6 -> number of batches of triangles
	volatile int*   metadata = (int *) &sharedmem;
	volatile float* vertexData =    (float *) &metadata[7];
	volatile int*   rayDoneMap =    (int *) &vertexData[MAX_TRIANGLES * 9];

	int blockIndex = blockIdx.x + gridDim.x * blockIdx.y;
	int curthread = threadIdx.x + NUM_THREADS_X * threadIdx.y;
	int offset = 0, batchCount = 0;

	if (curthread == 0)
	{
		if (blockIndex+offset < size)
		{
			if (blockIndex+offset == 0)
			{
				metadata[0] = 0;
				metadata[1] = d_prefixmap[blockIndex+offset];
			}
			else
			{
				metadata[0] = d_prefixmap[blockIndex+offset-1];
				metadata[1] = d_prefixmap[blockIndex+offset] - d_prefixmap[blockIndex+offset-1];
			}
			metadata[2] = metadata[1] / TOTAL_THREADS + 1;
			metadata[3] = 0;
		}
		else
		{
			metadata[1] = 0;
			metadata[2] = 0;
			metadata[3] = 0;
		}
	}

	__syncthreads();

	for (int i=0; i<metadata[2]; i++)
	{
		rayDoneMap[i*TOTAL_THREADS + curthread] = 0;
	}
	metadata[3] = 0;

	for (int p=0; p<NUM_SLABS; p++)
	{
		if (!metadata[3])
		{
			if (curthread == 0)
			{
				metadata[4] = blockcnt[d_map[IMAGE_SIZE + (metadata[0] + curthread)]*NUM_SLABS + p];
				// from where do these triangles start?
				metadata[5] = blockcntscan[d_map[IMAGE_SIZE + (metadata[0] + curthread)]*NUM_SLABS + p];
				// how many batches are there?
				metadata[6] = (int) ( metadata[4] / MAX_TRIANGLES ) + 1;
			}

			__syncthreads();

			for (int k=0; k<metadata[6]; k++)
			{
				(metadata[4] > MAX_TRIANGLES) ? batchCount=MAX_TRIANGLES : batchCount = metadata[4];
				if (curthread < batchCount)
				{
					int curface = curflist[metadata[5] + curthread];

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
				if (curthread == 0)
				{
					metadata[4] -= MAX_TRIANGLES;
					metadata[5] += MAX_TRIANGLES;
				}

				__syncthreads();

				for (int j=0; j<metadata[2]; j++)
				{
					int batchRayCount;
					float rayDirection[3], ptIntersection[3];

					(metadata[1] > TOTAL_THREADS) ? batchRayCount = TOTAL_THREADS : batchRayCount = metadata[1];
					if (curthread < batchRayCount)
					{
						// now get the pixelID
						int pseudoPixelId = d_map[(metadata[0] + curthread)];
						// get the data from d_ptAttributes, d_ptFlags 
						// corresponding to this pseudoPixelId
						// >>>>>>>>>>>>>>> LOTS OF TIME WASTED HERE... DO SOMETHING TO AVOID THIS 6 TIME GLOBAL ACCESS!! <<<<<<<<<<<<
						float tVal = t_value_list[pseudoPixelId];
						rayDirection[0] = cmPt[0] + tVal * ray_direction_list[pseudoPixelId*3 + 0];
						rayDirection[1] = cmPt[1] + tVal * ray_direction_list[pseudoPixelId*3 + 1];
						rayDirection[2] = cmPt[2] + tVal * ray_direction_list[pseudoPixelId*3 + 2];

						ptIntersection[0] = rayDirection[0];
						ptIntersection[1] = rayDirection[1];
						ptIntersection[2] = rayDirection[2];

						rayDirection[0] -= dd_camcoords[0];
						rayDirection[1] -= dd_camcoords[1];
						rayDirection[2] -= dd_camcoords[2];

						NORMALIZE(rayDirection);

						if (rayDoneMap[j*TOTAL_THREADS + curthread] != 2)
						{
							for ( int i=0; i < batchCount; i++ )
							{
								float value = intersectTri(&vertexData[i*9+0],&vertexData[i*9+3],&vertexData[i*9+6],rayDirection,999999.9f);
								if (value)
								{
									float pt[3];
									pt[0] = dd_camcoords[0] + value * rayDirection[0];
									pt[1] = dd_camcoords[1] + value * rayDirection[1];
									pt[2] = dd_camcoords[2] + value * rayDirection[2];

									if (isSmaller(pt, ptIntersection, dd_camcoords))
									{
										rayDoneMap[j*TOTAL_THREADS + curthread] = 2;
										is_shadowed[pseudoPixelId] = 1;
									}
								}
							}
						}
					}
					else
					{
						rayDoneMap[j*TOTAL_THREADS + curthread] = 2;
					}

					__syncthreads();

					if (curthread == 0)
					{
						metadata[0] += TOTAL_THREADS;
						metadata[1] -= TOTAL_THREADS;
					}

					__syncthreads();
				}

				// update back metadata[0], metadata[1] values here

				if (curthread == 0)
				{
					if (blockIndex+offset < size)
					{
						if (blockIndex+offset == 0)
						{
							metadata[0] = 0;
							metadata[1] = d_prefixmap[blockIndex+offset];
						}
						else
						{
							metadata[0] = d_prefixmap[blockIndex+offset-1];
							metadata[1] = d_prefixmap[blockIndex+offset] - d_prefixmap[blockIndex+offset-1];
						}
					}
					else
					{
						metadata[1] = 0;
					}
				}
			}

			__syncthreads();


			if (threadIdx.x == 0)
				metadata[3] = 1;

			__syncthreads();

			for (int i=0; i<metadata[2]; i++)
			{
				int batchRayCount;
				(metadata[1] > TOTAL_THREADS) ? batchRayCount = TOTAL_THREADS : batchRayCount = metadata[1];
				if (curthread < batchRayCount)
				{
					if (!rayDoneMap[i*TOTAL_THREADS + curthread])
						metadata[3] = 0;
				}
			}
			__syncthreads();
		}
	}

	return;
}

