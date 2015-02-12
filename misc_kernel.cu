#ifndef MISC_KERNEL_H
#define MISC_KERNEL_H

__global__ void do_scan_dump(unsigned int *key_list, unsigned int *threadArr, unsigned int *binaryPositionArr, int total_triangles )
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int thId = bx * 256 + tx;

	if ( thId < total_triangles )
	{
		threadArr[thId] = thId;
		if (thId == 0)
			binaryPositionArr[thId] = 1;
		else
		{
			if (key_list[thId] != key_list[thId-1])
				binaryPositionArr[thId] = 1;
			else
				binaryPositionArr[thId] = 0;
		}
	}
}

__global__ void set_as_zero(unsigned int *arr)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int thId = bx * 256 + tx;
	if (thId < NUM_BLOCKS_X * NUM_BLOCKS_Y * NUM_SLABS)
		arr[thId] = 0;
}

__global__ void create_histogram(unsigned int *compactedArr, int compactedSize, int totalSize, unsigned int *key_list, unsigned int *hist)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int thId = bx * 256 + tx;

	int span, offset;
	if (thId < compactedSize)
	{
		if (thId == compactedSize - 1)
		{
			span = totalSize - compactedArr[thId];
			offset = compactedArr[thId];
		}
		else
		{
			span = compactedArr[thId + 1] - compactedArr[thId];
			offset = compactedArr[thId];
		}

		unsigned int blockID = key_list[offset];
		hist[blockID] = span;
	}
}

// map sort kernel for light and shadows
__global__ void
mapSortkernel(float *t_val_list, float *ray_direction, unsigned int *map, float *cmPt, float nearPlane)
{
	// cmPt -> camera's point. After changing dd_camcoords, we no longer have the position of camera in the GPU
	int pixelId = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);

	// get the point of intersection
	// of the ray corresponding to
	// this pixel
	float ptIntersection[3];
	
	float tVal = t_val_list[pixelId];
	
	map[pixelId] = pixelId  ;
	map[IMAGE_SIZE + pixelId] = 0 ;

	ptIntersection[0] = cmPt[0] + tVal * ray_direction[pixelId*3 + 0];// cameraRayDirection[0];
	ptIntersection[1] = cmPt[1] + tVal * ray_direction[pixelId*3 + 1];// cameraRayDirection[1];
	ptIntersection[2] = cmPt[2] + tVal * ray_direction[pixelId*3 + 2];// cameraRayDirection[2];

	// now construct the ray from
	// the light to the point
	// of intersection
	float lightRayDirection[3];
	lightRayDirection[0] = ptIntersection[0] - dd_camcoords[0];
	lightRayDirection[1] = ptIntersection[1] - dd_camcoords[1];
	lightRayDirection[2] = ptIntersection[2] - dd_camcoords[2];
	NORMALIZE(lightRayDirection);

	// get the plane eq. of the near plane
	// make sure you don't normalize it as
	// we need the distances from it
	float frustum_plane[5];
	frustum_plane[0] = dd_camcoords[48+3] + dd_camcoords[48+2];
	frustum_plane[1] = dd_camcoords[48+7] + dd_camcoords[48+6];
	frustum_plane[2] = dd_camcoords[48+11] + dd_camcoords[48+10];
	frustum_plane[3] = dd_camcoords[48+15] + dd_camcoords[48+14];
	frustum_plane[4] = sqrt(frustum_plane[0]*frustum_plane[0] + frustum_plane[1]*frustum_plane[1] + frustum_plane[2]*frustum_plane[2]);

	tVal = -1.0f * (DOT(frustum_plane, dd_camcoords) + frustum_plane[3]) / (DOT(frustum_plane, lightRayDirection));

	// get that point on the near plane
	// where the ray intersects the plane
	float ptX = tVal * lightRayDirection[0];
	float ptY = tVal * lightRayDirection[1];
	float ptZ = tVal * lightRayDirection[2];

	// now determine the right 
	// and up distances
	float rightDistance = ptX * dd_camcoords[16+0] + ptY * dd_camcoords[16+4] + ptZ * dd_camcoords[16+8];
	float upDistance = ptX * dd_camcoords[16+1] + ptY * dd_camcoords[16+5] + ptZ * dd_camcoords[16+9];

	int blx = floor(((rightDistance + nearPlane*tanf(FOVY*M_PI/360.0f)) / (2.0 * nearPlane * tanf(FOVY*M_PI/360.0f))) * NUM_BLOCKS_X);
	int bly = floor(((upDistance + nearPlane*tanf(FOVY*M_PI/360.0f)) / (2.0 * nearPlane * tanf(FOVY*M_PI/360.0f))) * NUM_BLOCKS_Y);

	unsigned int blockIndex = NUM_BLOCKS_X * NUM_BLOCKS_Y;
	if (blx >= 0 && blx <= NUM_BLOCKS_X-1 && bly >= 0 && bly <= NUM_BLOCKS_Y-1)
	{
		blockIndex = blx * gridDim.y + bly;
	}
	else
	{
		blockIndex = NUM_BLOCKS_X * NUM_BLOCKS_Y;
	}

	map[IMAGE_SIZE + pixelId] = blockIndex;
}

__device__ float
get_x_angle(float *vec)
{
        float upDotValue = vec[0] * dd_camcoords[16+1] + vec[1] * dd_camcoords[16+5] + vec[2] * dd_camcoords[16+9];
        float tmp[3];
        tmp[0] = vec[0] - upDotValue * dd_camcoords[16+1];
        tmp[1] = vec[1] - upDotValue * dd_camcoords[16+5];
        tmp[2] = vec[2] - upDotValue * dd_camcoords[16+9];
        float val = getMagnitude(tmp);
        tmp[0] /= val;
        tmp[1] /= val;
        tmp[2] /= val;

        float forwardDotValue = tmp[0] * dd_camcoords[16+2] + tmp[1] * dd_camcoords[16+6] + tmp[2] * dd_camcoords[16+10];
        float angle = acosf(forwardDotValue);
        return angle;
}

/*__device__ unsigned int
getEffective_x(float *vec, float max)
{
        float upDotValue = vec[0] * dd_camcoords[16+1] + vec[1] * dd_camcoords[16+5] + vec[2] * dd_camcoords[16+9];
        float tmp[3];
        tmp[0] = vec[0] - upDotValue * dd_camcoords[16+1];
        tmp[1] = vec[1] - upDotValue * dd_camcoords[16+5];
        tmp[2] = vec[2] - upDotValue * dd_camcoords[16+9];
        float val = getMagnitude(tmp);
        tmp[0] /= val;
        tmp[1] /= val;
        tmp[2] /= val;

        float forwardDotValue = tmp[0] * dd_camcoords[16+2] + tmp[1] * dd_camcoords[16+6] + tmp[2] * dd_camcoords[16+10];
        float angle = acosf(forwardDotValue);

        float rightDotValue = tmp[0] * dd_camcoords[16+0] + tmp[1] * dd_camcoords[16+4] + tmp[2] * dd_camcoords[16+8];

        unsigned int blx;

        if (rightDotValue > 0)
                blx = NUM_BLOCKS_X / 2 + (int) ((angle / max) * (NUM_BLOCKS_X / 2));
        else
                blx = NUM_BLOCKS_X / 2 - (int) ((angle / max) * (NUM_BLOCKS_X / 2));

        return blx;
}*/

__device__ float
get_y_angle(float *vec)
{
        float tmp[3];
        float rightDotValue = vec[0] * dd_camcoords[16+0] + vec[1] * dd_camcoords[16+4] + vec[2] * dd_camcoords[16+8];

        tmp[0] = vec[0] - rightDotValue * dd_camcoords[16+0];
        tmp[1] = vec[1] - rightDotValue * dd_camcoords[16+4];
        tmp[2] = vec[2] - rightDotValue * dd_camcoords[16+8];
        float val = getMagnitude(tmp);
        tmp[0] /= val;
        tmp[1] /= val;
        tmp[2] /= val;

        float forwardDotValue = tmp[0] * dd_camcoords[16+2] + tmp[1] * dd_camcoords[16+6] * tmp[2] * dd_camcoords[16+10];
        float angle = acos(forwardDotValue);
        return angle;
}

/*__device__ unsigned int
getEffective_y(float *vec, float max)
{
        float tmp[3];
        float rightDotValue = vec[0] * dd_camcoords[16+0] + vec[1] * dd_camcoords[16+4] + vec[2] * dd_camcoords[16+8];

        tmp[0] = vec[0] - rightDotValue * dd_camcoords[16+0];
        tmp[1] = vec[1] - rightDotValue * dd_camcoords[16+4];
        tmp[2] = vec[2] - rightDotValue * dd_camcoords[16+8];

        float val = getMagnitude(tmp);
        tmp[0] /= val;
        tmp[1] /= val;
        tmp[2] /= val;

        float upDotValue = tmp[0] * dd_camcoords[16+1] + tmp[1] * dd_camcoords[16+5] + tmp[2] * dd_camcoords[16+9];
        float forwardDotValue = tmp[0] * dd_camcoords[16+2] + tmp[1] * dd_camcoords[16+6] * tmp[2] * dd_camcoords[16+10];

        float angle = acos(forwardDotValue);

        unsigned int bly;
        if (upDotValue > 0)
                bly = NUM_BLOCKS_Y / 2 + (angle / max) * (NUM_BLOCKS_Y / 2);
        else
                bly = NUM_BLOCKS_Y / 2 - (angle / max) * (NUM_BLOCKS_Y / 2);

        return bly;
}*/

__global__ void
map_spherical_Sortkernel(float *t_val_list, float *ray_direction, float *x_list, float *y_list, float *angle_list, float *cmPt)
{
	// cmPt -> camera's point. After changing dd_camcoords, we no longer have the position of camera in the GPU
	int pixelId = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);

	// get the point of intersection
	// of the ray corresponding to
	// this pixel
	float ptIntersection[3];
	
	float tVal = t_val_list[pixelId];
	
	ptIntersection[0] = cmPt[0] + tVal * ray_direction[pixelId*3 + 0];// cameraRayDirection[0];
	ptIntersection[1] = cmPt[1] + tVal * ray_direction[pixelId*3 + 1];// cameraRayDirection[1];
	ptIntersection[2] = cmPt[2] + tVal * ray_direction[pixelId*3 + 2];// cameraRayDirection[2];

	// now construct the ray from
	// the light to the point
	// of intersection
	float lightRayDirection[3];
	lightRayDirection[0] = ptIntersection[0] - dd_camcoords[0];
	lightRayDirection[1] = ptIntersection[1] - dd_camcoords[1];
	lightRayDirection[2] = ptIntersection[2] - dd_camcoords[2];
	NORMALIZE(lightRayDirection);

	x_list[pixelId] = get_x_angle(lightRayDirection);
	y_list[pixelId] = get_y_angle(lightRayDirection);
}

__global__ void
mapSort_Effective_kernel(float *t_value_list, float *ray_direction, unsigned int *d_map, float *cmPt, float xM, float yM)
{
	// cmPt -> camera's point. After changing dd_camcoords, we no longer have the position of camera in the GPU
	int pixelId = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);

	// get the point of intersection
	// of the ray corresponding to
	// this pixel
	float ptIntersection[3];

	float tVal = t_value_list[pixelId];

	ptIntersection[0] = cmPt[0] + tVal * ray_direction[pixelId*3 + 0];// cameraRayDirection[0];
	ptIntersection[1] = cmPt[1] + tVal * ray_direction[pixelId*3 + 1];// cameraRayDirection[1];
	ptIntersection[2] = cmPt[2] + tVal * ray_direction[pixelId*3 + 2];// cameraRayDirection[2];

	// now construct the ray from
	// the light to the point
	// of intersection
	float lightRayDirection[3];
	lightRayDirection[0] = ptIntersection[0] - dd_camcoords[0];
	lightRayDirection[1] = ptIntersection[1] - dd_camcoords[1];
	lightRayDirection[2] = ptIntersection[2] - dd_camcoords[2];
	NORMALIZE(lightRayDirection);

	int blx = getEffective_x(lightRayDirection, xM);
	int bly = getEffective_y(lightRayDirection, yM);

	int blockIndex;
	if (blx >= 0 && blx < NUM_BLOCKS_X && bly >= 0 && bly < NUM_BLOCKS_Y)
	{
		blockIndex = blx * gridDim.y + bly;
	}
	else
	{
		blockIndex = NUM_BLOCKS_X * NUM_BLOCKS_Y;
	}

	d_map[pixelId] = pixelId;
	d_map[IMAGE_SIZE + pixelId] = blockIndex;
}

__global__ void blockScan(unsigned int *iArr, unsigned int *oArr, unsigned int *pseudoArr, int size)
{
	int pixelId = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);
	oArr[pixelId] = 0;
	pseudoArr[pixelId] = 1;


	if (pixelId == 0)
		oArr[pixelId] = 1;
	else
	{
		if (iArr[pixelId] != iArr[pixelId-1])
			oArr[pixelId] = 1;
	}

	return;
}

__global__ void preStreamCompaction(unsigned int *iArr, unsigned int *oArr, int threshold)
{
	int pixelId = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);
	(iArr[pixelId] % threshold/*& (threshold - 1)*/ == 1) ? oArr[pixelId] = 1 : oArr[pixelId] = 0; //iArr[pixelId] % 128;//& log_threshold;

	/*if (pixelId == 0)
	  oArr[pixelId] = 1;*/

	return;
}

__global__ void tag_thread(unsigned int *threadArr)
{
	int pixelId = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);

	threadArr[pixelId] = pixelId;
	return;
}

#endif
