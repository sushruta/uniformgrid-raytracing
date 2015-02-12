#ifndef GRID_KERNEL_H
#define GRID_KERNEL_H

__device__ void
mulMatrixVector_D ( float *result, float *mat, float *vec )
{
	result[0] =  mat[0] * vec[0] + mat[4] * vec[1] + mat[8] * vec[2] + mat[12] * vec[3];
	result[1] =  mat[1] * vec[0] + mat[5] * vec[1] + mat[9] * vec[2] + mat[13] * vec[3];
	result[2] =  mat[2] * vec[0] + mat[6] * vec[1] + mat[10] * vec[2] + mat[14] * vec[3];
	result[3] =  mat[3] * vec[0] + mat[7] * vec[1] + mat[11] * vec[2] + mat[15] * vec[3];
}

__device__ void
getTransformedVertex(float *vertlist, int faceID, float *vert_mvp, float *vert_mv)
{
	float point[4], tmp[4];
	point[0] = vertlist[faceID+0];
	point[1] = vertlist[faceID+1];
	point[2] = vertlist[faceID+2];
	point[3] = 1.0f;

	mulMatrixVector_D(tmp, &dd_camcoords[16], point);       // modelview matrix
	point[0] = tmp[0] / tmp[3];
	point[1] = tmp[1] / tmp[3];
	point[2] = tmp[2] / tmp[3];
	point[3] = 1.0f;

	vert_mv[0] = point[0];
	vert_mv[1] = point[1];
	vert_mv[2] = point[2];

	mulMatrixVector_D(tmp, &dd_camcoords[32], point);
	vert_mvp[0] = tmp[0] / tmp[3];
	vert_mvp[1] = tmp[1] / tmp[3];
	vert_mvp[2] = tmp[2] / tmp[3];
}

__device__ bool
isFrontFacing(float *v1, float *v2, float *v3)
{
	bool frontFacing = true;
	float edge1[3], edge2[3], faceNormal[3];
	edge1[0] = v3[0] - v1[0];
	edge1[1] = v3[1] - v1[1];
	edge1[2] = v3[2] - v1[2];

	edge2[0] = v3[0] - v2[0];
	edge2[1] = v3[1] - v2[1];
	edge2[2] = v3[2] - v2[2];

	CROSS(faceNormal, edge1, edge2);
	float dotP = DOT(v1, faceNormal);
	if (dotP > 0)
		frontFacing = false;

	return frontFacing;
}

__device__ bool
test_x(float *_v1, float *_v2, float *_v3)
{
	int out_of_range = 0;
	if (_v1[0] <= -1.0f || _v1[0] >= 1.0f)
		out_of_range++;
	if (_v2[0] <= -1.0f || _v2[0] >= 1.0f)
		out_of_range++;
	if (_v3[0] <= -1.0f || _v2[0] >= 1.0f)
		out_of_range++;

	if (out_of_range == 3)
		return false;
}
	
__device__ bool
test_y(float *_v1, float *_v2, float *_v3)
{
	int out_of_range = 0;
	if (_v1[1] <= -1.0f || _v1[1] >= 1.0f)
		out_of_range++;
	if (_v2[1] <= -1.0f || _v2[1] >= 1.0f)
		out_of_range++;
	if (_v3[1] <= -1.0f || _v2[1] >= 1.0f)
		out_of_range++;

	if (out_of_range == 3)
		return false;
}
	
__device__ bool
test_z(float *_v1, float *_v2, float *_v3)
{
	int out_of_range = 0;
	if (_v1[1] <= 0.0f || _v1[1] >= 1.0f)
		out_of_range++;
	if (_v2[1] <= 0.0f || _v2[1] >= 1.0f)
		out_of_range++;
	if (_v3[1] <= 0.0f || _v2[1] >= 1.0f)
		out_of_range++;

	if (out_of_range == 3)
		return false;
}

__device__ bool
isTriangleInside(float *v1, float *v2, float *v3)
{
	bool result = false;
	bool vertex1_inside = false , vertex2_inside = false , vertex3_inside = false ;
	if (v1[0] > -1.0f && v1[0] < 1.0f && v1[1] > -1.0f && v1[1] < 1.0f && v1[2] > 0.0f && v1[2] < 1.0f)
		vertex1_inside = true;

	if (v2[0] > -1.0f && v2[0] < 1.0f && v2[1] > -1.0f && v2[1] < 1.0f && v2[2] > 0.0f && v2[2] < 1.0f)
		vertex2_inside = true;
	
	if (v3[0] > -1.0f && v3[0] < 1.0f && v3[1] > -1.0f && v3[1] < 1.0f && v3[2] > 0.0f && v3[2] < 1.0f)
		vertex3_inside = true;

	if (vertex1_inside || vertex2_inside || vertex3_inside)
		result = true;
	/*else
	{
		// handle the case where the triangle's 
		// vertices may go out of the grid 
		// but it's mostly in the grid :)
		if (test_x(v1, v2, v3) || test_y(v1, v2, v3) || test_z(v1, v2, v3))
			result = true;
	}*/

	return result;
}

__device__ float
min_d(float e1, float e2, float e3)
{
	float value;
	(e1 < e2) ? ((e1 < e3) ? value = e1 : value = e3) : ((e2 < e3) ? value = e2 : value = e3);
	return value;
}
	
__device__ float
max_d(float e1, float e2, float e3)
{
	float value;
	(e1 > e2) ? ((e1 > e3) ? value = e1 : value = e3) : ((e2 > e3) ? value = e2 : value = e3);
	return value;
}

__device__ int
min_d(int e1, int e2)
{
	if (e1 < e2)
		return e1;
	return e2;
}

__device__ int
max_d(int e1, int e2)
{
	if (e1 > e2)
		return e1;
	return e2;
}

__global__ void DSKernel(unsigned int *sizeList, float *projCoordZ, int *facelist, float *vertexlist, int *modelParams)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int curface;
	curface = bx * NUMTHREADSDS + tx;

	if (curface < modelParams[0])
	{
		float v1[3], v2[3], v3[3];
		float v1_mv[3], v2_mv[3], v3_mv[3];

		int face1, face2, face3;

		face1 = 3*facelist[curface*3+0];
		face2 = 3*facelist[curface*3+1];
		face3 = 3*facelist[curface*3+2];

		// query the point(s)
		// point 1
		getTransformedVertex(vertexlist, face1, v1, v1_mv);
		// point 2
		getTransformedVertex(vertexlist, face2, v2, v2_mv);
		// point 3
		getTransformedVertex(vertexlist, face3, v3, v3_mv);

		// check if they are front facing
		bool frontFacing = isFrontFacing(v1_mv, v2_mv, v3_mv);
		bool isInside = isTriangleInside(v1, v2, v3);

		float xmin, xmax, ymin, ymax, zmin;
		int gxmin, gxmax, gymin, gymax, gzmin;

		//if (isInside && !frontFacing)
		if (1)
		{
			xmin = min_d(v1[0], v2[0], v3[0]);
			ymin = min_d(v1[1], v2[1], v3[1]);
			zmin = min_d(v1[2], v2[2], v3[2]);

			xmax = max_d(v1[0], v2[0], v3[0]);
			ymax = max_d(v1[1], v2[1], v3[1]);

			gxmin = floor(((xmin + 1.0f) / 2.0f) * NUM_BLOCKS_X);
			gymin = floor(((ymin + 1.0f) / 2.0f) * NUM_BLOCKS_Y);

			gzmin = floor(zmin * NUM_SLABS); 
			projCoordZ[curface] = zmin;

			gxmax = floor(((xmax + 1.0f) / 2.0f) * NUM_BLOCKS_X);
			gymax = floor(((ymax + 1.0f) / 2.0f) * NUM_BLOCKS_Y);

			(gxmin < 0) ? gxmin = 0 : gxmin=gxmin;
			(gymin < 0) ? gymin = 0 : gymin=gymin;
			(gzmin < 0) ? gzmin = 0 : gzmin=gzmin;

			(gxmin > NUM_BLOCKS_X - 1 ) ? gxmin = NUM_BLOCKS_X - 1 : gxmin = gxmin;
			(gymin > NUM_BLOCKS_Y - 1 ) ? gymin = NUM_BLOCKS_Y - 1 : gymin = gymin;
			(gzmin > NUM_SLABS - 1 ) ? gzmin = NUM_SLABS - 1: gzmin = gzmin;

			(gxmax < 0) ? gxmax = 0 :gxmax=gxmax;
			(gymax < 0) ? gymax = 0 :gymax=gymax;

			(gxmax > NUM_BLOCKS_X - 1 ) ? gxmax = NUM_BLOCKS_X - 1 : gxmax = gxmax;
			(gymax > NUM_BLOCKS_Y - 1 ) ? gymax = NUM_BLOCKS_Y - 1 : gymax = gymax;

			sizeList[curface] = (gxmax - gxmin + 1) * (gymax - gymin + 1);
		}
		else
		{
			sizeList[curface] = 0;
			projCoordZ[curface] = -5.0f;
		}
	}

	__syncthreads();
	
	return;
}

__global__ void DSFillkernel(unsigned int *keyList, unsigned int *valueList, unsigned int *scanList, unsigned int *zSlabs, int *facelist, float *vertexlist, int *modelParams)
{
	int     tx = threadIdx.x;
	int     bx = blockIdx.x;

	int curface;
	curface = bx * NUMTHREADSDS + tx;
	
	if (curface < modelParams[0])
	{
		int offset, span;
		if (curface != 0)
		{
			span = scanList[curface] - scanList[curface-1];
			offset = scanList[curface-1];
		}
		else
		{
			offset = 0;
			span = scanList[curface];
		}

		float v1[3], v2[3], v3[3];
		float v1_mv[3], v2_mv[3], v3_mv[3];

		int face1, face2, face3;

		face1 = 3 * facelist[curface * 3 + 0];
		face2 = 3 * facelist[curface * 3 + 1];
		face3 = 3 * facelist[curface * 3 + 2];

		// query the point(s)
		// point 1
		getTransformedVertex(vertexlist, face1, v1, v1_mv);
		// point 2
		getTransformedVertex(vertexlist, face2, v2, v2_mv);
		// point 3
		getTransformedVertex(vertexlist, face3, v3, v3_mv);

		// check if they are front facing
		bool frontFacing = isFrontFacing(v1_mv, v2_mv, v3_mv);
		bool isInside = isTriangleInside(v1, v2, v3);

		float xmin, xmax, ymin, ymax, zmin;
		int gxmin, gxmax, gymin, gymax, gzmin;

		//if (isInside && !frontFacing)
		if (1)
		{
			xmin = min_d(v1[0], v2[0], v3[0]);
			ymin = min_d(v1[1], v2[1], v3[1]);

			xmax = max_d(v1[0], v2[0], v3[0]);
			ymax = max_d(v1[1], v2[1], v3[1]);

			gxmin = floor(((xmin + 1.0f) / 2.0f) * NUM_BLOCKS_X);
			gymin = floor(((ymin + 1.0f) / 2.0f) * NUM_BLOCKS_Y);

			gzmin = zSlabs[curface];

			gxmax = floor(((xmax + 1.0f) / 2.0f) * NUM_BLOCKS_X);
			gymax = floor(((ymax + 1.0f) / 2.0f) * NUM_BLOCKS_Y);

			gxmin = min_d(max_d(gxmin, 0), NUM_BLOCKS_X - 1);
			gymin = min_d(max_d(gymin, 0), NUM_BLOCKS_Y - 1);
			gzmin = min_d(max_d(gzmin, 0), NUM_SLABS - 1);

			gxmax = max_d(min_d(gxmax, NUM_BLOCKS_X - 1), 0);
			gymax = max_d(min_d(gymax, NUM_BLOCKS_Y - 1), 0);

			int size_x = (gxmax - gxmin + 1);
			int size_y = (gymax - gymin + 1);

			for (int i=0; i<size_x; i++)
			{
				for (int j=0; j<size_y; j++)
				{
					keyList[offset + i*size_y + j] = ((gxmin + i) * NUM_BLOCKS_Y + (gymin + j)) * NUM_SLABS + gzmin;
					valueList[offset + i*size_y + j] = curface;
				}
			}
		}
	}

	__syncthreads();

	return;
}

__global__ void SlabKernel (unsigned int *zList, float *projCoordZ, int *modelParams, float zMin, float zMax )
{
	unsigned int binID;
	int curface = blockIdx.x * NUMTHREADSDS + threadIdx.x;

	if (curface < modelParams[0])
	{
		float pCoord = projCoordZ[curface];
		if (pCoord >= 0.0f)
		{
			binID = (unsigned int) ( ( (unsigned int) (NUM_SLABS * (float) ( ( pCoord - zMin ) / ( zMax - zMin ) ) )) );
			if (binID >= NUM_SLABS)
				binID = NUM_SLABS - 1;
			zList[curface] = binID;
		}
	}

	return;
}

__device__ float
getMagnitude(float *vec)
{
	float rad = 0;
	for (int i=0; i<3; i++)
		rad += vec[i]*vec[i];

	float val = sqrt(rad);
	return val;
}

// get the angle b/w the vector
// and the forward vector
__device__ unsigned int
getBlock_x(float *vec)
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
		blx = NUM_BLOCKS_X / 2 + (int) ((angle / 3.141f) * (NUM_BLOCKS_X / 2));
	else
		blx = NUM_BLOCKS_X / 2 - (int) ((angle / 3.141f) * (NUM_BLOCKS_X / 2));

	return blx;
}

__device__ unsigned int
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
}

__device__ unsigned int
getBlock_y(float *vec)
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
		bly = NUM_BLOCKS_Y / 2 + (angle / 3.141f) * (NUM_BLOCKS_Y / 2);
	else
		bly = NUM_BLOCKS_Y / 2 - (angle / 3.141f) * (NUM_BLOCKS_Y / 2);

	return bly;
}

__device__ unsigned int
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
}

__global__ void DS_spherical_Kernel(unsigned int *sizeList, float *projCoordZ, int *facelist, float *vertexlist, int *modelParams, float x_min, float y_min)
{
	int	tx = threadIdx.x;
	int	bx = blockIdx.x;

	int curface;
	curface = bx * NUMTHREADSDS + tx;

	if (curface < modelParams[0])
	{
		float v_p1[3], v_p2[3], v_p3[3];
		int face1, face2, face3;
		float tmp[4], point[4], tmp2[4];
		float radius1, radius2, radius3;

		face1 = 3*facelist[curface*3+0];
		face2 = 3*facelist[curface*3+1];
		face3 = 3*facelist[curface*3+2];

		// point 1
		point[0] = vertexlist[face1 + 0];
		point[1] = vertexlist[face1 + 1];
		point[2] = vertexlist[face1 + 2];
		point[3] = 1.0f;

		mulMatrixVector_D(tmp, &dd_camcoords[16], point);       // modelview matrix
		tmp2[0] = tmp[0] / tmp[3];
		tmp2[1] = tmp[1] / tmp[3];
		tmp2[2] = tmp[2] / tmp[3];
		tmp2[3] = 1.0f;

		v_p1[0] = tmp2[0];
		v_p1[1] = tmp2[1];
		v_p1[2] = tmp2[2];

		point[0] -= dd_camcoords[0];
		point[1] -= dd_camcoords[1];
		point[2] -= dd_camcoords[2];

		radius1 = getMagnitude(point);
		point[0] /= radius1;
		point[1] /= radius1;
		point[2] /= radius1;

		int blx1, bly1;

		blx1 = getEffective_x(point, x_min);
		bly1 = getEffective_y(point, y_min);

		// point 2
		point[0] = vertexlist[face2 + 0];
		point[1] = vertexlist[face2 + 1];
		point[2] = vertexlist[face2 + 2];
		point[3] = 1.0f;

		mulMatrixVector_D(tmp, &dd_camcoords[16], point);       // modelview matrix
		tmp2[0] = tmp[0] / tmp[3];
		tmp2[1] = tmp[1] / tmp[3];
		tmp2[2] = tmp[2] / tmp[3];
		tmp2[3] = 1.0f;

		v_p2[0] = tmp2[0];
		v_p2[1] = tmp2[1];
		v_p2[2] = tmp2[2];

		point[0] -= dd_camcoords[0];
		point[1] -= dd_camcoords[1];
		point[2] -= dd_camcoords[2];

		radius2 = getMagnitude(point);
		point[0] /= radius2;
		point[1] /= radius2;
		point[2] /= radius2;

		int blx2, bly2;

		blx2 = getEffective_x(point, x_min);
		bly2 = getEffective_y(point, y_min);

		// point 3
		point[0] = vertexlist[face3 + 0];
		point[1] = vertexlist[face3 + 1];
		point[2] = vertexlist[face3 + 2];
		point[3] = 1.0f;

		mulMatrixVector_D(tmp, &dd_camcoords[16], point);       // modelview matrix
		tmp2[0] = tmp[0] / tmp[3];
		tmp2[1] = tmp[1] / tmp[3];
		tmp2[2] = tmp[2] / tmp[3];
		tmp2[3] = 1.0f;

		v_p3[0] = tmp2[0];
		v_p3[1] = tmp2[1];
		v_p3[2] = tmp2[2];

		point[0] -= dd_camcoords[0];
		point[1] -= dd_camcoords[1];
		point[2] -= dd_camcoords[2];

		radius3 = getMagnitude(point);
		point[0] /= radius3;
		point[1] /= radius3;
		point[2] /= radius3;

		int blx3, bly3;

		blx3 = getEffective_x(point, x_min);
		bly3 = getEffective_y(point, y_min);

		// check if the triangle is front facing
		bool frontFacing = true;
		float edge1[3], edge2[3], faceNormal[3];
		edge1[0] = v_p3[0] - v_p1[0];
		edge1[1] = v_p3[1] - v_p1[1];
		edge1[2] = v_p3[2] - v_p1[2];

		edge2[0] = v_p3[0] - v_p2[0];
		edge2[1] = v_p3[1] - v_p2[1];
		edge2[2] = v_p3[2] - v_p2[2];

		CROSS(faceNormal, edge1, edge2);
		float dotP = DOT(v_p1, faceNormal);
		if (dotP < 0)
			frontFacing = false;

		unsigned int	binID=0;
		int		gxmin, gxmax, gymin, gymax, gzmin=0;

		//if (((blx1 >= 0 && blx1 < NUM_BLOCKS_X && bly1 >= 0 && bly1 < NUM_BLOCKS_Y) || (blx2 >= 0 && blx2 < NUM_BLOCKS_X && bly2 >= 0 && bly2 < NUM_BLOCKS_Y) || (blx3 >= 0 && blx3 < NUM_BLOCKS_X && bly3 >= 0 && bly3 < NUM_BLOCKS_Y)) && !frontFacing)
		if (1)
		{
			(blx1 < blx2) ? ((blx1 < blx3) ? gxmin = blx1 : gxmin = blx3) : ((blx2 < blx3) ? gxmin = blx2: gxmin = blx3);
			(bly1 < bly2) ? ((bly1 < bly3) ? gymin = bly1 : gymin = bly3) : ((bly2 < bly3) ? gymin = bly2: gymin = bly3);

			(blx1 > blx2) ? ((blx1 > blx3) ? gxmax = blx1 : gxmax = blx3) : ((blx2 > blx3) ? gxmax = blx2: gxmax = blx3);
			(bly1 > bly2) ? ((bly1 > bly3) ? gymax = bly1 : gymax = bly3) : ((bly2 > bly3) ? gymax = bly2: gymax = bly3);

			float zmin;
			(radius1 < radius2) ? ((radius1 < radius3) ? zmin = radius1 : zmin = radius3) : ((radius2 < radius3) ? zmin = radius2: zmin = radius3);

			(gxmin < 0) ? gxmin = 0 : gxmin=gxmin;
			(gymin < 0) ? gymin = 0 : gymin=gymin;

			(gxmin > NUM_BLOCKS_X - 1) ? gxmin = NUM_BLOCKS_X - 1 : gxmin = gxmin;
			(gymin > NUM_BLOCKS_Y - 1) ? gymin = NUM_BLOCKS_Y - 1 : gymin = gymin;

			(gxmax < 0) ? gxmax = 0 :gxmax=gxmax;
			(gymax < 0) ? gymax = 0 :gymax=gymax;

			(gxmax > NUM_BLOCKS_X - 1 ) ? gxmax = NUM_BLOCKS_X - 1 : gxmax = gxmax;
			(gymax > NUM_BLOCKS_Y - 1 ) ? gymax = NUM_BLOCKS_Y - 1 : gymax = gymax;

			int size_x;
			if (gxmax >= NUM_BLOCKS_X/2 && gxmin < NUM_BLOCKS_X/2)
				size_x = (gxmax - NUM_BLOCKS_X/2) + (NUM_BLOCKS_X/2 - gxmin) + 1;
			else
				size_x = gxmax - gxmin + 1;

			int size_y;
			if (gymax >= NUM_BLOCKS_Y/2 && gymin < NUM_BLOCKS_Y/2)
				size_y = (gymax - NUM_BLOCKS_Y/2) + (NUM_BLOCKS_Y/2 - gymin) + 1;
			else
				size_y = gymax - gymin + 1;
			//int size_x = gxmax - gxmin + 1;
			//int size_y = gymax - gymin + 1;

			sizeList[curface] = size_x * size_y;
			
			projCoordZ[curface] = zmin;
		}
		else
		{
			sizeList[curface] = 0;
			projCoordZ[curface] = -5.0f;
		}
	}

	__syncthreads();
}

__global__ void DS_spherical_Fillkernel(unsigned int *keyList, unsigned int *valueList, unsigned int *scanList, unsigned int *zSlabs, int *facelist, float *vertexlist, int *modelParams, float x_min, float y_min)
{
	int	tx = threadIdx.x;
	int	bx = blockIdx.x;

	int curface;
	curface = bx * NUMTHREADSDS + tx;

	if (curface < modelParams[0])
	{
		int offset = 0;
		int span = 0;

		if (curface != 0)
		{
			span = scanList[curface] - scanList[curface-1];
			offset = scanList[curface-1];
		}
		else
		{
			span = scanList[curface];
		}

		float v_p1[3], v_p2[3], v_p3[3];
		int face1, face2, face3;
		float tmp[4], point[4], tmp2[4];
		float radius1, radius2, radius3;

		face1 = 3*facelist[curface*3+0];
		face2 = 3*facelist[curface*3+1];
		face3 = 3*facelist[curface*3+2];

		// point 1
		point[0] = vertexlist[face1 + 0];
		point[1] = vertexlist[face1 + 1];
		point[2] = vertexlist[face1 + 2];
		point[3] = 1.0f;

		mulMatrixVector_D(tmp, &dd_camcoords[16], point);       // modelview matrix
		tmp2[0] = tmp[0] / tmp[3];
		tmp2[1] = tmp[1] / tmp[3];
		tmp2[2] = tmp[2] / tmp[3];
		tmp2[3] = 1.0f;

		v_p1[0] = tmp2[0];
		v_p1[1] = tmp2[1];
		v_p1[2] = tmp2[2];

		point[0] -= dd_camcoords[0];
		point[1] -= dd_camcoords[1];
		point[2] -= dd_camcoords[2];

		radius1 = getMagnitude(point);
		point[0] /= radius1;
		point[1] /= radius1;
		point[2] /= radius1;

		int blx1, bly1;

		blx1 = getEffective_x(point, x_min);
		bly1 = getEffective_y(point, y_min);

		// point 2
		point[0] = vertexlist[face2 + 0];
		point[1] = vertexlist[face2 + 1];
		point[2] = vertexlist[face2 + 2];
		point[3] = 1.0f;

		mulMatrixVector_D(tmp, &dd_camcoords[16], point);       // modelview matrix
		tmp2[0] = tmp[0] / tmp[3];
		tmp2[1] = tmp[1] / tmp[3];
		tmp2[2] = tmp[2] / tmp[3];
		tmp2[3] = 1.0f;

		v_p2[0] = tmp2[0];
		v_p2[1] = tmp2[1];
		v_p2[2] = tmp2[2];

		point[0] -= dd_camcoords[0];
		point[1] -= dd_camcoords[1];
		point[2] -= dd_camcoords[2];

		radius2 = getMagnitude(point);
		point[0] /= radius2;
		point[1] /= radius2;
		point[2] /= radius2;

		int blx2, bly2;

		blx2 = getEffective_x(point, x_min);
		bly2 = getEffective_y(point, y_min);

		// point 3
		point[0] = vertexlist[face3 + 0];
		point[1] = vertexlist[face3 + 1];
		point[2] = vertexlist[face3 + 2];
		point[3] = 1.0f;

		mulMatrixVector_D(tmp, &dd_camcoords[16], point);       // modelview matrix
		tmp2[0] = tmp[0] / tmp[3];
		tmp2[1] = tmp[1] / tmp[3];
		tmp2[2] = tmp[2] / tmp[3];
		tmp2[3] = 1.0f;

		v_p3[0] = tmp2[0];
		v_p3[1] = tmp2[1];
		v_p3[2] = tmp2[2];

		point[0] -= dd_camcoords[0];
		point[1] -= dd_camcoords[1];
		point[2] -= dd_camcoords[2];

		radius3 = getMagnitude(point);
		point[0] /= radius3;
		point[1] /= radius3;
		point[2] /= radius3;

		int blx3, bly3;

		blx3 = getEffective_x(point, x_min);
		bly3 = getEffective_y(point, y_min);

		// check if the triangle is front facing
		bool frontFacing = true;
		float edge1[3], edge2[3], faceNormal[3];
		edge1[0] = v_p3[0] - v_p1[0];
		edge1[1] = v_p3[1] - v_p1[1];
		edge1[2] = v_p3[2] - v_p1[2];

		edge2[0] = v_p3[0] - v_p2[0];
		edge2[1] = v_p3[1] - v_p2[1];
		edge2[2] = v_p3[2] - v_p2[2];

		CROSS(faceNormal, edge1, edge2);
		float dotP = DOT(v_p1, faceNormal);
		if (dotP < 0)
			frontFacing = false;

		unsigned int	binID=0;
		int		gxmin, gxmax, gymin, gymax, gzmin=0;

		//if (((blx1 >= 0 && blx1 < NUM_BLOCKS_X && bly1 >= 0 && bly1 < NUM_BLOCKS_Y) || (blx2 >= 0 && blx2 < NUM_BLOCKS_X && bly2 >= 0 && bly2 < NUM_BLOCKS_Y) || (blx3 >= 0 && blx3 < NUM_BLOCKS_X && bly3 >= 0 && bly3 < NUM_BLOCKS_Y)) && !frontFacing)
		if (1)
		{
			(blx1 < blx2) ? ((blx1 < blx3) ? gxmin = blx1 : gxmin = blx3) : ((blx2 < blx3) ? gxmin = blx2: gxmin = blx3);
			(bly1 < bly2) ? ((bly1 < bly3) ? gymin = bly1 : gymin = bly3) : ((bly2 < bly3) ? gymin = bly2: gymin = bly3);

			(blx1 > blx2) ? ((blx1 > blx3) ? gxmax = blx1 : gxmax = blx3) : ((blx2 > blx3) ? gxmax = blx2: gxmax = blx3);
			(bly1 > bly2) ? ((bly1 > bly3) ? gymax = bly1 : gymax = bly3) : ((bly2 > bly3) ? gymax = bly2: gymax = bly3);

			gzmin = zSlabs[curface];

			(gxmin < 0) ? gxmin = 0 : gxmin=gxmin;
			(gymin < 0) ? gymin = 0 : gymin=gymin;
			(gzmin < 0) ? gzmin = 0 : gzmin=gzmin;

			(gxmin > NUM_BLOCKS_X - 1 ) ? gxmin = NUM_BLOCKS_X - 1 : gxmin = gxmin;
			(gymin > NUM_BLOCKS_Y - 1 ) ? gymin = NUM_BLOCKS_Y - 1 : gymin = gymin;
			(gzmin > NUM_SLABS - 1 ) ? gzmin = NUM_SLABS - 1: gzmin = gzmin;

			(gxmax < 0) ? gxmax = 0 :gxmax=gxmax;
			(gymax < 0) ? gymax = 0 :gymax=gymax;

			(gxmax > NUM_BLOCKS_X - 1 ) ? gxmax = NUM_BLOCKS_X - 1 : gxmax = gxmax;
			(gymax > NUM_BLOCKS_Y - 1 ) ? gymax = NUM_BLOCKS_Y - 1 : gymax = gymax;

			int size_x;
			if (gxmax >= NUM_BLOCKS_X/2 && gxmin < NUM_BLOCKS_X/2)
				size_x = (gxmax - NUM_BLOCKS_X/2) + (NUM_BLOCKS_X/2 - gxmin) + 1;
			else
				size_x = gxmax - gxmin + 1;

			int size_y;
			if (gymax >= NUM_BLOCKS_Y/2 && gymin < NUM_BLOCKS_Y/2)
				size_y = (gymax - NUM_BLOCKS_Y/2) + (NUM_BLOCKS_Y/2 - gymin) + 1;
			else
				size_y = gymax - gymin + 1;
			
			//int size_x = gxmax - gxmin + 1;
			//int size_y = gymax - gymin + 1;

			for (unsigned int i=0; i<size_x; i++)
			{
				for (unsigned int j=0; j<size_y; j++)
				{
					keyList[offset + i*size_y + j] = ((gxmin + i) * NUM_BLOCKS_Y + (gymin + j)) * NUM_SLABS + gzmin;
					valueList[offset + i*size_y + j] = curface;
				}
			}
		}
	}

	__syncthreads();
}

#endif
