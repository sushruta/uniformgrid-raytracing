#ifndef SHADER_KERNEL_CU
#define SHADER_KERNEL_CU

__device__ float InterPolation(float a, float b, float c)
{
	return a+(b-a)*c*c*(3-2*c);
}

__device__ float InterLinear(float a, float b, float c)
{
	return a*(1-c)+b*c;
}

__device__ float Noise(int x)
{
	x = (x<<13)^x;
	return (((x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483648.0);
}

__device__ float PerlinNoise(float x, float y, int width, int octaves, int seed, float periode)
{
	float a, b, value, freq;
	float zone_x, zone_y;
	int box, num, step_x, step_y;
	int amplitude = 324;
	int noisedata;

	freq = 1.0f / (float)(periode);

	for (int s=0; s<octaves; s++)
	{
		num = (int) (width * freq);
		step_x = (int) (x * freq);
		step_y = (int) (y * freq);
		zone_x = x * freq - step_x;
		zone_y = y * freq - step_y;
		box = step_x + step_y * num;
		noisedata = (box + seed);
		a = InterPolation(Noise(noisedata), Noise(noisedata+1), zone_x);
		b = InterPolation(Noise(noisedata+num), Noise(noisedata+1+num), zone_x);
		value = InterPolation(a, b, zone_y) * amplitude;
	}
	return value;
}

__device__ void lambert_color_pixel(float *point, float *normal, float *color, float *dir, float *material)
{
	float light_dir[3];
	float light_ambient[3] = {0.5f, 0.5f, 0.5f};
	float light_diffuse[3] = {1.0f, 1.0f, 1.0f};

	int lightID = 0;
	float light_position_view[3];
	light_position_view[0] = dd_camcoords[16+0]*dd_light_position[lightID*3 + 0] + dd_camcoords[16+4]*dd_light_position[lightID*3 + 1] + dd_camcoords[16+8]*dd_light_position[lightID*3 + 2];
	light_position_view[1] = dd_camcoords[16+1]*dd_light_position[lightID*3 + 0] + dd_camcoords[16+5]*dd_light_position[lightID*3 + 1] + dd_camcoords[16+9]*dd_light_position[lightID*3 + 2];
	light_position_view[2] = dd_camcoords[16+2]*dd_light_position[lightID*3 + 0] + dd_camcoords[16+6]*dd_light_position[lightID*3 + 1] + dd_camcoords[16+10]*dd_light_position[lightID*3 + 2];

	float point_view[3];
	point_view[0] = dd_camcoords[16+0] * point[0] + dd_camcoords[16+4] * point[1] + dd_camcoords[16+8] * point[2];
	point_view[1] = dd_camcoords[16+1] * point[0] + dd_camcoords[16+5] * point[1] + dd_camcoords[16+9] * point[2];
	point_view[2] = dd_camcoords[16+2] * point[0] + dd_camcoords[16+6] * point[1] + dd_camcoords[16+10] * point[2];

	float normal_view[3];
	normal_view[0] = dd_camcoords[16+0] * normal[0] + dd_camcoords[16+4] * normal[1] + dd_camcoords[16+8] * normal[2];
	normal_view[1] = dd_camcoords[16+1] * normal[0] + dd_camcoords[16+5] * normal[1] + dd_camcoords[16+9] * normal[2];
	normal_view[2] = dd_camcoords[16+2] * normal[0] + dd_camcoords[16+6] * normal[1] + dd_camcoords[16+10] * normal[2];
	NORMALIZE(normal_view);

	light_dir[0] = point_view[0] - light_position_view[0];
	light_dir[1] = point_view[1] - light_position_view[1];
	light_dir[2] = point_view[2] - light_position_view[2];
	NORMALIZE(light_dir);

	color[0] += material[0] * light_ambient[0];
	color[1] += material[1] * light_ambient[1];
	color[2] += material[2] * light_ambient[2];

	float dot_diffuse = DOT(light_dir, normal_view);
	(dot_diffuse > 0) ? dot_diffuse *= 1 : dot_diffuse *= -1;
	if (dot_diffuse > 0)
	{
		color[0] += material[3] * light_diffuse[0] * dot_diffuse;
		color[1] += material[4] * light_diffuse[1] * dot_diffuse;
		color[2] += material[5] * light_diffuse[2] * dot_diffuse;
	}
}

__device__ void lambert_color_drop_off_pixel(float *point, float *normal, float *color, float *dir, float *material, float drop_off)
{
	float light_dir[3];
	float light_ambient[3] = {0.5f, 0.5f, 0.5f};
	float light_diffuse[3] = {1.0f, 1.0f, 1.0f};

	int lightID = 0;
	float light_position_view[3];
	light_position_view[0] = dd_camcoords[16+0]*dd_light_position[lightID*3 + 0] + dd_camcoords[16+4]*dd_light_position[lightID*3 + 1] + dd_camcoords[16+8]*dd_light_position[lightID*3 + 2];
	light_position_view[1] = dd_camcoords[16+1]*dd_light_position[lightID*3 + 0] + dd_camcoords[16+5]*dd_light_position[lightID*3 + 1] + dd_camcoords[16+9]*dd_light_position[lightID*3 + 2];
	light_position_view[2] = dd_camcoords[16+2]*dd_light_position[lightID*3 + 0] + dd_camcoords[16+6]*dd_light_position[lightID*3 + 1] + dd_camcoords[16+10]*dd_light_position[lightID*3 + 2];

	float point_view[3];
	point_view[0] = dd_camcoords[16+0] * point[0] + dd_camcoords[16+4] * point[1] + dd_camcoords[16+8] * point[2];
	point_view[1] = dd_camcoords[16+1] * point[0] + dd_camcoords[16+5] * point[1] + dd_camcoords[16+9] * point[2];
	point_view[2] = dd_camcoords[16+2] * point[0] + dd_camcoords[16+6] * point[1] + dd_camcoords[16+10] * point[2];

	float normal_view[3];
	normal_view[0] = dd_camcoords[16+0] * normal[0] + dd_camcoords[16+4] * normal[1] + dd_camcoords[16+8] * normal[2];
	normal_view[1] = dd_camcoords[16+1] * normal[0] + dd_camcoords[16+5] * normal[1] + dd_camcoords[16+9] * normal[2];
	normal_view[2] = dd_camcoords[16+2] * normal[0] + dd_camcoords[16+6] * normal[1] + dd_camcoords[16+10] * normal[2];
	NORMALIZE(normal_view);

	light_dir[0] = point_view[0] - light_position_view[0];
	light_dir[1] = point_view[1] - light_position_view[1];
	light_dir[2] = point_view[2] - light_position_view[2];
	NORMALIZE(light_dir);

	color[0] += material[0] * light_ambient[0] * drop_off;
	color[1] += material[1] * light_ambient[1] * drop_off;
	color[2] += material[2] * light_ambient[2] * drop_off;

	float dot_diffuse = DOT(light_dir, normal_view);
	(dot_diffuse > 0) ? dot_diffuse *= 1 : dot_diffuse *= -1;
	if (dot_diffuse > 0)
	{
		color[0] += material[3] * light_diffuse[0] * dot_diffuse * drop_off;
		color[1] += material[4] * light_diffuse[1] * dot_diffuse * drop_off;
		color[2] += material[5] * light_diffuse[2] * dot_diffuse * drop_off;
	}
}

__device__ float get_material(int idx)
{
	int seed = 37;
	int width = 12413;
	float scale = 1.0f;

	float x, y;
	//float t_value = dd_t_value[pixelID];
	x = blockIdx.x * NUM_THREADS_X + threadIdx.x; // d_cam_pos[0] + t_value * dd_dir[pixelID*3+0];
	y = blockIdx.y * NUM_THREADS_Y + threadIdx.y; // d_cam_pos[1] + t_value * dd_dir[pixelID*3+1];

	float value_1, value_2, value_3, value_4, value_5, value_6;
	value_1 = PerlinNoise(x*scale,y*scale,width,1,seed,25);
	value_2 = PerlinNoise(x*scale,y*scale,width,1,seed,12.5);
	value_3 = PerlinNoise(x*scale,y*scale,width,1,seed,6.25);
	value_4 = PerlinNoise(x*scale,y*scale,width,1,seed,3.125);
	value_5 = PerlinNoise(x*scale,y*scale,width,1,seed,1.56);
	value_6 = PerlinNoise(x*scale,y*scale,width,1,seed,0.78);

	float tmp = (int) value_1 + (int) (value_2 * 0.25) + (int)(value_3 * 0.125) + (int)(value_4 * 0.0625) + (int) (value_5 * 0.03125) + (int) (value_6 * 0.0156);

	float res;
	if (idx == 0)
		res = InterLinear(tmp,0,0);
	else if (idx == 1)
		res = InterLinear(0,tmp, tmp);
	else
		res = InterLinear(tmp,0,tmp);

	(res > 255.0) ? res = 255.0 : res = res;
	res /= 255.0;

	return res;
}

__global__ void lambertian_shade(unsigned char *d_img, float *dd_normal, float *dd_t_value, float *dd_dir, int *dd_intersect_id, float *d_cam_pos, int *mat_idx, float *mat_list, int mat_count)
{
	int pixelID = (blockIdx.y*NUM_BLOCKS_X*NUM_THREADS_X*NUM_THREADS_Y + blockIdx.x*NUM_THREADS_X + threadIdx.y*SCREEN_WIDTH + threadIdx.x);
	float material[6];
	int tri_intersected = dd_intersect_id[pixelID];
	int idx = mat_idx[tri_intersected];
	float color[3];
	color[0] = 0.0f;
	color[1] = 0.0f;
	color[2] = 0.0f;

	dd_intersect_id[pixelID] = idx;

	if (idx >= 0 && idx < mat_count)
	{
		material[0] = mat_list[idx*MATERIAL_SIZE + 3];
		material[1] =mat_list[idx*MATERIAL_SIZE + 4];
		material[2] =mat_list[idx*MATERIAL_SIZE + 5];

		material[3] =mat_list[idx*MATERIAL_SIZE + 3];//get_material(0); //0.8f;//mat_list[idx*MATERIAL_SIZE + 3];
		material[4] =mat_list[idx*MATERIAL_SIZE + 4];//get_material(1); //0.8f;//mat_list[idx*MATERIAL_SIZE + 4];
		material[5] =mat_list[idx*MATERIAL_SIZE + 5];//get_material(2); //0.8f;//mat_list[idx*MATERIAL_SIZE + 5];
	}
	else if (idx == 0)
	{
		material[0] = mat_list[idx*MATERIAL_SIZE + 0];
		material[1] = mat_list[idx*MATERIAL_SIZE + 1];
		material[2] = mat_list[idx*MATERIAL_SIZE + 2];

		material[3] = mat_list[idx*MATERIAL_SIZE + 3];
		material[4] = mat_list[idx*MATERIAL_SIZE + 4];
		material[5] = mat_list[idx*MATERIAL_SIZE + 5];
	}

	if (idx >= 0 && idx < mat_count)
	{
		float point[3];

		float t_value = dd_t_value[pixelID];
		if (t_value > 0)
		{
		point[0] = d_cam_pos[0] + t_value * dd_dir[pixelID*3+0];
		point[1] = d_cam_pos[1] + t_value * dd_dir[pixelID*3+1];
		point[2] = d_cam_pos[2] + t_value * dd_dir[pixelID*3+2];

		lambert_color_pixel(point, &dd_normal[pixelID*3+0], color, &dd_dir[pixelID*3+0], material);

		color[0] > 1.0f ? color[0] = 1.0f: color[0] = color[0];
		color[1] > 1.0f ? color[1] = 1.0f: color[1] = color[1];
		color[2] > 1.0f ? color[2] = 1.0f: color[2] = color[2];
		}
	}
	
	d_img[pixelID*3+0] = color[0] * 255;
	d_img[pixelID*3+1] = color[1] * 255;
	d_img[pixelID*3+2] = color[2] * 255;
}

__device__ float get_along_x(float *vec)
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

	if (rightDotValue > 0)
		angle = angle;
	else
		angle = -1.0 *  angle;

	return angle;
}

__device__ float get_along_y(float *vec)
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

	if (upDotValue > 0)
		angle = angle;
	else
		angle = -1.0 * angle;

	return angle;
}

__global__ void spot_shade(unsigned char *d_img, float *dd_normal, float *dd_t_value, float *dd_dir, int *dd_intersect_id, float *d_cam_pos, int *mat_idx, float *mat_list, int mat_count, float *dump)
{
	int pixelID = (blockIdx.y*NUM_BLOCKS_X*NUM_THREADS_X*NUM_THREADS_Y + blockIdx.x*NUM_THREADS_X + threadIdx.y*SCREEN_WIDTH + threadIdx.x);

	float ptIntersection[3];
	float tVal = dd_t_value[pixelID];
	ptIntersection[0] = d_cam_pos[0] + tVal * dd_dir[pixelID*3 + 0];// cameraRayDirection[0];
	ptIntersection[1] = d_cam_pos[1] + tVal * dd_dir[pixelID*3 + 1];// cameraRayDirection[1];
	ptIntersection[2] = d_cam_pos[2] + tVal * dd_dir[pixelID*3 + 2];// cameraRayDirection[2];

	// now construct the ray from
	// the light to the point
	// of intersection
	float lightRayDirection[3];
	lightRayDirection[0] = ptIntersection[0] - dd_camcoords[0];
	lightRayDirection[1] = ptIntersection[1] - dd_camcoords[1];
	lightRayDirection[2] = ptIntersection[2] - dd_camcoords[2];
	NORMALIZE(lightRayDirection);

	float drop_off;
	float x = get_along_x(lightRayDirection);
	float y = get_along_y(lightRayDirection);
	dump[pixelID*2 + 0] = x;
	dump[pixelID*2 + 1] = y;

	if (x < M_PI/4 && x > -M_PI/4 && y < M_PI/4 && y > -M_PI/4)
		drop_off = 1.0f;
	else
		drop_off = 0.25f;

	float material[6];
	int tri_intersected = dd_intersect_id[pixelID];
	int idx = mat_idx[tri_intersected];
	float color[3];
	color[0] = 0.0f;
	color[1] = 0.0f;
	color[2] = 0.0f;

	dd_intersect_id[pixelID] = idx;

	if (idx >= 0 && idx < mat_count)
	{
		material[0] = mat_list[idx*MATERIAL_SIZE + 3];
		material[1] = mat_list[idx*MATERIAL_SIZE + 4];
		material[2] = mat_list[idx*MATERIAL_SIZE + 5];

		material[3] = mat_list[idx*MATERIAL_SIZE + 3];//get_material(0); //0.8f;//mat_list[idx*MATERIAL_SIZE + 3];
		material[4] = mat_list[idx*MATERIAL_SIZE + 4];//get_material(1); //0.8f;//mat_list[idx*MATERIAL_SIZE + 4];
		material[5] = mat_list[idx*MATERIAL_SIZE + 5];//get_material(2); //0.8f;//mat_list[idx*MATERIAL_SIZE + 5];
	}

	if (idx >= 0 && idx < mat_count)
	{
		float point[3];

		float t_value = dd_t_value[pixelID];
		point[0] = d_cam_pos[0] + t_value * dd_dir[pixelID*3+0];
		point[1] = d_cam_pos[1] + t_value * dd_dir[pixelID*3+1];
		point[2] = d_cam_pos[2] + t_value * dd_dir[pixelID*3+2];

		lambert_color_drop_off_pixel(point, &dd_normal[pixelID*3+0], color, &dd_dir[pixelID*3+0], material, drop_off);

		color[0] > 1.0f ? color[0] = 1.0f: color[0] = color[0];
		color[1] > 1.0f ? color[1] = 1.0f: color[1] = color[1];
		color[2] > 1.0f ? color[2] = 1.0f: color[2] = color[2];
	}
	
	d_img[pixelID*3+0] = color[0] * 255;
	d_img[pixelID*3+1] = color[1] * 255;
	d_img[pixelID*3+2] = color[2] * 255;
}

__global__ void shadow_kernel(unsigned char *d_img, int *is_shadowed)
{
	int pixelID = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);

	if (is_shadowed[pixelID] == 1)
	{
		d_img[pixelID*3 + 0] /= 3;
		d_img[pixelID*3 + 1] /= 3;
		d_img[pixelID*3 + 2] /= 3;
	}

	return;
}

/******************************/
/*** PERLIN NOISE FUNCTIONS ***/
/******************************/

/*// function to find out the gradient corresponding to the coordinates
__device__ int index(int i, int j, int k, int l, volatile int *permute)
{
	        return (permute[(l + permute[(k + permute[(j + permute[i & 0xff]) & 0xff]) & 0xff]) & 0xff] & 0x1f);
}

// function to compute dor product
__device__ float Prod(float a, float b)
{
	        if (b > 0)
			                return a;
		        if (b < 0)
				                return -a;

			        return 0;
}

__device__ float Dot_Prod(float x1, char x2, float y1, char y2, float z1, char z2, float t1, char t2)
{
	        return (Prod(x1, x2) + Prod(y1, y2) + Prod(z1, z2) + Prod(t1, t2));
}

__device__ float LinearInterpolation(float start, float end, float state)
{
	        return start + state * (end - start);
}

__device__ float Spline5(float state)
{
	        // 3x^2 + 2x^3 is not as good as 6x^5 - 15x^4 + 10x^3
	        float square = state * state;
		        float cubic = square * state;

			        return cubic * (6 * square - 15 * state + 10);
}

//
// Noise function, returning the Perlin Noise at a given point
//
__device__ float Noise(float x, float y, float z, float t)
{
	// The unit hypercube containing the point
	int x1 = (int) (x > 0 ? x : x - 1);
	int y1 = (int) (y > 0 ? y : y - 1);
	int z1 = (int) (z > 0 ? z : z - 1);
	int t1 = (int) (t > 0 ? t : t - 1);
	int x2 = x1 + 1;
	int y2 = y1 + 1;
	int z2 = z1 + 1;
	int t2 = t1 + 1;

	// The 16 corresponding gradients
	char * g0000 = gradient[Indice (x1, y1, z1, t1)];
	char * g0001 = gradient[Indice (x1, y1, z1, t2)];
	char * g0010 = gradient[Indice (x1, y1, z2, t1)];
	char * g0011 = gradient[Indice (x1, y1, z2, t2)];
	char * g0100 = gradient[Indice (x1, y2, z1, t1)];
	char * g0101 = gradient[Indice (x1, y2, z1, t2)];
	char * g0110 = gradient[Indice (x1, y2, z2, t1)];
	char * g0111 = gradient[Indice (x1, y2, z2, t2)];
	char * g1000 = gradient[Indice (x2, y1, z1, t1)];
	char * g1001 = gradient[Indice (x2, y1, z1, t2)];
	char * g1010 = gradient[Indice (x2, y1, z2, t1)];
	char * g1011 = gradient[Indice (x2, y1, z2, t2)];
	char * g1100 = gradient[Indice (x2, y2, z1, t1)];
	char * g1101 = gradient[Indice (x2, y2, z1, t2)];
	char * g1110 = gradient[Indice (x2, y2, z2, t1)];
	char * g1111 = gradient[Indice (x2, y2, z2, t2)];

	// The 16 vectors
	float dx1 = x - x1;
	float dx2 = x - x2;
	float dy1 = y - y1;
	float dy2 = y - y2;
	float dz1 = z - z1;
	float dz2 = z - z2;
	float dt1 = t - t1;
	float dt2 = t - t2;

	// The 16 dot products
	float b0000 = Dot_prod(dx1, g0000[0], dy1, g0000[1],
			dz1, g0000[2], dt1, g0000[3]);
	float b0001 = Dot_prod(dx1, g0001[0], dy1, g0001[1],
			dz1, g0001[2], dt2, g0001[3]);
	float b0010 = Dot_prod(dx1, g0010[0], dy1, g0010[1],
			dz2, g0010[2], dt1, g0010[3]);
	float b0011 = Dot_prod(dx1, g0011[0], dy1, g0011[1],
			dz2, g0011[2], dt2, g0011[3]);
	float b0100 = Dot_prod(dx1, g0100[0], dy2, g0100[1],
			dz1, g0100[2], dt1, g0100[3]);
	float b0101 = Dot_prod(dx1, g0101[0], dy2, g0101[1],
			dz1, g0101[2], dt2, g0101[3]);
	float b0110 = Dot_prod(dx1, g0110[0], dy2, g0110[1],
			dz2, g0110[2], dt1, g0110[3]);
	float b0111 = Dot_prod(dx1, g0111[0], dy2, g0111[1],
			dz2, g0111[2], dt2, g0111[3]);
	float b1000 = Dot_prod(dx2, g1000[0], dy1, g1000[1],
			dz1, g1000[2], dt1, g1000[3]);
	float b1001 = Dot_prod(dx2, g1001[0], dy1, g1001[1],
			dz1, g1001[2], dt2, g1001[3]);
	float b1010 = Dot_prod(dx2, g1010[0], dy1, g1010[1],
			dz2, g1010[2], dt1, g1010[3]);
	float b1011 = Dot_prod(dx2, g1011[0], dy1, g1011[1],
			dz2, g1011[2], dt2, g1011[3]);
	float b1100 = Dot_prod(dx2, g1100[0], dy2, g1100[1],
			dz1, g1100[2], dt1, g1100[3]);
	float b1101 = Dot_prod(dx2, g1101[0], dy2, g1101[1],
			dz1, g1101[2], dt2, g1101[3]);
	float b1110 = Dot_prod(dx2, g1110[0], dy2, g1110[1],
			dz2, g1110[2], dt1, g1110[3]);
	float b1111 = Dot_prod(dx2, g1111[0], dy2, g1111[1],
			dz2, g1111[2], dt2, g1111[3]);
	// Then the interpolations, down to the result
	dx1 = Spline5 (dx1);
	dy1 = Spline5 (dy1);
	dz1 = Spline5 (dz1);
	dt1 = Spline5 (dt1);

	float b111 = LinearInterpolation (b1110, b1111, dt1);
	float b110 = LinearInterpolation (b1100, b1101, dt1);
	float b101 = LinearInterpolation (b1010, b1011, dt1);
	float b100 = LinearInterpolation (b1000, b1001, dt1);
	float b011 = LinearInterpolation (b0110, b0111, dt1);
	float b010 = LinearInterpolation (b0100, b0101, dt1);
	float b001 = LinearInterpolation (b0010, b0011, dt1);
	float b000 = LinearInterpolation (b0000, b0001, dt1);

	float b11 = LinearInterpolation (b110, b111, dz1);
	float b10 = LinearInterpolation (b100, b101, dz1);
	float b01 = LinearInterpolation (b010, b011, dz1);
	float b00 = LinearInterpolation (b000, b001, dz1);

	float b1 = LinearInterpolation (b10, b11, dy1);
	float b0 = LinearInterpolation (b00, b01, dy1);

	return LinearInterpolation (b0, b1, dx1);
}*/



__global__ void perlin_noise_shade(unsigned char *d_img, float *dd_t_value, float *dd_dir, float *d_cam_pos, int *dd_intersect_id)
{
	//extern __shared__ int sharedmem;
	int pixelID = (blockIdx.y * NUM_BLOCKS_X * NUM_THREADS_X * NUM_THREADS_Y + blockIdx.x * NUM_THREADS_X + threadIdx.y * SCREEN_WIDTH + threadIdx.x);

	int seed = 63;
	int width = 12413;
	float scale = 1.0f;

	float x, y;
	//float t_value = dd_t_value[pixelID];
	x = blockIdx.x * NUM_THREADS_X + threadIdx.x; // d_cam_pos[0] + t_value * dd_dir[pixelID*3+0];
	y = blockIdx.y * NUM_THREADS_Y + threadIdx.y; // d_cam_pos[1] + t_value * dd_dir[pixelID*3+1];

	float value_1, value_2, value_3, value_4, value_5, value_6;
	value_1 = PerlinNoise(x*scale,y*scale,width,1,seed,100);
	value_2 = PerlinNoise(x*scale,y*scale,width,1,seed,25);
	value_3 = PerlinNoise(x*scale,y*scale,width,1,seed,12.5);
	value_4 = PerlinNoise(x*scale,y*scale,width,1,seed,6.25);
	value_5 = PerlinNoise(x*scale,y*scale,width,1,seed,3.125);
	value_6 = PerlinNoise(x*scale,y*scale,width,1,seed,1.56);

	float tmp = (int) value_1 + (int) (value_2 * 0.25) + (int)(value_3 * 0.125) + (int)(value_4 * 0.0625) + (int) (value_5 * 0.03125) + (int) (value_6 * 0.0156);
	int r = InterLinear(tmp,0,0);
	int g = InterLinear(0,tmp,0);
	int b = InterLinear(0,0,tmp);
	(r > 255) ? r = 255 : r = r;
	(g > 255) ? g = 255 : g = g;
	(b > 255) ? b = 255 : b = b;

	if (dd_intersect_id[pixelID] >= 0)
	{
		d_img[pixelID*3 + 0] = (unsigned char) r;
		d_img[pixelID*3 + 1] = (unsigned char) g;
		d_img[pixelID*3 + 2] = (unsigned char) b;
	}
	else
	{
		d_img[pixelID*3 + 0] = 0;
		d_img[pixelID*3 + 1] = 0;
		d_img[pixelID*3 + 2] = 0;
	}
}

#endif
