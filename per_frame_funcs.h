#include "misc_kernel.cu"
#include "light_kernel.cu"

void setDirectionTexture();

void updateLightPosition()
{ 
	h_light_position[0] = 10.0f;//0.55 * cosf(lightRotFactor);//1.5;// * cosf(lightRotFactor);//0.55 * cosf(lightRotFactor);//(model->xMin + model->xMax) / 2.0f;		//0 + 0.4 * cosf(lightRotFactor);
	h_light_position[1] = 12.0f;//(model->yMin + model->yMax) / 2.0f;//1.5;//(model->yMin + model->yMax) / 2.0f;
	h_light_position[2] = 6;//0.55 * sinf(lightRotFactor);//1.5;// * sinf(lightRotFactor);//0.55 * sinf(lightRotFactor);//(model->zMin + model->zMax) / 2.0f - 2.0f;	//0 + 0.4 * sinf(lightRotFactor);

	printf("in world,light:(%f,%f,%f) and rotation factor is %f\n",h_light_position[0],h_light_position[1],h_light_position[2], lightRotFactor);

	cutilSafeCall(cudaMemcpyToSymbol(dd_light_position, h_light_position, sizeof(float)*6));
	lightRotFactor += 0.05;
}

void fillCoordinatesData()
{
	camcoords[0] = camera->worldori[0];
	camcoords[1] = camera->worldori[1];
	camcoords[2] = camera->worldori[2];
	camcoords[3] = camera->worldori[3];

	for (int i=0; i<4; i++)
	{
		camcoords[i*3+0+4] = camera->frustumcorner[i][0];
		camcoords[i*3+1+4] = camera->frustumcorner[i][1];
		camcoords[i*3+2+4] = camera->frustumcorner[i][2];
	}

	for (int i=0; i<16; i++)
		camcoords[16+i] = camera->modelview_matrix[i];
	for (int i=0; i<16; i++)
		camcoords[32+i] = camera->projection_matrix[i];
	for (int i=0; i<16; i++)
		camcoords[48+i] = camera->mvp_matrix[i];

	cutilSafeCall(cudaMemcpyToSymbol(dd_camcoords, camcoords, sizeof(float) * 16 * 4));
	setDirectionTexture();

	return;
}

void build_frustum_grid()
{
	unsigned ds_build_time = 0;
	CUT_SAFE_CALL(cutCreateTimer(&ds_build_time));
	CUT_SAFE_CALL(cutStartTimer(ds_build_time));

	fGrid->buildGrid(model->d_facelist, model->d_vertexlist);

	CUT_SAFE_CALL(cutStopTimer(ds_build_time));
	printf("time taken to build DS : %f (ms)\n", cutGetTimerValue(ds_build_time));
}

void build_secondary_frustum_grid(float xM, float yM)
{
	unsigned ds_build_time = 0;
	CUT_SAFE_CALL(cutCreateTimer(&ds_build_time));
	CUT_SAFE_CALL(cutStartTimer(ds_build_time));

	fGrid->buildSphericalGrid(model->d_facelist, model->d_vertexlist, xM, yM);

	CUT_SAFE_CALL(cutStopTimer(ds_build_time));
	printf("time taken to build DS : %f (ms)\n", cutGetTimerValue(ds_build_time));
}

// pre process the data
// rearrange the data
// and stuff
void getRayGridMapping(float *t_val_list, float *ray_dir_list, float *x_list, float *y_list, float *angle, float *cam_pt)
{
	unsigned int timer = 0;
	printf("shooting rays to determine the grid mapping!\n");

	dim3 mapSortgrid(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 mapSortthreads(NUM_THREADS_X, NUM_THREADS_Y, 1);

	CUT_SAFE_CALL( cutCreateTimer(&timer));
	CUT_SAFE_CALL( cutStartTimer(timer));
	cudaThreadSynchronize();
	//mapSortkernel<<< mapSortgrid, mapSortthreads, 0 >>>(t_val_list, ray_dir_list, map, cam_pt, near);
	map_spherical_Sortkernel<<< mapSortgrid, mapSortthreads, 0 >>>(t_val_list, ray_dir_list, x_list, y_list, angle, cam_pt);
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("w.r.t to light, shooting and getting map Time : %f (msec)\n", cutGetTimerValue(timer));

	/*unsigned int *h_map = (unsigned int *) malloc(sizeof(unsigned int) * SCREEN_WIDTH * SCREEN_HEIGHT * 2);
	cutilSafeCall(cudaMemcpy(h_map, d_map, sizeof(unsigned int) * IMAGE_SIZE * 2, cudaMemcpyDeviceToHost));
	for (int i=0; i<IMAGE_SIZE; i++)
		printf("map value : %d\n", h_map[i]);*/

	return;
}

void getEffectiveRayGridMapping(float *t_val_list, float *ray_dir_list, unsigned int *map, float *cam_pt, float xM, float yM)
{
	unsigned int timer = 0;
	printf("shooting rays to determine the effective grid mapping!\n");

	dim3 mapSortgrid(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 mapSortthreads(NUM_THREADS_X, NUM_THREADS_Y, 1);

	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));
	mapSort_Effective_kernel<<<mapSortgrid, mapSortthreads, 0>>>(t_val_list, ray_dir_list, map, cam_pt, xM, yM);
	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer( timer));
	printf("w.r.t to light, shooting and getting effective map Time : %f (msec)\n", cutGetTimerValue(timer));

	return;
}

void processData()
{
	// do the sorting thing first!
	dData->sort_by_block_indices();

	// get pseudo-valid array
	dData->get_pseudo_valid_array();

	// segment the pseudo-valid array
	dData->segment_pseudo_valid_array();

	// get the corresponding cuda block array
	dData->get_cuda_block_array();

	// copy threadId to corresponding location
	dData->copy_thread_id();

	// run stream compaction
	dData->stream_compact();

	return;
}

void check_for_shadows(int light_id)
{
	unsigned int timer = 0;
	int sharedmem = (sizeof(int) * 7);
	sharedmem += sizeof(float) * MAX_TRIANGLES * 9;
	dim3 light_rcgrid(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
	dim3 light_rcthreads(NUM_THREADS_X, NUM_THREADS_Y, 1);

	CUT_SAFE_CALL( cutCreateTimer(&timer));
	CUT_SAFE_CALL( cutStartTimer(timer));
	cudaThreadSynchronize();
	mod_light_rckernel<<< light_rcgrid, light_rcthreads, sharedmem >>>(fGrid->d_triangle_value_list, model->d_vertexlist, model->d_facelist, fGrid->d_span, fGrid->d_offset, dData->d_primary_ray_t_value, dData->d_primary_ray_direction, dData->d_is_shadowed, dData->d_map, dData->d_prefixMap, camera->d_cam_position, (int) *(dData->h_numCudaBlocks));
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(timer));
	printf("w.r.t to light, shooting and getting shadow map Time : %f (msec)\n", cutGetTimerValue(timer));

	// it might be a good idea to let the light revolve in 
	// 360 degrees and note down the timings and plot it

	return;
}

void	setDirectionTexture()
{
	// Update the Direction Texture
	int cnt = 0;
	float tempvec1[4], tempvec2[4];
	
	// Row 1
	//1
	dirtex[cnt++] = camcoords[4];
	dirtex[cnt++] = camcoords[5];
	dirtex[cnt++] = camcoords[6];
	dirtex[cnt++] = 0;

	//2	
	dirtex[cnt++] = camcoords[4] + 0.25 * ( camcoords[7] - camcoords[4] );
	dirtex[cnt++] = camcoords[5] + 0.25 * ( camcoords[8] - camcoords[5] );
	dirtex[cnt++] = camcoords[6] + 0.25 * ( camcoords[9] - camcoords[6] );
	dirtex[cnt++] = 0;

	//3	
	dirtex[cnt++] = camcoords[4] + 0.5 * ( camcoords[7] - camcoords[4] );
	dirtex[cnt++] = camcoords[5] + 0.5 * ( camcoords[8] - camcoords[5] );
	dirtex[cnt++] = camcoords[6] + 0.5 * ( camcoords[9] - camcoords[6] );
	dirtex[cnt++] = 0;

	//4	
	dirtex[cnt++] = camcoords[4] + 0.75 * ( camcoords[7] - camcoords[4] );
	dirtex[cnt++] = camcoords[5] + 0.75 * ( camcoords[8] - camcoords[5] );
	dirtex[cnt++] = camcoords[6] + 0.75 * ( camcoords[9] - camcoords[6] );
	dirtex[cnt++] = 0;

	//5
	dirtex[cnt++] = camcoords[7];
	dirtex[cnt++] = camcoords[8];
	dirtex[cnt++] = camcoords[9];
	dirtex[cnt++] = 0;

	//printf("%f\t%f\t%f%f\n",dirtex[cnt-4],dirtex[cnt-8],dirtex[cnt-12],dirtex[cnt-16]);
	
	//Row 2
	//1
	dirtex[cnt++] = camcoords[4] + 0.25 * ( camcoords[13] - camcoords[4] );
	dirtex[cnt++] = camcoords[5] + 0.25 * ( camcoords[14] - camcoords[5] );
	dirtex[cnt++] = camcoords[6] + 0.25 * ( camcoords[15] - camcoords[6] );
	dirtex[cnt++] = 0;

	//2	
	tempvec1[0] = camcoords[4] + 0.25 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] + 0.25 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] + 0.25 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] + 0.25 * ( camcoords[10]  - camcoords[13] );
	tempvec2[1] = camcoords[14] + 0.25 * ( camcoords[11]  - camcoords[14] );
	tempvec2[2] = camcoords[15] + 0.25 * ( camcoords[12]  - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] +  0.25 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] +  0.25 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] +  0.25 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//3
	tempvec1[0] = camcoords[4] +  0.5 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] +  0.5 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] +  0.5 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] +  0.5 * ( camcoords[10] - camcoords[13] );
	tempvec2[1] = camcoords[14] +  0.5 * ( camcoords[11] - camcoords[14] );
	tempvec2[2] = camcoords[15] +  0.5 * ( camcoords[12] - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] + 0.25 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] + 0.25 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] + 0.25 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//4
	tempvec1[0] = camcoords[4] +  0.75 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] +  0.75 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] +  0.75 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] +  0.75 * ( camcoords[10] - camcoords[13] );
	tempvec2[1] = camcoords[14] +  0.75 * ( camcoords[11] - camcoords[14] );
	tempvec2[2] = camcoords[15] +  0.75 * ( camcoords[12] - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] + 0.25 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] + 0.25 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] + 0.25 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//5
	dirtex[cnt++] = camcoords[7] + 0.25 * ( camcoords[10] - camcoords[7] );
	dirtex[cnt++] = camcoords[8] + 0.25 * ( camcoords[11] - camcoords[8] );
	dirtex[cnt++] = camcoords[9] + 0.25 * ( camcoords[12] - camcoords[9] );
	dirtex[cnt++] = 0;

	//printf("%f\t%f\t%f%f\n",dirtex[cnt-4],dirtex[cnt-8],dirtex[cnt-12],dirtex[cnt-16]);

	//Row 3
	//1
	dirtex[cnt++] = camcoords[4] + 0.5 * ( camcoords[13] - camcoords[4] );
	dirtex[cnt++] = camcoords[5] + 0.5 * ( camcoords[14] - camcoords[5] );
	dirtex[cnt++] = camcoords[6] + 0.5 * ( camcoords[15] - camcoords[6] );
	dirtex[cnt++] = 0;

	//2	
	tempvec1[0] = camcoords[4] + 0.25 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] + 0.25 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] + 0.25 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] + 0.25 * ( camcoords[10]  - camcoords[13] );
	tempvec2[1] = camcoords[14] + 0.25 * ( camcoords[11]  - camcoords[14] );
	tempvec2[2] = camcoords[15] + 0.25 * ( camcoords[12]  - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] +  0.5 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] +  0.5 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] +  0.5 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//3
	tempvec1[0] = camcoords[4] +  0.5 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] +  0.5 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] +  0.5 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] +  0.5 * ( camcoords[10] - camcoords[13] );
	tempvec2[1] = camcoords[14] +  0.5 * ( camcoords[11] - camcoords[14] );
	tempvec2[2] = camcoords[15] +  0.5 * ( camcoords[12] - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] + 0.5 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] + 0.5 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] + 0.5 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//4
	tempvec1[0] = camcoords[4] +  0.75 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] +  0.75 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] +  0.75 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] +  0.75 * ( camcoords[10] - camcoords[13] );
	tempvec2[1] = camcoords[14] +  0.75 * ( camcoords[11] - camcoords[14] );
	tempvec2[2] = camcoords[15] +  0.75 * ( camcoords[12] - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] + 0.5 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] + 0.5 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] + 0.5 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//5
	dirtex[cnt++] = camcoords[7] + 0.5 * ( camcoords[10] - camcoords[7] );
	dirtex[cnt++] = camcoords[8] + 0.5 * ( camcoords[11] - camcoords[8] );
	dirtex[cnt++] = camcoords[9] + 0.5 * ( camcoords[12] - camcoords[9] );
	dirtex[cnt++] = 0;


	//printf("%f\t%f\t%f%f\n",dirtex[cnt-4],dirtex[cnt-8],dirtex[cnt-12],dirtex[cnt-16]);
	
	//Row 4
	//1
	dirtex[cnt++] = camcoords[4] + 0.75 * ( camcoords[13] - camcoords[4] );
	dirtex[cnt++] = camcoords[5] + 0.75 * ( camcoords[14] - camcoords[5] );
	dirtex[cnt++] = camcoords[6] + 0.75 * ( camcoords[15] - camcoords[6] );
	dirtex[cnt++] = 0;

	//2	
	tempvec1[0] = camcoords[4] + 0.25 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] + 0.25 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] + 0.25 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] + 0.25 * ( camcoords[10]  - camcoords[13] );
	tempvec2[1] = camcoords[14] + 0.25 * ( camcoords[11]  - camcoords[14] );
	tempvec2[2] = camcoords[15] + 0.25 * ( camcoords[12]  - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] +  0.75 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] +  0.75 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] +  0.75 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//3
	tempvec1[0] = camcoords[4] +  0.5 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] +  0.5 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] +  0.5 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] +  0.5 * ( camcoords[10] - camcoords[13] );
	tempvec2[1] = camcoords[14] +  0.5 * ( camcoords[11] - camcoords[14] );
	tempvec2[2] = camcoords[15] +  0.5 * ( camcoords[12] - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] + 0.75 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] + 0.75 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] + 0.75 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//4
	tempvec1[0] = camcoords[4] +  0.75 * ( camcoords[7] - camcoords[4] );
        tempvec1[1] = camcoords[5] +  0.75 * ( camcoords[8] - camcoords[5] );
	tempvec1[2] = camcoords[6] +  0.75 * ( camcoords[9] - camcoords[6] );
	tempvec1[3] = 0;

	tempvec2[0] = camcoords[13] +  0.75 * ( camcoords[10] - camcoords[13] );
	tempvec2[1] = camcoords[14] +  0.75 * ( camcoords[11] - camcoords[14] );
	tempvec2[2] = camcoords[15] +  0.75 * ( camcoords[12] - camcoords[15] );
	tempvec2[3] = 0;

	dirtex[cnt++] = tempvec1[0] + 0.75 * ( tempvec2[0] - tempvec1[0] );
	dirtex[cnt++] = tempvec1[1] + 0.75 * ( tempvec2[1] - tempvec1[1] );
	dirtex[cnt++] = tempvec1[2] + 0.75 * ( tempvec2[2] - tempvec1[2] );
	dirtex[cnt++] = 0;

	//5
	dirtex[cnt++] = camcoords[7] + 0.75 * ( camcoords[10] - camcoords[7] );
	dirtex[cnt++] = camcoords[8] + 0.75 * ( camcoords[11] - camcoords[8] );
	dirtex[cnt++] = camcoords[9] + 0.75 * ( camcoords[12] - camcoords[9] );
	dirtex[cnt++] = 0;
	
	// Row 5
	//1
	dirtex[cnt++] = camcoords[13];
	dirtex[cnt++] = camcoords[14];
	dirtex[cnt++] = camcoords[15];
	dirtex[cnt++] = 0;

	//2	
	dirtex[cnt++] = camcoords[13] + 0.25  * ( camcoords[10] - camcoords[13] );
	dirtex[cnt++] = camcoords[14] + 0.25  * ( camcoords[11] - camcoords[14] );
	dirtex[cnt++] = camcoords[15] + 0.25  * ( camcoords[12] - camcoords[15] );
	dirtex[cnt++] = 0;

	//3	
	dirtex[cnt++] = camcoords[13] + 0.5  * ( camcoords[10] - camcoords[13] );
	dirtex[cnt++] = camcoords[14] + 0.5  * ( camcoords[11] - camcoords[14] );
	dirtex[cnt++] = camcoords[15] + 0.5  * ( camcoords[12] - camcoords[15] );
	dirtex[cnt++] = 0;

	//4	
	dirtex[cnt++] = camcoords[13] + 0.75  * ( camcoords[10] - camcoords[13] );
	dirtex[cnt++] = camcoords[14] + 0.75  * ( camcoords[11] - camcoords[14] );
	dirtex[cnt++] = camcoords[15] + 0.75  * ( camcoords[12] - camcoords[15] );
	dirtex[cnt++] = 0;

	//5
	dirtex[cnt++] = camcoords[10];
	dirtex[cnt++] = camcoords[11];
	dirtex[cnt++] = camcoords[12];
	dirtex[cnt++] = 0;
	
	//printf("%f\t%f\t%f\t%f\n",dirtex[cnt-4],dirtex[cnt-8],dirtex[cnt-12],dirtex[cnt-16]);

	// 2-D Global Texture for Direction Texture
	cudaChannelFormatDesc cf1 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	//CUDA_SAFE_CALL(cudaMallocArray(&d_dirtex, &cf1, xDim, yDim));
	cutilSafeCall(cudaMemcpyToArray(d_dirtex, 0, 0, dirtex, sizeof(float)*4*xDim*yDim, cudaMemcpyHostToDevice));

	// Set texture parameters
	texdir.addressMode[0] = cudaAddressModeClamp;
	texdir.addressMode[1] = cudaAddressModeClamp;
	texdir.filterMode = cudaFilterModeLinear;
	texdir.normalized = true;

	// Bind the array to the texture
	cutilSafeCall(cudaBindTextureToArray(texdir, d_dirtex, cf1));
}

