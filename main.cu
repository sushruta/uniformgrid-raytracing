#define GL_GLEXT_PROTOTYPES

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cutil.h>
#include <cuda_gl_interop.h>
#include <cutil_inline.h>

#include <GL/glut.h>
#include "cudpp/cudpp.h"

#include "main.cu.h"
#include "scene.h"
#include "camera.h"
#include "frustum_grid.h"
#include "uniform_grid.h"

#include "decision_data.h"
#include "frustum_tracer.h"

#include "shader.h"

Model *model;
FrustumGrid *fGrid;
Camera *camera;
FrustumTracer *fTracer;
DecisionData *dData;
Shader *shader;

int frame_cnt = 0;
bool frame_freeze = false;

#include "per_app_funcs.h"
#include "per_frame_funcs.h"

char *fl;

void cleanup()
{
	printf("cleaning up...\n");

	if (!model)
		delete(model);
	if (!fGrid)
		delete(fGrid);
	if (!camera)
		delete(camera);
	if (!fTracer)
		delete(fTracer);
	if (!dData)
		delete(dData);
	if (!shader)
		delete(shader);
}

void display()
{
	if (frame_freeze)
		return;

	// increment the frame count
	frame_cnt++;
	printf("\n\n------\tFrame # %d\t------\n", frame_cnt);

	//model->rotate_bunny(lightRotFactor);

	// display function
	glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
	updateLightPosition();
	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	/*// define the camera
	camera->setCameraCenter(_cx, _cy, _cz);
	camera->setCameraLookAt(_lx, _ly, _lz);
	camera->setCameraUp(_ux, _uy, _uz);*/
	
	camera->setCameraCenter(0, 20, 5);
	camera->setCameraLookAt(27, 1, 3);
	camera->setCameraUp(0, 0, 1);
	camera->setNearFar(0.1, 100);

	camera->setCameraCenter(3, 15, 5);
	camera->setCameraLookAt(13, 13, 3);
	camera->setCameraUp(0, 0, 1);
	camera->setNearFar(0.1, 100);
	
	/*//camera->setCameraCenter(16.88, 2.239, 4.620);
	camera->setCameraCenter(8.88, -0.2, 3.620);
	camera->setCameraLookAt(0, 0, 0);
	camera->setCameraUp(0, 0, 1);
	camera->setNearFar(0.1, 5000);

	camera->setCameraCenter(0.065, 0.13, 0.13);
	camera->setCameraLookAt(-0.06, 0.11, 0);
	camera->setCameraUp(0, 1, 0);
	camera->setNearFar(0.01, 1.5);*/
	
	/*camera->setCameraCenter(2.05, 1.3, 2.05);
	//camera->setCameraCenter(1.125, 0.95, 1.125);
	camera->setCameraLookAt(0, 0.15, 0);
	camera->setCameraUp(0, 1, 0);
	camera->setNearFar(0.1, 100);*/
	
	/*camera->setCameraCenter(4, 17, 6.6);
	camera->setCameraLookAt(25, 3, 3);
	camera->setCameraUp(0, 0, 1);
	camera->setNearFar(0.1, 100);*/

	/*// for cornell box
	camera->setCameraCenter((model->xMin + model->xMax) / 2.0f, (model->yMin + model->yMax) / 2.0f, (model->zMin + model->zMax) / 2.0f);
	camera->setCameraLookAt((model->xMin + model->xMax) / 2.0f, (model->yMin + model->yMax) / 2.0f, (model->zMin + model->zMax));
	camera->setCameraUp(0, 1, 0);
	camera->setNearFar(1, 650);*/
	
	// put the values in the GL system
	camera->adjustCameraAndPosition();

	// now get the frustum information
	camera->getGLMatrices();
	camera->getFrustumProperties();

	// copy the camera's position somewhere (create a backup copy)
	cutilSafeCall(cudaMemcpy(camera->d_cam_position, camera->worldori, sizeof(float)*3, cudaMemcpyHostToDevice));

	fillCoordinatesData();

	/******* DS Building starts here ********/

	build_frustum_grid();

	/********DS Building ends here  ********/

	// register the pbo object with CUDA
	cutilSafeCall(cudaGLMapBufferObject((void**)&d_image, pbo));

	// now start raytracing
	fTracer->trace(fGrid->d_triangle_value_list, fGrid->d_span, fGrid->d_offset, 
			dData->d_primary_ray_normal, dData->d_primary_ray_t_value, dData->d_primary_ray_direction,
			dData->d_is_shadowed, dData->d_intersect_id, 
			model->d_vertexlist, model->d_facelist);

	// do some shadow work here!!
	for (int l=0; l<h_numLights; l++)
	{
		printf("\n>>>\tdealing with light# %d...\n", l);
		//every light is treated like a camera

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		// do some initializations here
		//camera->setCameraCenter(h_light_position[0], h_light_position[1], h_light_position[2]);
		camera->setCameraCenter(14, 13, 8);
		//camera->setCameraCenter(14.5, 13, 6.5);
		//camera->setCameraLookAt(11.0 - lightRotFactor * 1.5, 10.0, 5.5 + lightRotFactor * 0.5);
		//camera->setCameraLookAt(0, 0, 0);
		camera->setCameraLookAt(14, 13, 0.0);
		camera->setCameraUp(0, 1, 0);
		camera->setNearFar(0.1, 100);
		
		camera->adjustCameraAndPosition();
		camera->getGLMatrices();
		camera->getFrustumProperties();

		fillCoordinatesData();

		getRayGridMapping(dData->d_primary_ray_t_value, dData->d_primary_ray_direction, dData->d_x_list, dData->d_y_list, dData->d_angle, camera->d_cam_position);

		float x_max = -9999.9f, y_max = -9999.9f;
		cudaMemcpy(dData->h_x_list, dData->d_x_list, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost);
		cudaMemcpy(dData->h_y_list, dData->d_y_list, sizeof(float) * SCREEN_HEIGHT * SCREEN_WIDTH, cudaMemcpyDeviceToHost);

		for (int j=0; j<SCREEN_HEIGHT * SCREEN_WIDTH; j++)
		{
			if (x_max < dData->h_x_list[j])
				x_max = dData->h_x_list[j];

			if (y_max < dData->h_y_list[j])
				y_max = dData->h_y_list[j];
		}
		x_max = M_PI;
		y_max = M_PI;

		getEffectiveRayGridMapping(dData->d_primary_ray_t_value, dData->d_primary_ray_direction, dData->d_map, camera->d_cam_position, x_max, y_max);
		//getRayGridMapping(dData->d_primary_ray_t_value, dData->d_primary_ray_direction, dData->d_map, camera->d_cam_position, 0.1);
		printf("x_extent = %f\ty_extent = %f\n", x_max, y_max);

		build_secondary_frustum_grid(x_max, y_max);

		// do some processing
		// and rearrangement
		processData();

		check_for_shadows(l);

		// do secondary ray cast now
		printf("\n>>>\tdone with light# %d\n", l);
	}

	if (frame_cnt < 2)
	{
			shader->simpleShade(d_image,
					dData->d_primary_ray_normal, dData->d_primary_ray_t_value, dData->d_primary_ray_direction, 
					dData->d_intersect_id,
					camera->d_cam_position,
					model->d_materiallist_index, model->d_materiallist, model->num_materials);
	}
	else
	{	
		shader->spotlight_shade(d_image,
				dData->d_primary_ray_normal, dData->d_primary_ray_t_value, dData->d_primary_ray_direction, 
				dData->d_intersect_id, camera->d_cam_position,
				model->d_materiallist_index, model->d_materiallist, model->num_materials);
	}

	//shader->perlinShade(d_image, dData->d_primary_ray_t_value, dData->d_primary_ray_direction, camera->d_cam_position, dData->d_intersect_id);

	shader->add_shadows(d_image, dData->d_is_shadowed);

	//dData->printSomething();

	/*char cmd[75];
	CUDA_SAFE_CALL(cudaMemcpy(h_image,d_image,sizeof(unsigned char)*SCREEN_WIDTH*SCREEN_HEIGHT*3 , cudaMemcpyDeviceToHost));
	CUT_CHECK_ERROR(" cudaMemcpy failed ");
	char    imgname[128];
	sprintf(imgname,"results/f-%s.ppm",fl);
	printf("file : %s\n", imgname);
	writePPM(imgname);
	printf("file : %s\n", imgname);
	sprintf(cmd, "convert results/f-%s.ppm results/f-%s.jpg", fl, fl);
	printf("file : %s\n", imgname);
	system(cmd);
	printf("file : %s\n", imgname);
	sprintf(cmd, "convert results/f-%s.jpg -flip results/f-%s.jpg", fl, fl);
	printf("file : %s\n", imgname);
	system(cmd);
	printf("file : %s\n", imgname);*/
	
	char cmd[75];
	CUDA_SAFE_CALL(cudaMemcpy(h_image,d_image,sizeof(unsigned char)*SCREEN_WIDTH*SCREEN_HEIGHT*3 , cudaMemcpyDeviceToHost));
	CUT_CHECK_ERROR(" cudaMemcpy failed ");
	char    imgname[128];
	sprintf(imgname,"results/img-%d.ppm", frame_cnt+19);
	printf("file : %s\n", imgname);
	writePPM(imgname);
	printf("file : %s\n", imgname);
	sprintf(cmd, "convert results/img-%d.ppm results/img-%d.jpg", frame_cnt+20, frame_cnt+19);
	printf("file : %s\n", imgname);
	system(cmd);
	printf("file : %s\n", imgname);
	sprintf(cmd, "convert results/img-%d.jpg -flip results/fimg-%d.jpg", frame_cnt+19, frame_cnt+19);
	printf("file : %s\n", imgname);
	system(cmd);
	printf("file : %s\n", imgname);

	// unregister it
	cutilSafeCall(cudaGLUnmapBufferObject(pbo));
	
	//Render from Texture
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1, 1, -1, 1, 1, 5);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// load texture from pbo
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	glColor3f(1, 1, 1);
	glBegin(GL_QUADS);
	glTexCoord2i(0, 0);
	glVertex3f(-1, -1, -3);

	glTexCoord2i(0, 1);
	glVertex3f(-1, 1, -3);

	glTexCoord2i(1, 1);
	glVertex3f(1, 1, -3);

	glTexCoord2i(1, 0);
	glVertex3f(1, -1, -3);
	glEnd();

	glutSwapBuffers();

	if (frame_cnt >= MAXFRAMES)
	{
		cleanup();
		exit(0);
	}
}

void keyboard(unsigned char key, int x, int y)
{
	// all the keyboard related stuff here
	if (key == 27)
	{
		cleanup();
		exit(0);
	}
	else if (key == 32)
	{
		frame_freeze = !frame_freeze;
	}
}

void initData()
{
	// allocate memory for split stuff
	NUMBLOCKSDS = model->num_faces / NUMTHREADSDS + 1;
	h_modelParams[0] = model->num_faces;

	h_image = (unsigned char *) malloc(sizeof(unsigned char) * IMAGE_SIZE * 3);

	dirtex = (float *) malloc(4 * sizeof(float) * xDim * yDim);
	cudaChannelFormatDesc cf1 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cutilSafeCall(cudaMallocArray(&d_dirtex, &cf1, xDim, yDim));
}

int main(int argc, char **argv)
{
	// init the openGL subsystem
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	glutCreateWindow("Ray Tracing on CUDA");
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(keyboard);

	//create pixel buffer object
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, 1*SCREEN_WIDTH*SCREEN_HEIGHT*sizeof(unsigned int), NULL, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	cutilSafeCall(cudaGLRegisterBufferObject(pbo));

	//create texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,SCREEN_WIDTH,SCREEN_HEIGHT,0,GL_RGB,GL_UNSIGNED_BYTE,NULL);

	// load the scene
	init_model(argc, argv);

	// initialize all the data parameters
	initData();

	fGrid = new FrustumGrid(h_modelParams[0]);
	camera = new Camera();
	fTracer = new FrustumTracer();
	dData = new DecisionData();
	shader = new Shader();

	/*float *h_test = (float *) malloc(sizeof(float) * 35947 * 3);
	cutilSafeCall(cudaMemcpy(h_test, model->h_vertexlist[166940], sizeof(float) * 35947 * 3, cudaMemcpyDeviceToHost));

	for (int i=0; i<35947; i++)
	{
		printf("value : %f %f %f\n", h_test[i*3+0], h_test[i*3+1], h_test[i*3+2]);
		//if (h_test[i] != h_vertexlist[vertices_offset + i])
		//      printf("NO NOT CORRECT!!!!\n");
	}

	free(h_test);
	exit(-1);*/

	// only for bunny
	//model->init_orig_list(35947, 166940);

	fl = argv[4];

	// call the display function
	glutMainLoop();

	// cleanup
	cleanup();

	return 0;
}
