#ifndef SCENE_H
#define SCENE_H

#define DEBUG 1

#include <string.h>

#include "obj_parser/obj_parser.h"
#include "obj_parser/objLoader.h"

#include "transformation_kernel.cu"

class Model
{
	public:
		Model(int _frames);
		~Model();

		int *h_facelist;
		float *h_vertexlist;
		float *h_materiallist;
		int *h_materiallist_index;

		int *d_facelist;
		float *d_vertexlist;
		float *d_materiallist;
		int *d_materiallist_index;
		float *d_vertex_orig_list;

		int num_faces;
		int num_vertices;

		int num_frames;
		int num_materials;
		int num_textures;

		void load_model(char *path);
		void some_material(char *file);
		void tmp_model(char *path, int i);
		void rotate_bunny(float rot_factor);
		void init_orig_list(int vertices_size, int vertices_offset);

		float xMin, xMax, yMin, yMax, zMin, zMax;
		bool is_dynamic;
		int size_orig_list;
		int offset_orig_list;

	private:
		//float xMin, xMax, yMin, yMax, zMin, zMax;
		void min_max(int axis, obj_vector *vec);

		objLoader *objData;

		int __numblocks_ds;
		int __numthreads_ds;
		int flaggg;
};

Model::Model(int _frames)
{
	num_frames = _frames;
	if (num_frames > 1)
		is_dynamic = true;
	else
		is_dynamic = false;

	flaggg = 0;
}

void Model::tmp_model(char *path, int i)
{
	int dirPathSize = strlen(path);
	int totalSize = dirPathSize + 15;
	char *filename = (char *) malloc(sizeof(char) * totalSize);
	char frame[10];

	filename[0] = '\0';
	sprintf(frame, "f_%d.obj", i);
	filename = strcat(filename, path);
	filename = strcat(filename, "/");
	filename = strcat(filename, frame);
	filename[strlen(filename)] = '\0';

	printf("file to be loaded: %s\n", filename);

	objData = new objLoader();
	objData->load(filename);

	// now the vertex data
	xMin = 9999.9f;
	yMin = 9999.9f;
	zMin = 9999.9f;
	xMax = -9999.9f;
	yMax = -9999.9f;
	zMax = -9999.9f;

	float *vertlist = &h_vertexlist[0*num_vertices*3];
	for (int v=0; v<objData->vertexCount; v++)
	{
		vertlist[v*3+0] = objData->vertexList[v]->e[0];
		(vertlist[v*3+0] < xMin) ? xMin = vertlist[v*3+0] : vertlist[v*3+0] = vertlist[v*3+0];
		(vertlist[v*3+0] > xMax) ? xMax = vertlist[v*3+0] : vertlist[v*3+0] = vertlist[v*3+0];

		vertlist[v*3+1] = objData->vertexList[v]->e[1];
		(vertlist[v*3+1] < yMin) ? yMin = vertlist[v*3+1] : vertlist[v*3+1] = vertlist[v*3+1];
		(vertlist[v*3+1] > yMax) ? yMax = vertlist[v*3+1] : vertlist[v*3+1] = vertlist[v*3+1];

		vertlist[v*3+2] = objData->vertexList[v]->e[2];
		(vertlist[v*3+2] < zMin) ? zMin = vertlist[v*3+2] : vertlist[v*3+2] = vertlist[v*3+2];
		(vertlist[v*3+2] > zMax) ? zMax = vertlist[v*3+2] : vertlist[v*3+2] = vertlist[v*3+2];
	}
#if DEBUG == 1
	printf("for the scene in frame#%d, min : (%f, %f, %f) and max : (%f, %f, %f)\n", i, xMin, yMin, zMin, xMax, yMax, zMax);
#endif
	// copy other stuff if required
	delete(objData);

	cutilSafeCall(cudaMemcpy(d_facelist, h_facelist, sizeof(int) * num_faces * 3, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_vertexlist, h_vertexlist, sizeof(float) * num_vertices * 3, cudaMemcpyHostToDevice));
}

void Model::rotate_bunny(float rot_factor)
{
	//this function is only for bunny 
	//in the conmference model

	// first copy the data to the main vertex list
	int numThreads = 256;
	int numBlocks = size_orig_list / numThreads + 1;

	dim3 copyGridSize(numBlocks, 1, 1);
	dim3 copyThreadSize(numThreads, 1, 1);

	cudaThreadSynchronize();
	copy_data_transform<<< copyGridSize, copyThreadSize >>>(d_vertexlist, d_vertex_orig_list, size_orig_list, offset_orig_list, rot_factor);
	cudaThreadSynchronize();

	return;
}

void Model::load_model(char *path)
{
	if (is_dynamic && flaggg == 0)
	{
		int dirPathSize = strlen(path);
		int totalSize = dirPathSize + 15;
		char *filename = (char *) malloc(sizeof(char) * totalSize);
		char frame[10];
		flaggg = 1;

		//for (int i=0; i<num_frames; i++)
		if (flaggg == 1)
		{
			int i=0;
			filename[0] = '\0';
			sprintf(frame, "f_%d.obj", i);
			filename = strcat(filename, path);
			filename = strcat(filename, "/");
			filename = strcat(filename, frame);
			filename[strlen(filename)] = '\0';

#if DEBUG == 1
			printf("file to be loaded: %s\n", filename);
#endif
			
			// in a loop load all the frames' data
			objData = new objLoader();
			objData->load(filename);

			num_faces = objData->faceCount;
			num_vertices = objData->vertexCount;



			if (i == 0)	// allocate memory only during the first frame
			{
				h_vertexlist = (float *) malloc(sizeof(float) * num_vertices * 3);
				h_facelist = (int *) malloc(sizeof(int) * num_faces * 3);

				cutilSafeCall(cudaMalloc((void **) &d_vertexlist, sizeof(float) * num_vertices * 3));
				cutilSafeCall(cudaMalloc((void **) &d_facelist, sizeof(int) * num_faces * 3));

				for (int f=0; f<num_faces; f++)
				{
					h_facelist[f*3+0] = objData->faceList[f]->vertex_index[0];
					h_facelist[f*3+1] = objData->faceList[f]->vertex_index[1];
					h_facelist[f*3+2] = objData->faceList[f]->vertex_index[2];
				}
			}

			// now the vertex data
			xMin = 9999.9f;
			yMin = 9999.9f;
			zMin = 9999.9f;
			xMax = -9999.9f;
			yMax = -9999.9f;
			zMax = -9999.9f;

			float *vertlist = &h_vertexlist[i*num_vertices*3];
			for (int v=0; v<objData->vertexCount; v++)
			{
				vertlist[v*3+0] = objData->vertexList[v]->e[0];
				(vertlist[v*3+0] < xMin) ? xMin = vertlist[v*3+0] : vertlist[v*3+0] = vertlist[v*3+0];
				(vertlist[v*3+0] > xMax) ? xMax = vertlist[v*3+0] : vertlist[v*3+0] = vertlist[v*3+0];

				vertlist[v*3+1] = objData->vertexList[v]->e[1];
				(vertlist[v*3+1] < yMin) ? yMin = vertlist[v*3+1] : vertlist[v*3+1] = vertlist[v*3+1];
				(vertlist[v*3+1] > yMax) ? yMax = vertlist[v*3+1] : vertlist[v*3+1] = vertlist[v*3+1];

				vertlist[v*3+2] = objData->vertexList[v]->e[2];
				(vertlist[v*3+2] < zMin) ? zMin = vertlist[v*3+2] : vertlist[v*3+2] = vertlist[v*3+2];
				(vertlist[v*3+2] > zMax) ? zMax = vertlist[v*3+2] : vertlist[v*3+2] = vertlist[v*3+2];
			}
#if DEBUG == 1
			printf("for the scene in frame#%d, min : (%f, %f, %f) and max : (%f, %f, %f)\n", i, xMin, yMin, zMin, xMax, yMax, zMax);
#endif
			// copy other stuff if required
			delete(objData);
		}

		// copy all the vertices to GPU
		cutilSafeCall(cudaMemcpy(d_facelist, h_facelist, sizeof(int) * num_faces * 3, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_vertexlist, h_vertexlist, sizeof(float) * num_vertices * 3, cudaMemcpyHostToDevice));
	}
	else
	{
		printf("static loading in action!\n");
		objData = new objLoader();
		objData->load(path);

		num_faces = objData->faceCount;
		num_vertices = objData->vertexCount;

		printf("vertices : %d\tfaces : %d\n", num_vertices, num_faces);

		/*// for replication!!
		num_faces *= 3;
		num_vertices *= 3;
		// replication ends!!*/

		h_vertexlist = (float *) malloc(sizeof(float) * num_vertices * 3);
		h_facelist = (int *) malloc(sizeof(int) * num_faces * 3);
		h_materiallist_index = (int *) malloc(sizeof(int) * num_faces);

		cutilSafeCall(cudaMalloc((void **) &d_vertexlist, sizeof(float) * num_vertices * 3));
		cutilSafeCall(cudaMalloc((void **) &d_facelist, sizeof(int) * num_faces * 3));
		cutilSafeCall(cudaMalloc((void **) &d_materiallist_index, sizeof(int) * num_faces));

		for (int f=0; f<objData->faceCount; f++)
		{
			h_facelist[f*3+0] = objData->faceList[f]->vertex_index[0];
			h_facelist[f*3+1] = objData->faceList[f]->vertex_index[1];
			h_facelist[f*3+2] = objData->faceList[f]->vertex_index[2];
		
			/*// replication
			h_facelist[objData->faceCount * 3 +f*3+0] = objData->faceList[f]->vertex_index[0] + objData->vertexCount*3;
			h_facelist[objData->faceCount * 3 +f*3+1] = objData->faceList[f]->vertex_index[1] + objData->vertexCount*3;
			h_facelist[objData->faceCount * 3 +f*3+2] = objData->faceList[f]->vertex_index[2] + objData->vertexCount*3;
			
			h_facelist[objData->faceCount * 6 + f*3+0] = objData->faceList[f]->vertex_index[0] + objData->vertexCount*6;
			h_facelist[objData->faceCount * 6 + f*3+1] = objData->faceList[f]->vertex_index[1] + objData->vertexCount*6;
			h_facelist[objData->faceCount * 6 + f*3+2] = objData->faceList[f]->vertex_index[2] + objData->vertexCount*6;
			// replication*/

			h_materiallist_index[f] = objData->faceList[f]->material_index;
			/*// replication
			h_materiallist_index[objData->faceCount * 3 + f] = objData->faceList[f]->material_index;
			h_materiallist_index[objData->faceCount * 6 + f] = objData->faceList[f]->material_index;
			// replication*/
		}

		// now the vertex data
		xMin = 9999.9f;
		yMin = 9999.9f;
		zMin = 9999.9f;
		xMax = -9999.9f;
		yMax = -9999.9f;
		zMax = -9999.9f;

		for (int v=0; v<objData->vertexCount; v++)
		{
			h_vertexlist[v*3+0] = objData->vertexList[v]->e[0];
			(h_vertexlist[v*3+0] < xMin) ? xMin = h_vertexlist[v*3+0] : h_vertexlist[v*3+0] = h_vertexlist[v*3+0];
			(h_vertexlist[v*3+0] > xMax) ? xMax = h_vertexlist[v*3+0] : h_vertexlist[v*3+0] = h_vertexlist[v*3+0];

			h_vertexlist[v*3+1] = objData->vertexList[v]->e[1];
			(h_vertexlist[v*3+1] < yMin) ? yMin = h_vertexlist[v*3+1] : h_vertexlist[v*3+1] = h_vertexlist[v*3+1];
			(h_vertexlist[v*3+1] > yMax) ? yMax = h_vertexlist[v*3+1] : h_vertexlist[v*3+1] = h_vertexlist[v*3+1];

			h_vertexlist[v*3+2] = objData->vertexList[v]->e[2];
			(h_vertexlist[v*3+2] < zMin) ? zMin = h_vertexlist[v*3+2] : h_vertexlist[v*3+2] = h_vertexlist[v*3+2];
			(h_vertexlist[v*3+2] > zMax) ? zMax = h_vertexlist[v*3+2] : h_vertexlist[v*3+2] = h_vertexlist[v*3+2];

			/*// replication			
			h_vertexlist[objData->vertexCount *3 + v*3+0] = objData->vertexList[v]->e[0];
			(h_vertexlist[objData->vertexCount *3 + v*3+0] < xMin) ? xMin = h_vertexlist[objData->vertexCount *3 + v*3+0] : h_vertexlist[objData->vertexCount *3 + v*3+0] = h_vertexlist[objData->vertexCount *3 + v*3+0];
			(h_vertexlist[objData->vertexCount *3 + v*3+0] > xMax) ? xMax = h_vertexlist[objData->vertexCount *3 + v*3+0] : h_vertexlist[objData->vertexCount *3 + v*3+0] = h_vertexlist[objData->vertexCount *3 + v*3+0];

			h_vertexlist[objData->vertexCount *3 + v*3+1] = objData->vertexList[v]->e[1];
			(h_vertexlist[objData->vertexCount *3 + v*3+1] < yMin) ? yMin = h_vertexlist[objData->vertexCount *3 + v*3+1] : h_vertexlist[objData->vertexCount *3 + v*3+1] = h_vertexlist[objData->vertexCount *3 + v*3+1];
			(h_vertexlist[objData->vertexCount *3 + v*3+1] > yMax) ? yMax = h_vertexlist[objData->vertexCount *3 + v*3+1] : h_vertexlist[objData->vertexCount *3 + v*3+1] = h_vertexlist[objData->vertexCount *3 + v*3+1];

			h_vertexlist[objData->vertexCount *3 + v*3+2] = objData->vertexList[v]->e[2];
			(h_vertexlist[objData->vertexCount *3 + v*3+2] < zMin) ? zMin = h_vertexlist[objData->vertexCount *3 + v*3+2] : h_vertexlist[objData->vertexCount *3 + v*3+2] = h_vertexlist[objData->vertexCount *3 + v*3+2];
			(h_vertexlist[objData->vertexCount *3 + v*3+2] > zMax) ? zMax = h_vertexlist[objData->vertexCount *3 + v*3+2] : h_vertexlist[objData->vertexCount *3 + v*3+2] = h_vertexlist[objData->vertexCount *3 + v*3+2];
			
			h_vertexlist[objData->vertexCount *6 + v*3+0] = objData->vertexList[v]->e[0];
			(h_vertexlist[objData->vertexCount *6 + v*3+0] < xMin) ? xMin = h_vertexlist[objData->vertexCount *6 + v*3+0] : h_vertexlist[objData->vertexCount *6 + v*3+0] = h_vertexlist[objData->vertexCount *6 + v*3+0];
			(h_vertexlist[objData->vertexCount *6 + v*3+0] > xMax) ? xMax = h_vertexlist[objData->vertexCount *6 + v*3+0] : h_vertexlist[objData->vertexCount *6 + v*3+0] = h_vertexlist[objData->vertexCount *6 + v*3+0];

			h_vertexlist[objData->vertexCount *6 + v*3+1] = objData->vertexList[v]->e[1];
			(h_vertexlist[objData->vertexCount *6 + v*3+1] < yMin) ? yMin = h_vertexlist[objData->vertexCount *6 + v*3+1] : h_vertexlist[objData->vertexCount *6 + v*3+1] = h_vertexlist[objData->vertexCount *6 + v*3+1];
			(h_vertexlist[objData->vertexCount *6 + v*3+1] > yMax) ? yMax = h_vertexlist[objData->vertexCount *6 + v*3+1] : h_vertexlist[objData->vertexCount *6 + v*3+1] = h_vertexlist[objData->vertexCount *6 + v*3+1];

			h_vertexlist[objData->vertexCount *6 + v*3+2] = objData->vertexList[v]->e[2];
			(h_vertexlist[objData->vertexCount *6 + v*3+2] < zMin) ? zMin = h_vertexlist[objData->vertexCount *6 + v*3+2] : h_vertexlist[objData->vertexCount *6 + v*3+2] = h_vertexlist[objData->vertexCount *6 + v*3+2];
			(h_vertexlist[objData->vertexCount *6 + v*3+2] > zMax) ? zMax = h_vertexlist[objData->vertexCount *6 + v*3+2] : h_vertexlist[objData->vertexCount *6 + v*3+2] = h_vertexlist[objData->vertexCount *6 + v*3+2];
			// replication*/
		}

#if DEBUG == 1
		printf("for the scene, min : (%f, %f, %f) and max : (%f, %f, %f)\n", xMin, yMin, zMin, xMax, yMax, zMax);
#endif

		// copy data onto the GPU
		cutilSafeCall(cudaMemcpy(d_vertexlist, h_vertexlist, sizeof(float) * num_vertices * 3, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_facelist, h_facelist, sizeof(int) * num_faces * 3, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(d_materiallist_index, h_materiallist_index, sizeof(int) * num_faces, cudaMemcpyHostToDevice));

		delete(objData);
	}

	return;
}

void Model::init_orig_list(int vertices_size, int vertices_offset)
{
	cutilSafeCall(cudaMalloc((void **) &d_vertex_orig_list, sizeof(float) * 3 * vertices_size));

	printf("offset : %d size : %d\n", vertices_offset, vertices_size);

	float *h_vertex_orig_list = (float *) malloc(sizeof(float) * vertices_size * 3);
	for (int i=0; i<vertices_size; i++)
	{
		h_vertex_orig_list[i*3 + 0] = h_vertexlist[(vertices_offset + i)*3 + 0];
		h_vertex_orig_list[i*3 + 1] = h_vertexlist[(vertices_offset + i)*3 + 1];
		h_vertex_orig_list[i*3 + 2] = h_vertexlist[(vertices_offset + i)*3 + 2];
	}

	size_orig_list = vertices_size;
	offset_orig_list = vertices_offset;
	cutilSafeCall(cudaMemcpy(d_vertex_orig_list, h_vertex_orig_list, sizeof(float) * vertices_size * 3, cudaMemcpyHostToDevice));
	
	/*float *h_test = (float *) malloc(sizeof(float) * vertices_size * 3);
	cutilSafeCall(cudaMemcpy(h_test, d_vertex_orig_list, sizeof(float) * vertices_size * 3, cudaMemcpyDeviceToHost));

	for (int i=0; i<vertices_size; i++)
	{
		printf("value : %f %f %f\n", h_test[i*3+0], h_test[i*3+1], h_test[i*3+2]);
		//if (h_test[i] != h_vertexlist[vertices_offset + i])
		//	printf("NO NOT CORRECT!!!!\n");
	}

	exit(-1);*/

	free(h_vertex_orig_list);
	return;
}

void Model::some_material(char *file)
{
	FILE *fp = fopen(file, "r");
	char cjunk[50];
	num_materials=0;
	num_textures=0;

	while (fscanf(fp, "%s", cjunk) != EOF)
	{
		if (strcmp(cjunk, "newmtl") == 0)
			num_materials++;

		if (strcmp(cjunk, "map") == 0)
			num_textures++;

		if (strcmp(cjunk, "NA") == 0)
			num_textures--;
	}
	fclose(fp);

#if DEBUG == 1
	printf("number of materials : %d\n", num_materials);
	printf("number of textures : %d\n", num_textures);
#endif

	h_materiallist = (float *) malloc(sizeof(float) * num_materials * MATERIAL_SIZE);
	cutilSafeCall(cudaMalloc((void **) &d_materiallist, sizeof(float) * num_materials * MATERIAL_SIZE));

	int tex_count = 0;
	fp = fopen(file, "r");
	for (int mt=0; mt<num_materials; mt++)
	{
		for (int i=0; i<3; i++)
			fscanf(fp, "%s", cjunk);

		fscanf(fp, "%f", &h_materiallist[mt*MATERIAL_SIZE + 0]);
		fscanf(fp, "%f", &h_materiallist[mt*MATERIAL_SIZE + 1]);
		fscanf(fp, "%f", &h_materiallist[mt*MATERIAL_SIZE + 2]);

		fscanf(fp, "%s", cjunk);

		fscanf(fp, "%f", &h_materiallist[mt*MATERIAL_SIZE + 3]);
		fscanf(fp, "%f", &h_materiallist[mt*MATERIAL_SIZE + 4]);
		fscanf(fp, "%f", &h_materiallist[mt*MATERIAL_SIZE + 5]);

		for (int i=0; i<11; i++)
			fscanf(fp, "%s", cjunk);

		fscanf(fp, "%s", cjunk);
		if (strcmp(cjunk, "NA") != 0)
		{
			printf("texture file is : %s\n", cjunk);

			// copy the textures here

			tex_count++;
		}

		//printf("material # %d added!\n", mt);

		//printf("ambient : %f, %f, %f\n", h_materiallist[mt*MATERIAL_SIZE + 0], h_materiallist[mt*MATERIAL_SIZE + 1], h_materiallist[mt*MATERIAL_SIZE + 2]);
		//printf("diffuse : %f, %f, %f\n", h_materiallist[mt*MATERIAL_SIZE + 3], h_materiallist[mt*MATERIAL_SIZE + 4], h_materiallist[mt*MATERIAL_SIZE + 5]);
	}
	fclose(fp);

	// copy data onto the GPU
	cutilSafeCall(cudaMemcpy(d_materiallist, h_materiallist, sizeof(float) * num_materials * MATERIAL_SIZE, cudaMemcpyHostToDevice));

	return;
}

Model::~Model()
{
#if DEBUG == 1
	printf("cleaning up Model instantiation...\n");
#endif

	cutilSafeCall(cudaFree(d_facelist));
	cutilSafeCall(cudaFree(d_vertexlist));
	cutilSafeCall(cudaFree(d_materiallist_index));
	cutilSafeCall(cudaFree(d_materiallist));
	cutilSafeCall(cudaFree(d_vertex_orig_list));

	free(h_vertexlist);
	free(h_facelist);
	free(h_materiallist);
	free(h_materiallist_index);
}

#endif
