#include <stdio.h>
#include "objLoader.h"

void printVector(obj_vector *v)
{
	printf("%.2f, ", v->e[0]);
	printf("%.2f, ", v->e[1]);
	printf("%.2f, ", v->e[2]);
}

int main(int argc, char **argv)
{
	objLoader *objData = new objLoader();
	objData->load(argv[1]);

	printf("Number of vertices: %i\n", objData->vertexCount);
	printf("Number of vertex normals: %i\n", objData->normalCount);
	printf("Number of texture coordinates: %i\n", objData->textureCount);
	printf("\n");

	printf("Number of faces: %i\n", objData->faceCount);

	for(int i=0; i<objData->faceCount; i++)
	{
		obj_face *o = objData->faceList[i];
		printf(" face ");
		for(int j=0; j<3; j++)
		{
			printVector(objData->vertexList[ o->vertex_index[j] ]);
		}
		printf("\n");
	}

	printf("Number of materials: %i\n", objData->materialCount);
	for(int i=0; i<objData->materialCount; i++)
	{
		obj_material *mtl = objData->materialList[i];
		printf(" name: %s", mtl->name);
		printf(" amb: %.2f ", mtl->amb[0]);
		printf("%.2f ", mtl->amb[1]);
		printf("%.2f\n", mtl->amb[2]);

		printf(" diff: %.2f ", mtl->diff[0]);
		printf("%.2f ", mtl->diff[1]);
		printf("%.2f\n", mtl->diff[2]);

		printf(" spec: %.2f ", mtl->spec[0]);
		printf("%.2f ", mtl->spec[1]);
		printf("%.2f\n", mtl->spec[2]);
		printf(" reflect: %.2f\n", mtl->reflect);
		printf(" trans: %.2f\n", mtl->trans);
		printf(" glossy: %i\n", mtl->glossy);
		printf(" shiny: %i\n", mtl->shiny);
		printf(" refact: %.2f\n", mtl->refract_index);

		printf(" texture: %s\n", mtl->texture_filename);
		printf("\n");
	}

	printf("\n");
	return 0;
}
