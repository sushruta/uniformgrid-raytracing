#ifndef CAMERA_H
#define CAMERA_H

#include <stdio.h>
#include <math.h>

class Camera
{
	public:
		Camera();
		~Camera();

		void setCameraCenter(float _cx, float _cy, float _cz);
		void setCameraLookAt(float _lx, float _ly, float _lz);
		void setCameraUp(float _ux, float _uy, float _uz);
		void setNearFar(float near, float far);

		// camera definition
		float cx, cy, cz;
		float lx, ly, lz;
		float ux, uy, uz;

		float nearPlane, farPlane;
		float worldori[4];

		// camera matrices
		float modelview_matrix[16];
		float projection_matrix[16];
		float mvp_matrix[16];

		// frustum planes, points
		float frustum_plane_eq[6][6];
		float frustumcorner[8][3];

		float *d_cam_position;

		void getGLMatrices();
		void adjustCameraAndPosition();
		void getFrustumProperties();
		void printFrustumCorners();

	private:
		void Intersect3Planes(float *p,float *n1,float *n2,float *n3);
		void getFrustumPlanes();
		void getFrustumCorners();
		void getMVPMatrix();
};

Camera::Camera()
{
	cutilSafeCall(cudaMalloc((void **) &d_cam_position, sizeof(float) * 3));
}

Camera::~Camera()
{
	cutilSafeCall(cudaFree(d_cam_position));
}

void Camera::setNearFar(float near, float far)
{
	nearPlane = near;
	farPlane = far;
}

void Camera::setCameraCenter(float _cx, float _cy, float _cz)
{
	cx = _cx;
	cy = _cy;
	cz = _cz;
}

void Camera::setCameraLookAt(float _lx, float _ly, float _lz)
{
	lx = _lx;
	ly = _ly;
	lz = _lz;
}

void Camera::setCameraUp(float _ux, float _uy, float _uz)
{
	ux = _ux;
	uy = _uy;
	uz = _uz;
}

void Camera::getGLMatrices()
{
	glGetFloatv(GL_MODELVIEW_MATRIX, modelview_matrix);
	glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix);

	getMVPMatrix();

	/*for (int i=0; i<4; i++)
	{
		printf("%f\t%f\t%f\t%f\n", modelview_matrix[i*4+0], modelview_matrix[i*4+1], modelview_matrix[i*4+2], modelview_matrix[i*4+3]);
	}

	printf("\n\n");

	for (int i=0; i<4; i++)
	{
		printf("%f\t%f\t%f\t%f\n", projection_matrix[i*4+0], projection_matrix[i*4+1], projection_matrix[i*4+2], projection_matrix[i*4+3]);
	}

	printf("\n\n");
	
	for (int i=0; i<4; i++)
	{
		printf("%f\t%f\t%f\t%f\n", mvp_matrix[i*4+0], mvp_matrix[i*4+1], mvp_matrix[i*4+2], mvp_matrix[i*4+3]);
	}*/

	return;
}

void Camera::getFrustumProperties()
{
	getFrustumPlanes();
	getFrustumCorners();

	return;
}

void Camera::printFrustumCorners()
{
	printf("Near-Bottom-Left :: %f\t%f\t%f\n\n",frustumcorner[0][0],frustumcorner[0][1],frustumcorner[0][2]);
	printf("Near-Bottom-Right :: %f\t%f\t%f\n\n",frustumcorner[1][0],frustumcorner[1][1],frustumcorner[1][2]);
	printf("Near-Top-Right :: %f\t%f\t%f\n\n",frustumcorner[2][0],frustumcorner[2][1],frustumcorner[2][2]);
	printf("Near-Top-Left :: %f\t%f\t%f\n\n",frustumcorner[3][0],frustumcorner[3][1],frustumcorner[3][2]);
	printf("Far-Bottom-Left :: %f\t%f\t%f\n\n",frustumcorner[4][0],frustumcorner[4][1],frustumcorner[4][2]);
	printf("Far-Bottom-Right :: %f\t%f\t%f\n\n",frustumcorner[5][0],frustumcorner[5][1],frustumcorner[5][2]);
	printf("Far-Top-Right :: %f\t%f\t%f\n\n",frustumcorner[6][0],frustumcorner[6][1],frustumcorner[6][2]);
	printf("Far-Top-Left :: %f\t%f\t%f\n\n",frustumcorner[7][0],frustumcorner[7][1],frustumcorner[7][2]);
}

void Camera::adjustCameraAndPosition()
{
	gluPerspective(FOVY, (float) SCREEN_WIDTH / (float) SCREEN_HEIGHT, nearPlane, farPlane);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(cx, cy, cz, lx, ly, lz, ux, uy, uz);
	worldori[0] = cx;
	worldori[1] = cy;
	worldori[2] = cz;
	worldori[3] = 1.0f;

	return;
}

void Camera::getMVPMatrix()
{
	for(int i=0; i<4; i++)
	{
		for(int k=0; k<4; k++)
		{
			mvp_matrix[i*4+k]=0;
			for(int j=0; j<4; j++)
			{
				mvp_matrix[i*4+k]+=(modelview_matrix[i*4+j] * projection_matrix[j*4+k]);
			}
		}
	}

	return;
}

void Camera::getFrustumPlanes()
{
	float *m = mvp_matrix;

	frustum_plane_eq[0][0] = m[3] - m[0]; //left
	frustum_plane_eq[0][1] = m[7] - m[4];
	frustum_plane_eq[0][2] = m[11] - m[8];
	frustum_plane_eq[0][3] = m[15] - m[12];
	frustum_plane_eq[0][4] = sqrt(frustum_plane_eq[0][0]*frustum_plane_eq[0][0]+frustum_plane_eq[0][1]*frustum_plane_eq[0][1]+frustum_plane_eq[0][2]*frustum_plane_eq[0][2]);

	frustum_plane_eq[1][0] = m[3] + m[0]; //right
	frustum_plane_eq[1][1] = m[7] + m[4];
	frustum_plane_eq[1][2] = m[11] + m[8];
	frustum_plane_eq[1][3] = m[15] + m[12];
	frustum_plane_eq[1][4] = sqrt(frustum_plane_eq[1][0]*frustum_plane_eq[1][0]+frustum_plane_eq[1][1]*frustum_plane_eq[1][1]+frustum_plane_eq[1][2]*frustum_plane_eq[1][2]);

	frustum_plane_eq[2][0] = m[3] + m[1]; //bottom
	frustum_plane_eq[2][1] = m[7] + m[5];
	frustum_plane_eq[2][2] = m[11] + m[9];
	frustum_plane_eq[2][3] = m[15] + m[13];
	frustum_plane_eq[2][4] = sqrt(frustum_plane_eq[2][0]*frustum_plane_eq[2][0]+frustum_plane_eq[2][1]*frustum_plane_eq[2][1]+frustum_plane_eq[2][2]*frustum_plane_eq[2][2]);

	frustum_plane_eq[3][0] = m[3] - m[1]; //top
	frustum_plane_eq[3][1] = m[7] - m[5];
	frustum_plane_eq[3][2] = m[11] - m[9];
	frustum_plane_eq[3][3] = m[15] - m[13];
	frustum_plane_eq[3][4] = sqrt(frustum_plane_eq[3][0]*frustum_plane_eq[3][0]+frustum_plane_eq[3][1]*frustum_plane_eq[3][1]+frustum_plane_eq[3][2]*frustum_plane_eq[3][2]);

	frustum_plane_eq[4][0] = m[3] + m[2]; //near
	frustum_plane_eq[4][1] = m[7] + m[6];
	frustum_plane_eq[4][2] = m[11] + m[10];
	frustum_plane_eq[4][3] = m[15] + m[14];
	frustum_plane_eq[4][4] = sqrt(frustum_plane_eq[4][0]*frustum_plane_eq[4][0]+frustum_plane_eq[4][1]*frustum_plane_eq[4][1]+frustum_plane_eq[4][2]*frustum_plane_eq[4][2]);

	frustum_plane_eq[5][0] = m[3] - m[2]; //far
	frustum_plane_eq[5][1] = m[7] - m[6];
	frustum_plane_eq[5][2] = m[11] - m[10];
	frustum_plane_eq[5][3] = m[15] - m[14];
	frustum_plane_eq[5][4] = sqrt(frustum_plane_eq[5][0]*frustum_plane_eq[5][0]+frustum_plane_eq[5][1]*frustum_plane_eq[5][1]+frustum_plane_eq[5][2]*frustum_plane_eq[5][2]);

	for (int i=0; i<6; i++)
	{
		frustum_plane_eq[i][0] /= frustum_plane_eq[i][4];
		frustum_plane_eq[i][1] /= frustum_plane_eq[i][4];
		frustum_plane_eq[i][2] /= frustum_plane_eq[i][4];
		frustum_plane_eq[i][3] /= frustum_plane_eq[i][4];
	}

	return;
}

void Camera::Intersect3Planes(float *p,float *n1,float *n2,float *n3)
{
	float n2n3[3],n3n1[3],n1n2[3];
	//cross products
	n1n2[0]=(n1[1]*n2[2]-n2[1]*n1[2]);
	n1n2[1]=(n1[2]*n2[0]-n1[0]*n2[2]);
	n1n2[2]=(n1[0]*n2[1]-n2[0]*n1[1]);

	n2n3[0]=(n2[1]*n3[2]-n3[1]*n2[2]);
	n2n3[1]=(n2[2]*n3[0]-n2[0]*n3[2]);
	n2n3[2]=(n2[0]*n3[1]-n3[0]*n2[1]);

	n3n1[0]=(n3[1]*n1[2]-n1[1]*n3[2]);
	n3n1[1]=(n3[2]*n1[0]-n3[0]*n1[2]);
	n3n1[2]=(n3[0]*n1[1]-n1[0]*n3[1]);

	//now the values
	float den=n1[0]*n2n3[0]+n1[1]*n2n3[1]+n1[2]*n2n3[2];
	p[0]=-(n1[3]*n2n3[0]+n2[3]*n3n1[0]+n3[3]*n1n2[0])/den;
	p[1]=-(n1[3]*n2n3[1]+n2[3]*n3n1[1]+n3[3]*n1n2[1])/den;
	p[2]=-(n1[3]*n2n3[2]+n2[3]*n3n1[2]+n3[3]*n1n2[2])/den;
}

void Camera::getFrustumCorners()
{
	Intersect3Planes(frustumcorner[0], frustum_plane_eq[0], frustum_plane_eq[2], frustum_plane_eq[4]);
	Intersect3Planes(frustumcorner[1], frustum_plane_eq[1], frustum_plane_eq[2], frustum_plane_eq[4]);
	Intersect3Planes(frustumcorner[2], frustum_plane_eq[1], frustum_plane_eq[3], frustum_plane_eq[4]);
	Intersect3Planes(frustumcorner[3], frustum_plane_eq[0], frustum_plane_eq[3], frustum_plane_eq[4]);
	Intersect3Planes(frustumcorner[4], frustum_plane_eq[0], frustum_plane_eq[2], frustum_plane_eq[5]);
	Intersect3Planes(frustumcorner[5], frustum_plane_eq[1], frustum_plane_eq[2], frustum_plane_eq[5]);
	Intersect3Planes(frustumcorner[6], frustum_plane_eq[1], frustum_plane_eq[3], frustum_plane_eq[5]);
	Intersect3Planes(frustumcorner[7], frustum_plane_eq[0], frustum_plane_eq[3], frustum_plane_eq[5]);

	return;
}

#endif
