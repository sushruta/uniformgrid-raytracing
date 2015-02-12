#define MAXFRAMES       100
#define WRITETOPPM      0
#define DEFORM          0
#define ROT             0
#define FRONT_FACING    0
#define SPHERICAL_MAPPING 0

#define SHADOWS 0

#define SCREEN_WIDTH    1024
#define SCREEN_HEIGHT   1024
#define IMAGE_SIZE	(SCREEN_WIDTH * SCREEN_HEIGHT)

#define FOVY            45

#define NUM_BLOCKS_X    128
#define NUM_BLOCKS_Y    128
#define NUM_SLABS       1

#define NUM_CUDA_BLOCKS_X       128
#define NUM_CUDA_BLOCKS_Y       128

#define MAX_BIN         33554432

#define NUM_THREADS_X   8
#define NUM_THREADS_Y   8

#define MAX_TRIANGLES   (NUM_THREADS_X * NUM_THREADS_Y)
#define MAX_TRIANGLES_BLOCK     ((NUM_THREADS_X * NUM_THREADS_Y)/4)
#define TOTAL_THREADS   (NUM_THREADS_X * NUM_THREADS_Y)

#define MAX_RAYS_PER_BLOCK      64

#define MATERIAL_SIZE	6

#define NUMTHREADSDS    64
int     NUMBLOCKSDS = 0;

int h_maxBinsPerTriangle;
int h_numLights = 1;

#define EPSILON	0.000000000000000000001//0.0000000001

#define CROSS(dest,v1,v2) \
	        dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
dest[2]=v1[0]*v2[1]-v1[1]*v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2) \
	        dest[0]=v1[0]-v2[0]; \
        dest[1]=v1[1]-v2[1]; \
        dest[2]=v1[2]-v2[2];

#define NORMALIZE(A)    {float l=1/sqrtf(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);A[0]*=l;A[1]*=l;A[2]*=l;}

texture<float4, 2, cudaReadModeElementType> texdir;

float camcoords[16*4];
float h_light_position[6];
__device__ __constant__ float dd_camcoords[16*4];
__device__ __constant__ float dd_light_position[6];

// 0 - number of triangles
int h_modelParams[1];

GLuint pbo;
GLuint tex;

float *dirtex;
cudaArray *d_dirtex;
int xDim = 5;
int yDim = 5;

unsigned char *h_image;
unsigned char *d_image;

unsigned int *d_map;

float _cx = 19.808, _cy = 2.239, _cz = 1.620;
float _lx = 0.0, _ly = 0.0, _lz = 0.0;
float _ux = 0.0, _uy = 1.0, _uz = 0.0;

float lightRotFactor = 1.81;
