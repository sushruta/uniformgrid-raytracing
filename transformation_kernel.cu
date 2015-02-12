#ifndef TRANSFORMATION_KERNEL_H
#define TRANSFORMATION_KERNEL_H

__global__ void copy_data_transform(float *vertexlist, float *orig_list, int size, int offset, float rot_factor)
{
	int vert = threadIdx.x + blockIdx.x * blockDim.x;

	if (vert < size)
	{
		float x = ((orig_list[vert*3 + 0] - 12.0f) / 12.0f);
		float y = ((orig_list[vert*3 + 1] - 11.0f) / 12.0f);
		float z = ((orig_list[vert*3 + 2] - 4.5f) / 12.0f);

		vertexlist[(offset+vert)*3 + 0] = (x * cosf(rot_factor) - y * sinf(rot_factor)) * 9.0f + 14.5f;
		vertexlist[(offset+vert)*3 + 1] = (x * sinf(rot_factor) + y * cosf(rot_factor)) * 9.0f + 13.0f;
		vertexlist[(offset+vert)*3 + 2] = z * 9.0f + 4.0f;
	}
}

#endif
