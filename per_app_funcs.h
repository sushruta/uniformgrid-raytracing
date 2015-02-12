#ifndef PER_APP_H
#define PER_APP_H

#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>

void init_model(int argc, char **argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "USAGE: ./exec {file/dir_name}.obj [number_frames]\n");
		exit(-1);
	}

	int frames = 1;
	struct stat *buffer = (struct stat *) malloc(sizeof(struct stat));
	stat(argv[1], buffer);
	if (S_ISDIR(buffer->st_mode))
	{
		if (argc < 4)
		{
			fprintf(stderr, "USAGE: ./exec {dir_name}.ply [desc_file] [number_frames]\n");
			exit(-1);
		}
		else
		{
			frames = atoi(argv[3]);
		}
	}

	model = new Model(frames);
	model->some_material(argv[2]);
	model->load_model(argv[1]);

	return;
}

void writePPM(char *filename)
{
	FILE *fp;
	fp = fopen(filename,"w");
	if (!fp)
	{
		perror("fopen");
		exit(1);
	}

	float value_f;
	int value_i;

	fprintf(fp,"P3\n");
	fprintf(fp,"%d %d\n",SCREEN_WIDTH,SCREEN_HEIGHT);
	fprintf(fp,"%d\n",255);

	for (int i = 0; i < ( 3 * SCREEN_WIDTH * SCREEN_HEIGHT ) ; i++ )
	{
		if ( i%(3*SCREEN_WIDTH) == 0 )
			fprintf(fp,"\n");
		value_f = h_image[i];
		value_i = (int)value_f;
		fprintf(fp, "%d ",value_i);
	}
	fprintf(fp,"\n");
	fclose (fp);
}

#endif
