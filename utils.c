/*
**************************************************************************************************************
**************************************************************************************************************
******************************** Dynamic Time Warping(DTW) ***************************************************
********************* using Tree - based fast Nearest Neighbor(fastNN) ***************************************
******************************** Version: 0.4 ****************************************************************
**************************************************************************************************************
**************************************************************************************************************
**
** This code shows an example of performing Dynamic Time Warping using the tree - based fast Nearest Neighbor
** algorithm of Katsavounidis et al.[1]. In case you use this code for research purposes, please cite [1].
**
**[1] I.Katsavounidis, C. - C.J.Kuo, and Zhen Zhang.Fast tree - structured nearest neighbor
** encoding for vector quantization.IEEE Transactions on Image Processing, 5 (2) : 398 - 404, 1996.
**
**************************************************************************************************************
**************************************************************************************************************
**
** This code was jointly written by Stergios Poularakis(stpoular@inf.uth.gr) and
** Prof.Ioannis Katsavounidis(ioannis.k@inf.uth.gr), VideoTeam, University of Thessaly, Greece.
** For further information please contact Prof.Ioannis Katsavounidis.
** Volos, November 2014
**************************************************************************************************************
**************************************************************************************************************
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"


void read_info_from_file(char *filename, int *num_of_users, int *num_of_categories, int *num_of_examples, int *num_of_points)
{
	FILE *fd;

	fd = fopen(filename, "r");
	fscanf(fd, "%d ", num_of_users);
	fscanf(fd, "%d ", num_of_categories);
	fscanf(fd, "%d ", num_of_examples);
	fscanf(fd, "%d ", num_of_points);
	fclose(fd);
}



void read_data_from_file(char *data_folder, int *persons, int num_of_persons,  int num_of_categories, int num_of_examples, int num_of_points, int dim, MY_DOUBLE *data_vectors, int *class_labels, int *total_num_of_vectors, int max_number_of_vectors)
{
	int cat_id, person_id, example_id, person_i;
	char filename[301];
	int i = 0;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_DOUBLE));

	for(person_i=0; person_i<num_of_persons; person_i++)
	{
		person_id = persons[person_i];

		for(cat_id=1; cat_id<=num_of_categories; cat_id++)
		{
			//printf("%d: %d\n", person_id, cat_id);
			for(example_id=1; example_id<=num_of_examples; example_id++)
			{
				sprintf(filename, "%s/data_%d_%d_%d.txt", data_folder, person_id, cat_id, example_id);
				read_vector_from_file(filename, dim, dim2, data_vectors + i*dim2);
				
				class_labels[i] = cat_id;
				i = i + 1;

				if(i==max_number_of_vectors)
				{
					*total_num_of_vectors = i;
					return;
				}
			}
		}
	}

	*total_num_of_vectors = i;
}


void prepare_tmp_training_persons_vector(int *tmp_training_persons, int num_of_users, int person_id)
{
	int i, j=0;

	for(i=1; i<=num_of_users; i++)
	{
		if(i != person_id)
		{
			tmp_training_persons[j++] = i;
		}
	}
}



void read_vector_from_file(char *filename, int dim, int dim2, MY_DOUBLE *v)
{
	FILE *fd;
	int i = 0;
	MY_DOUBLE *tmp = v;

	fd = fopen(filename, "r");
	
	for(i=0; i<dim; i++)
	{
		fscanf(fd, "%f ", tmp++);
	}
	for(;i<dim2;i++)
	{
		*tmp = (MY_DOUBLE) 0;
		tmp++;
	}

	fclose(fd);
}




void initialize_confusion_matrix(int *confusion_matrix, int n)
{
	int i, j;
	int *p = confusion_matrix;

	for (i = 0; i<n; i++)
	{
		for (j = 0; j<n; j++)
		{
			*p = 0;
			p++;
		}
	}
}



void print_confusion_matrix(int *confusion_matrix, int n)
{
	int i, j;
	int *p = confusion_matrix;

	for (i = 0; i<n; i++)
	{
		for (j = 0; j<n; j++)
		{
			printf("%d ", *p);
			p++;
		}
		printf("\n");
	}
}



void fprint_confusion_matrix(int *confusion_matrix, int n, char *filename)
{
	int i, j;
	int *p = confusion_matrix;
	FILE *fd = fopen(filename, "w");

	for (i = 0; i<n; i++)
	{
		for (j = 0; j<n; j++)
		{
			fprintf(fd, "%d ", *p);
			p++;
		}
		fprintf(fd, "\n");
	}

	fclose(fd);
}


char* find_program_name(char *argv0)
{
	int i;
	char *curr = argv0;
	char *prev = curr;

	char *final_name = NULL;
	int n = 0;

	printf("%s\n", argv0);

	curr = strchr(curr, '\\');

	while (curr != NULL)
	{
		printf("curr = %s\n", curr);
		curr++;
		prev = curr;
		curr = strchr(curr, '\\');
	}
	printf("prev = %s\n", prev);

	n = strlen(prev) - 4 + 1;

	final_name = (char*)malloc(n * sizeof(char));

	for (i = 0; i<n - 1; i++)
	{
		final_name[i] = prev[i];
	}
	final_name[n - 1] = '\0';

	return final_name;
}





