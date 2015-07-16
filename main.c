#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"
#include "dtw_full.h"



int main(int argc, char *argv[])
{
	char filename[301];
	char training_folder[101];
	char query_folder[101];
	int num_of_users, num_of_categories, num_of_examples, num_of_points, dim, dim2;
	int person_id;
	MY_DOUBLE *training_vectors = NULL;
	int *training_labels = NULL;
	MY_DOUBLE *query_vectors = NULL;
	int *query_labels = NULL;
	
	char *program_name = NULL;
	int *tmp_training_persons = NULL;
	int *confusion_matrix = NULL;

	int num_of_training_vectors = 0;
	int num_of_query_vectors = 0;
	double tmp_time = 0;
	double total_time_full = 0;
	int iter = 0;
	FILE *fd = NULL;
	char output_file[301];
	int i;

	int num_of_training_examples = 0;
	double fastNN_initialization_time = 0;
	
	double tmp_start_time = 0;
	double total_time_initialization = 0;

	MY_DOUBLE *Ls = NULL;
	MY_DOUBLE *Us = NULL;
	MY_DOUBLE *L;
	MY_DOUBLE *U;

	__int64 total_num_of_adds = 0;
	__int64 total_num_of_subs = 0;
	__int64 total_num_of_muls = 0;
	__int64 total_dtw_computations = 0;
	__int64 total_num_of_query_searches_performed = 0;

	int LOOP_ITERATIONS;
	int TARGET_NUMBER_OF_EXAMPLES;
	int NUM_OF_EXPERIMENT_ITERATIONS;

	int SAKORE_R = 2;

	if (argc < 6)
	{
		printf("Usage: DTWfastNN [training_dataset_name] [query_dataset_name] [LOOP_ITERATIONS] [TARGET_NUMBER_OF_EXAMPLES] [NUM_OF_EXPERIMENT_ITERATIONS]\n");

		system("pause");
		exit(-1);
	}

	LOOP_ITERATIONS = atoi(argv[3]);
	TARGET_NUMBER_OF_EXAMPLES = atoi(argv[4]);
	NUM_OF_EXPERIMENT_ITERATIONS = atoi(argv[5]);

	program_name = find_program_name(argv[0]);
	sprintf(output_file, "%s_time_profile_%d.csv", program_name, TARGET_NUMBER_OF_EXAMPLES);
	fd = fopen(output_file, "w");
	fprintf(fd, "totalAdds, totalSubs, totalMuls, totalSearches, numTrainingExamples, totalDTWcomputations, timeDTW, timeInitFastNN\n");
	fclose(fd);


	sprintf(training_folder, "%s", argv[1]);
	sprintf(query_folder, "%s", argv[2]);

	sprintf(filename,"%s/dataset_info.info", training_folder);

	read_info_from_file(filename, &num_of_users, &num_of_categories, &num_of_examples, &num_of_points);
	
	num_of_training_examples = TARGET_NUMBER_OF_EXAMPLES;

	dim = 2*num_of_points;
	dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_DOUBLE));

	training_vectors = (MY_DOUBLE*)my_malloc((num_of_users-1)*num_of_categories*num_of_examples*dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	training_labels = (int*)malloc((num_of_users-1)*num_of_categories*num_of_examples*dim*sizeof(int));
	query_vectors = (MY_DOUBLE*)my_malloc(1*num_of_categories*num_of_examples*dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	query_labels = (int*)malloc(1*num_of_categories*num_of_examples*dim*sizeof(int));
	tmp_training_persons = (int*)malloc((num_of_users-1)*sizeof(int));
	confusion_matrix = (int*)malloc(num_of_categories*num_of_categories*sizeof(int));

	Ls = (MY_DOUBLE*)malloc((num_of_users-1)*num_of_categories*num_of_examples*2*num_of_points*sizeof(MY_DOUBLE));
	Us = (MY_DOUBLE*)malloc((num_of_users-1)*num_of_categories*num_of_examples*2*num_of_points*sizeof(MY_DOUBLE));

	for(iter=0; iter<NUM_OF_EXPERIMENT_ITERATIONS; iter++)
	{
		printf("iter = %d ------ \n", iter);
		total_time_full = 0;
		fastNN_initialization_time = 0;

		initialize_confusion_matrix(confusion_matrix, num_of_categories);
		for(person_id=1; person_id<=num_of_users; person_id++)
		{
			//printf("person_id = %d ------ \n", person_id);
			prepare_tmp_training_persons_vector(tmp_training_persons, num_of_users, person_id);
			read_data_from_file(training_folder, tmp_training_persons, num_of_users-1,  num_of_categories, num_of_training_examples, num_of_points, dim, training_vectors, training_labels, &num_of_training_vectors,9999);
			read_data_from_file(query_folder, &person_id, 1,  num_of_categories, num_of_examples, num_of_points, dim, query_vectors, query_labels, &num_of_query_vectors,10000);

			tmp_start_time = clock();
			L = Ls;
			U = Us;
			for(i=0; i<num_of_training_vectors; i++)
			{
				create_L_U_signals_multi(training_vectors + i*dim2, num_of_points, 2, SAKORE_R, L, U);

				L += dim2;
				U += dim2;
			}
			total_time_initialization += ((clock() - tmp_start_time));

			test_full_search(training_vectors, training_labels, num_of_training_vectors, query_vectors, query_labels, num_of_query_vectors, dim, num_of_categories, SAKORE_R, confusion_matrix, &tmp_time, LOOP_ITERATIONS, Ls, Us, &fastNN_initialization_time, &total_num_of_adds, &total_num_of_subs, &total_num_of_muls, &total_dtw_computations);
			total_time_full += tmp_time;
		}

		total_time_full /= CLOCKS_PER_SEC;
		fastNN_initialization_time  /= CLOCKS_PER_SEC;
		total_num_of_query_searches_performed = num_of_users * num_of_query_vectors * LOOP_ITERATIONS;

		fd = fopen(output_file, "a");
		fprintf(fd, "%I64d %I64d %I64d %I64d %d %I64d %.0lf %.1lf\n", total_num_of_adds, total_num_of_subs, total_num_of_muls, total_num_of_query_searches_performed, num_of_training_vectors, total_dtw_computations, 1000*total_time_full, 1000*fastNN_initialization_time);
		fclose(fd);
	}

	print_confusion_matrix(confusion_matrix, num_of_categories);
	
	sprintf(output_file, "%s_conf_matrix_%d.txt", program_name, TARGET_NUMBER_OF_EXAMPLES);
	fprint_confusion_matrix(confusion_matrix, num_of_categories, output_file);
	free(program_name);


	// free all memory
	free(Ls);
	free(Us);
	free(tmp_training_persons);
	my_free(training_vectors);
	my_free(query_vectors);
	free(training_labels);
	free(query_labels);
	free(confusion_matrix);

	return 0;
}


