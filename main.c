#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"
#include "dtw_full.h"
#include "nn_fastNN.h"

int main(int argc, char *argv[])
{
	char filename[301];
	char training_folder[101];
	char query_folder[101];
	int dim, dim2;
	int person_id;
	MY_DOUBLE *training_vectors = NULL;
	int *training_labels = NULL;
	MY_DOUBLE *query_vectors = NULL;
	int *query_labels = NULL;
	
	char *program_name = NULL;
	int *tmp_training_persons = NULL;
	int *confusion_matrix = NULL;

	int num_of_points = 0;
	int num_of_categories = 0;

	int training_num_of_users = 0;
	int training_num_of_categories = 0;
	int training_num_of_examples = 0;
	int training_num_of_points = 0;

	int query_num_of_users = 0;
	int query_num_of_categories = 0;
	int query_num_of_examples = 0;
	int query_num_of_points = 0;


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

	if (argc < 8)
	{
		printf("Usage: DTWfastNN [training_dataset_name] [query_dataset_name] [LOOP_ITERATIONS] [TARGET_NUMBER_OF_EXAMPLES] [NUM_OF_EXPERIMENT_ITERATIONS] [DIM] [METHOD_CHOOSER_ID]\n");

		system("pause");
		exit(-1);
	}

	LOOP_ITERATIONS = atoi(argv[3]);
	TARGET_NUMBER_OF_EXAMPLES = atoi(argv[4]);
	NUM_OF_EXPERIMENT_ITERATIONS = atoi(argv[5]);

	int DIM = atoi(argv[6]);
	int METHOD_CHOOSER_ID = atoi(argv[7]);
	char METHOD_CHOOSER_NAME[101];
#ifdef DTW_PROFILE
	char *all_method_chooser_names[] = { "profile_dtw_full_examples", "profile_dtw_full_sakoe_examples", "profile_dtw_full_sakoe_LB_Keogh_precomputed_examples", 
		"profile_dtw8_fastNN_init_DOS_examples", "profile_dtw8_fastNN_init_examples", "profile_dtw8_ideal_init_examples", "profile_dtw8_fastNN_limited_init_examples", 
		"profile_fastNN_examples", "profile_fastNN_DOS_examples", "profile_fastNN_limited_examples"};
#else
	char *all_method_chooser_names[] = { "dtw_full_examples", "dtw_full_sakoe_examples", "dtw_full_sakoe_LB_Keogh_precomputed_examples",
		"dtw8_fastNN_init_DOS_examples", "dtw8_fastNN_init_examples", "dtw8_ideal_init_examples", "dtw8_fastNN_limited_init_examples",
		"fastNN_examples", "fastNN_DOS_examples", "fastNN_limited_examples"};
#endif

	//program_name = find_program_name(argv[0]);
	program_name = all_method_chooser_names[METHOD_CHOOSER_ID];
	sprintf(output_file, "%s_time_%d.txt", program_name, TARGET_NUMBER_OF_EXAMPLES);
	fd = fopen(output_file, "w");
	//fprintf(fd, "totalAdds, totalSubs, totalMuls, totalSearches, numTrainingExamples, totalDTWcomputations, timeDTW, timeInitFastNN\n");
	fclose(fd);


	sprintf(training_folder, "%s", argv[1]);
	sprintf(query_folder, "%s", argv[2]);

	sprintf(filename,"%s/dataset_info.info", training_folder);
	read_info_from_file(filename, &training_num_of_users, &training_num_of_categories, &training_num_of_examples, &training_num_of_points);
	
	sprintf(filename, "%s/dataset_info.info", query_folder);
	read_info_from_file(filename, &query_num_of_users, &query_num_of_categories, &query_num_of_examples, &query_num_of_points);

	if (training_num_of_points != query_num_of_points){
		printf("ERROR : training_num_of_points != query_num_of_points ....\n");
		printf("Program ends here...\n");
		exit(-1);
	}
	num_of_points = training_num_of_points;

	if (training_num_of_categories > query_num_of_categories){
		num_of_categories = training_num_of_categories;
	}
	else{
		num_of_categories = query_num_of_categories;
	}

	dim = DIM * num_of_points;
	dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	num_of_training_examples = TARGET_NUMBER_OF_EXAMPLES;

	dim = DIM * training_num_of_points;
	dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_DOUBLE));

	training_vectors = (MY_DOUBLE*)my_malloc((training_num_of_users - 1)*training_num_of_categories*training_num_of_examples*dim2*sizeof(MY_DOUBLE), ALIGNMENT);
	training_labels = (int*)malloc((training_num_of_users - 1)*training_num_of_categories*training_num_of_examples*dim*sizeof(int));
	query_vectors = (MY_DOUBLE*)my_malloc(1 * query_num_of_categories*query_num_of_examples*dim2*sizeof(MY_DOUBLE), ALIGNMENT);
	query_labels = (int*)malloc(1 * query_num_of_categories*query_num_of_examples*dim*sizeof(int));
	tmp_training_persons = (int*)malloc((training_num_of_users-1)*sizeof(int));
	confusion_matrix = (int*)malloc(num_of_categories*num_of_categories*sizeof(int));

	Ls = (MY_DOUBLE*)malloc((training_num_of_users - 1)*num_of_categories*training_num_of_examples * DIM * num_of_points*sizeof(MY_DOUBLE));
	Us = (MY_DOUBLE*)malloc((training_num_of_users - 1)*num_of_categories*training_num_of_examples * DIM * num_of_points*sizeof(MY_DOUBLE));

	for(iter=0; iter<NUM_OF_EXPERIMENT_ITERATIONS; iter++)
	{
		printf("iter = %d ------ \n", iter);
		total_time_full = 0;
		fastNN_initialization_time = 0;

		initialize_confusion_matrix(confusion_matrix, num_of_categories);
		for(person_id=1; person_id<=query_num_of_users; person_id++)
		{
			//printf("person_id = %d ------ \n", person_id);

			if (person_id > training_num_of_users)
				prepare_tmp_training_persons_vector(tmp_training_persons, training_num_of_users, 1);
			else
				prepare_tmp_training_persons_vector(tmp_training_persons, training_num_of_users, person_id);

			read_data_from_file(training_folder, tmp_training_persons, training_num_of_users-1,  training_num_of_categories, num_of_training_examples, num_of_points, dim, training_vectors, training_labels, &num_of_training_vectors,9999);
			read_data_from_file(query_folder, &person_id, 1,  query_num_of_categories, query_num_of_examples, num_of_points, dim, query_vectors, query_labels, &num_of_query_vectors,10000);

			///////////////////////////////////////////////////////////////////////////////////////
			// Normalize data to unit vectors
			training_vectors = my_normalize_matrix_rows(training_vectors, num_of_training_vectors, dim);
			query_vectors = my_normalize_matrix_rows(query_vectors, num_of_query_vectors, dim);
			///////////////////////////////////////////////////////////////////////////////////////

			tmp_start_time = clock();
			L = Ls;
			U = Us;
			for(i=0; i<num_of_training_vectors; i++)
			{
				create_L_U_signals_multi(training_vectors + i*dim2, num_of_points, DIM, SAKORE_R, L, U);

				L += dim2;
				U += dim2;
			}
			total_time_initialization += ((clock() - tmp_start_time));

			test_full_search(training_vectors, training_labels, num_of_training_vectors, query_vectors, query_labels, num_of_query_vectors, dim, num_of_categories, SAKORE_R, confusion_matrix, &tmp_time, LOOP_ITERATIONS, Ls, Us, &fastNN_initialization_time, &total_num_of_adds, &total_num_of_subs, &total_num_of_muls, &total_dtw_computations, DIM, METHOD_CHOOSER_ID);
			total_time_full += tmp_time;
		}

		total_time_full /= CLOCKS_PER_SEC;
		fastNN_initialization_time  /= CLOCKS_PER_SEC;
		total_num_of_query_searches_performed = query_num_of_users * num_of_query_vectors * LOOP_ITERATIONS;

		fd = fopen(output_file, "a");
		fprintf(fd, "%I64d %I64d %I64d %I64d %d %I64d %.0lf %.1lf\n", total_num_of_adds, total_num_of_subs, total_num_of_muls, total_num_of_query_searches_performed, num_of_training_vectors, total_dtw_computations, 1000*total_time_full, 1000*fastNN_initialization_time);
		fclose(fd);
	}

	print_confusion_matrix(confusion_matrix, num_of_categories);
	
	sprintf(output_file, "%s_conf_matrix_%d.txt", program_name, TARGET_NUMBER_OF_EXAMPLES);
	fprint_confusion_matrix(confusion_matrix, num_of_categories, output_file);
	//free(program_name);


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


