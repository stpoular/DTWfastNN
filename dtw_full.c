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
#include <time.h>
#include <math.h>

#include "dtw_full.h"
#include "classification_analysis.h"
#include "nn_fastNN.h"


__int64 num_of_adds = 0;
__int64 num_of_subs = 0;
__int64 num_of_muls = 0;
__int64 dtw_computations = 0;


/*************************************************************************************************
****************************** DTW  demonstration function ***************************************
*************************************************************************************************/
void test_full_search(MY_DOUBLE *training_vectors, int *training_labels, int num_of_training_vectors, MY_DOUBLE *query_vectors, int *query_labels, int num_of_query_vectors, int dim, int num_of_categories, int r, int *confusion_matrix, double *total_search_time, int LOOP_ITERATIONS, MY_DOUBLE *Ls, MY_DOUBLE *Us, double *fastNN_initialization_time, __int64 *total_num_of_adds, __int64 *total_num_of_subs, __int64 *total_num_of_muls, __int64 *total_dtw_computations, int DIM, int METHOD_CHOOSER_ID)
{
	int i;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_DOUBLE));
	int indx = 0;
	double min_distance = 0;
	int true_class = 0;
	int found_class = 0;
	double start_time = 0;
	double total_time = 0;
	int P = 0;
	double *Cost = NULL;
	int num_of_points = dim/DIM;

	__int64 prev_num_of_adds;
	__int64 prev_num_of_subs;
	__int64 prev_num_of_muls;
	__int64 prev_dtw_computations;

	__int64 prev_num_of_fastNN_adds = num_of_fastNN_adds;
	__int64 prev_num_of_fastNN_subs = num_of_fastNN_subs;
	__int64 prev_num_of_fastNN_muls = num_of_fastNN_muls;

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	struct node *root = NULL;
	struct context *storage = NULL;
	int initial_indx = 0;
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	struct paired *all_LBKeoghs = (struct paired*)malloc(num_of_training_vectors*sizeof(struct paired));
	for(i=0; i<num_of_training_vectors; i++)
	{
		all_LBKeoghs[i].index = i;
	}

	Cost = (double*)malloc(num_of_points*num_of_points*sizeof(double));
	init_Cost_matrix(Cost, num_of_points);

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	prev_num_of_adds = num_of_adds;
	prev_num_of_subs = num_of_subs;
	prev_num_of_muls = num_of_muls;
	prev_dtw_computations = dtw_computations;
	prev_num_of_fastNN_adds = num_of_fastNN_adds;
	prev_num_of_fastNN_subs = num_of_fastNN_subs;
	prev_num_of_fastNN_muls = num_of_fastNN_muls;


	/**fastNN_initialization_time = 0;
	start_time = clock();
	for (i = 0; i < 10; i++){
		NN_initialization_fastNN(training_vectors, num_of_training_vectors, dim, &root, &storage);
		NN_free_memory_fastNN(&root, &storage);
	}
	*fastNN_initialization_time += ((clock() - start_time));
	printf("fastNN_initialization_time = %.2f\n", (*fastNN_initialization_time)/10);
	*/

	start_time = clock();
	NN_initialization_fastNN(training_vectors, num_of_training_vectors, dim, &root, &storage);
	*fastNN_initialization_time += ((clock() - start_time));


	int TH_BACKTRACKINGS = (int)(0.25*num_of_training_vectors + 0.5);

	num_of_adds = prev_num_of_adds;
	num_of_subs = prev_num_of_subs;
	num_of_muls = prev_num_of_muls;
	dtw_computations = prev_dtw_computations;
	num_of_fastNN_adds = prev_num_of_fastNN_adds;
	num_of_fastNN_subs = prev_num_of_fastNN_subs;
	num_of_fastNN_muls = prev_num_of_fastNN_muls;
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	for(i=0; i<num_of_query_vectors; i++)
	{
		//printf("---------------- qi = %d -------------------------\n", i);
		prev_num_of_adds = num_of_adds;
		prev_num_of_subs = num_of_subs;
		prev_num_of_muls = num_of_muls;
		prev_dtw_computations = dtw_computations;
		initial_indx = dtw_search_sakoe(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, DIM);
		num_of_adds = prev_num_of_adds;
		num_of_subs = prev_num_of_subs;
		num_of_muls = prev_num_of_muls;
		dtw_computations = prev_dtw_computations;


		start_time = clock();
		for(P=0; P<LOOP_ITERATIONS; P++)
		{
			// Comment/Uncomment the following lines appropriately to switch between different DTW Search schemes

			//char *all_method_chooser_names[] = { "profile_dtw_full_examples", "profile_dtw_full_sakoe_examples", "profile_dtw_full_sakoe_LB_Keogh_precomputed_examples",
			//	"profile_dtw8_fastNN_init_DOS_examples", "profile_dtw8_fastNN_init_examples", "profile_dtw8_ideal_init_examples", "profile_dtw8_fastNN_limited_init_examples" };
			
			/*int indx0 = dtw_search_sakoe_LB_Keogh(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, all_LBKeoghs, DIM);
			initial_indx = nn_findNN_fastNN_depth_only(query_vectors + i*dim2, training_vectors, root, storage, dim, &min_distance);
			int indx1 = dtw_full_search_initial_estimate(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, initial_indx, all_LBKeoghs, DIM);
			int indx2 = dtw_search_sakoe(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, DIM);
			if (indx0 != indx1){
				printf("error...!");
			}
			if (indx0 != indx2){
				printf("error...!");
			}*/

			switch (METHOD_CHOOSER_ID){
			case 0:
				// DTW (full search)
				indx = dtw_search_full(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, DIM);
				break;
			case 1:
				// DTW Sakoe
				indx = dtw_search_sakoe(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, DIM);
				break;
			case 2:
				// DTW Sakoe and LB Keogh
				indx = dtw_search_sakoe_LB_Keogh(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, all_LBKeoghs, DIM);
				break;
			case 3:
				// DOS initialization
				initial_indx = nn_findNN_fastNN_depth_only(query_vectors + i*dim2, training_vectors, root, storage, dim, &min_distance);
				indx = dtw_full_search_initial_estimate(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, initial_indx, all_LBKeoghs, DIM);
				break;
			case 4:
				// fastNN initialization
				initial_indx = nn_findNN_fastNN(query_vectors + i*dim2, training_vectors, root, storage, dim, &min_distance);
				indx = dtw_full_search_initial_estimate(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, initial_indx, all_LBKeoghs, DIM);
				break;
			case 5:
				// Ideal initialization
				indx = dtw_full_search_initial_estimate(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, initial_indx, all_LBKeoghs, DIM);
				break;
			case 6:
				// fastNN_limited initialization
				initial_indx = nn_findNN_fastNN_limited(query_vectors + i*dim2, training_vectors, root, storage, dim, &min_distance, TH_BACKTRACKINGS);
				indx = dtw_full_search_initial_estimate(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, initial_indx, all_LBKeoghs, DIM);
				break;
			case 7:
				// fastNN only
				indx = nn_findNN_fastNN(query_vectors + i*dim2, training_vectors, root, storage, dim, &min_distance, TH_BACKTRACKINGS);
				break;
			case 8:
				// DOS only
				indx = nn_findNN_fastNN_depth_only(query_vectors + i*dim2, training_vectors, root, storage, dim, &min_distance);
				break;
			case 9:
				// fastNN limited only
				indx = nn_findNN_fastNN_limited(query_vectors + i*dim2, training_vectors, root, storage, dim, &min_distance, TH_BACKTRACKINGS);
				break;
			default:
				indx = dtw_full_search_initial_estimate(query_vectors + i*dim2, training_vectors, num_of_training_vectors, num_of_points, dim, Cost, &min_distance, r, Ls, Us, initial_indx, all_LBKeoghs, DIM);
				break;
			}			
		}
		total_time += ((clock() - start_time));

#ifdef DTW_PROFILE
		num_of_adds += num_of_fastNN_adds;
		num_of_subs += num_of_fastNN_subs;
		num_of_muls += num_of_fastNN_muls;
#endif

		true_class = query_labels[i]-1;
		found_class = training_labels[indx]-1;

		confusion_matrix[true_class*num_of_categories + found_class]++;
	}

	*total_search_time = total_time;

	free(Cost);

	//print_confusion_matrix(confusion_matrix, num_of_categories);

	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	NN_free_memory_fastNN(&root, &storage);
	///////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

	*total_num_of_adds = num_of_adds;
	*total_num_of_subs = num_of_subs;
	*total_num_of_muls = num_of_muls;
	*total_dtw_computations = dtw_computations;
}




/*************************************************************************************************
**************************************************************************************************
****************************** DTW search functions **********************************************
**************************************************************************************************
*************************************************************************************************/
int dtw_full_search_initial_estimate(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r, MY_DOUBLE *Ls, MY_DOUBLE *Us, int initial_indx, struct paired *all_LBKeoghs, int DIM)
{
	int i;
	double min_d = 0;
	int min_i = -1;
	double d = 0;
	MY_DOUBLE *v = D;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));
	double LB_Keogh = 0;

	MY_DOUBLE *L = Ls;
	MY_DOUBLE *U = Us;

	for (i = 0; i<M; i++)
	{
		all_LBKeoghs[i].index = i;
		all_LBKeoghs[i].signed_distance = (float)apply_LB_Keogh_multi(q, num_of_points, DIM, L, U);
		L += dim2;
		U += dim2;
		v += dim2;
	}

	v = D;

	min_d = compute_dtw_full_sakoe(D + initial_indx*dim2, q, num_of_points, num_of_points, Cost, r, DIM);
	min_i = initial_indx;

	for (i = 0; i<M; i++)
	{
		if (all_LBKeoghs[i].index == initial_indx) continue;

		LB_Keogh = all_LBKeoghs[i].signed_distance;

		if (LB_Keogh >= min_d)
		{
			continue;
		}

		d = compute_dtw_full_sakoe(D + all_LBKeoghs[i].index * dim2, q, num_of_points, num_of_points, Cost, r, DIM);

		if (d<min_d)
		{
			min_d = d;
			min_i = all_LBKeoghs[i].index;
		}
	}

	*min_distance = min_d;
	return min_i;
}

int dtw_search_full(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int DIM)
{
	//find row vector of D with max abs inner product with q
	//q and row vectors of D are assumed to have unit L2-norm.
	//D is MxN, q is 1XN

	int i;
	double min_d = 0;
	int min_i = -1;
	double d = 0;
	MY_DOUBLE *v = D;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	min_d = compute_dtw_full(v, q, num_of_points, num_of_points, Cost, DIM);
	min_i = 0;
	v += dim2;

	for (i = 1; i<M; i++)
	{
		d = compute_dtw_full(v, q, num_of_points, num_of_points, Cost, DIM);

		if (d<min_d)
		{
			min_d = d;
			min_i = i;
		}
		v += dim2;
	}

	*min_distance = min_d;

	return min_i;
}

int dtw_search_sakoe(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r, int DIM)
{
	int i;
	double min_d = 0;
	int min_i = -1;
	double d = 0;
	MY_DOUBLE *v = D;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));
	double LB_Keogh = 0;

	min_d = compute_dtw_full_sakoe(v, q, num_of_points, num_of_points, Cost, r, DIM);
	min_i = 0;
	v += dim2;

	for (i = 1; i<M; i++)
	{
		d = compute_dtw_full_sakoe(v, q, num_of_points, num_of_points, Cost, r, DIM);

		if (d<min_d)
		{
			min_d = d;
			min_i = i;
		}
		v += dim2;
	}

	*min_distance = min_d;
	return min_i;
}

int dtw_search_sakoe_LB_Keogh(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r, MY_DOUBLE *Ls, MY_DOUBLE *Us, struct paired *all_LBKeoghs, int DIM)
{
	int i;
	double min_d = 0;
	int min_i = -1;
	double d = 0;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));
	double LB_Keogh = 0;
	MY_DOUBLE *v = D;
	MY_DOUBLE *L = Ls;
	MY_DOUBLE *U = Us;

	for (i = 0; i<M; i++)
	{
		all_LBKeoghs[i].index = i;
		all_LBKeoghs[i].signed_distance = (float)apply_LB_Keogh_multi(q, num_of_points, DIM, L, U);
		L += dim2;
		U += dim2;
		v += dim2;
	}


	min_d = compute_dtw_full_sakoe(D + all_LBKeoghs[0].index * dim2, q, num_of_points, num_of_points, Cost, r, DIM);
	min_i = all_LBKeoghs[0].index;

	for (i = 1; i<M; i++)
	{
		LB_Keogh = all_LBKeoghs[i].signed_distance;

		if (LB_Keogh >= min_d)
		{
			continue;
		}

		d = compute_dtw_full_sakoe(D + all_LBKeoghs[i].index * dim2, q, num_of_points, num_of_points, Cost, r, DIM);

		if (d<min_d)
		{
			min_d = d;
			min_i = all_LBKeoghs[i].index;
		}
	}

	*min_distance = min_d;
	return min_i;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////




/*************************************************************************************************
**************************************************************************************************
****************************** DTW core functions ************************************************
**************************************************************************************************
*************************************************************************************************/

double compute_dtw_full_sakoe(MY_DOUBLE *X, MY_DOUBLE *Y, int m, int n, double *Cost, int r, int DIM)
{
	double distance = 0;
	int i, j, dij;
	double d1, d2, d3;
  
	Cost[0] = DTW_C(X, 0, Y, 0, DIM);
  
	for(i=1; i<m; i++)
	{ // first column
		if(i<=r)
		{
			Cost[i] = Cost[i - 1] + DTW_C(X, i, Y, 0, DIM);
#ifdef DTW_PROFILE
			num_of_adds += 1;
#endif
		}
		else
			Cost[i] = -1;
	}
  
	for(j=1; j<n; j++)
	{ // first row
		if(j<=r)
		{
			Cost[m*j] = Cost[m*(j - 1)] + DTW_C(Y, j, X, 0, DIM);
#ifdef DTW_PROFILE
			num_of_adds += 1;
#endif
		}
		else
			Cost[m*j] = -1;
	}
  
	// Main Loop
	for(i=1; i<m; i++)
	{
		for(j=1; j<n; j++)
		{
			dij = i - j;
			if(dij >= -r && dij <= r)
			{
				d1 = Cost[i-1 + (j-1)*m];
				d2 = Cost[i-1 + j*m];
				d3 = Cost[i + (j-1)*m];
				
				if(d2<0) d2 = d1 + 100;
				if(d3<0) d3 = d1 + 300;
			
				Cost[i + j*m] = DTW_C(X, i, Y, j, DIM) + my_min(d1, d2, d3);
#ifdef DTW_PROFILE
				num_of_adds += 1;
#endif
			}
			else
				Cost[i + j*m] = -1;
		}
	}

	distance = Cost[m*n -1];
	
#ifdef DTW_PROFILE
	dtw_computations++;
#endif

	return distance;
}

double compute_dtw_full(MY_DOUBLE *X, MY_DOUBLE *Y, int m, int n, double *Cost, int DIM)
{
	double distance = 0;
	int i, j;
	double d1, d2, d3;

	Cost[0] = DTW_C(X, 0, Y, 0, DIM);

	for (i = 1; i<m; i++)
	{ // first column
		Cost[i] = Cost[i - 1] + DTW_C(X, i, Y, 0, DIM);
#ifdef DTW_PROFILE
		num_of_adds += 1;
#endif
	}

	for (j = 1; j<n; j++)
	{ // first row
		Cost[m*j] = Cost[m*(j - 1)] + DTW_C(Y, j, X, 0, DIM);
#ifdef DTW_PROFILE
		num_of_adds += 1;
#endif
	}

	// Main Loop
	for (i = 1; i<m; i++)
	{
		for (j = 1; j<n; j++)
		{
			d1 = Cost[i - 1 + (j - 1)*m];
			d2 = Cost[i - 1 + j*m];
			d3 = Cost[i + (j - 1)*m];

			//d1 = val_at(Cost, i-1, j-1, m, n);
			//d2 = val_at(Cost, i-1, j, m, n);
			//d3 = val_at(Cost, i, j-1, m, n);

			Cost[i + j*m] = DTW_C(X, i, Y, j, DIM) + (double)(my_min(d1, d2, d3));
#ifdef DTW_PROFILE
			num_of_adds += 1;
#endif
		}
	}

	distance = Cost[m*n - 1];

#ifdef DTW_PROFILE
	dtw_computations++;
#endif

	return distance;
}

double apply_LB_Keogh_multi(MY_DOUBLE *v, int num_of_points, int max_dim, MY_DOUBLE *L, MY_DOUBLE *U)
{
	int curr_dim;
	int i;
	double LB_Keogh = 0;
	double vi = 0, ui = 0, li = 0;
	double di = 0;

	for (curr_dim = 0; curr_dim<max_dim; curr_dim++)
	{
		for (i = 0; i<num_of_points; i++)
		{
			vi = (double)(v[i*max_dim + curr_dim]);
			ui = (double)(U[i*max_dim + curr_dim]);
			li = (double)(L[i*max_dim + curr_dim]);

			if (vi>ui) di = vi - ui;
			else if (vi<li) di = vi - li;
			else di = 0;

			LB_Keogh += di * di;

#ifdef DTW_PROFILE
			num_of_adds += 1;
			num_of_subs += 1;
			num_of_muls += 1;
#endif
		}
	}

	return LB_Keogh;
}

double val_at(double p[], int i, int j, int m, int n)
{
	return p[i + j*m];
}

double DTW_C(MY_DOUBLE x[], int i, MY_DOUBLE y[], int j, int DIM)
{
	MY_DOUBLE dist0 = 0.0;
	double dist = 0.0;

	for (int d = 0; d < DIM; d++){
		dist0 = x[i*DIM + d] - y[j * DIM + d];
		dist += dist0*dist0;
	}

#ifdef DTW_PROFILE
	num_of_adds += DIM - 1;
	num_of_subs += DIM;
	num_of_muls += DIM;
#endif

	return dist;
}

double my_min(double a, double b, double c)
{
	if(a<=b && a<=c) return a;
	if(b<=a && b<=c) return b;
	return c;
}

void init_Cost_matrix(double *Cost, int n)
{
	int i, j;
	double *p = Cost;

	for(i=0; i<n; i++)
	{
		for(j=0; j<n; j++)
		{
			*p = 0;
			p++;
		}
	}
}





/*************************************************************************************************
************************ Auxilliary functions for DTW ********************************************
*************************************************************************************************/
void create_L_U_signals_multi(MY_DOUBLE *q, int num_of_points, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U)
{
	int curr_dim;

	for (curr_dim = 0; curr_dim<max_dim; curr_dim++)
	{
		create_L_U_signals(q, num_of_points, curr_dim, max_dim, r, L, U);
	}
}

void create_L_U_signals(MY_DOUBLE *q, int num_of_points, int curr_dim, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U)
{
	int i;
	int a, b;

	for (i = 0; i<num_of_points; i++)
	{
		a = i - r;
		b = i + r;

		if (a<0) a = 0;
		if (b>num_of_points - 1) b = num_of_points - 1;

		L[i*max_dim + curr_dim] = my_min_vector(q, num_of_points, curr_dim, max_dim, a, b);
		U[i*max_dim + curr_dim] = my_max_vector(q, num_of_points, curr_dim, max_dim, a, b);
	}
}


MY_DOUBLE my_max_vector(MY_DOUBLE *v, int N, int curr_dim, int max_dim, int a, int b)
{
	int i;
	MY_DOUBLE max_v = v[a*max_dim + curr_dim];

	for (i = a + 1; i <= b; i++)
	{
		if (v[i*max_dim + curr_dim]>max_v)
		{
			max_v = v[i*max_dim + curr_dim];
		}
	}

	return max_v;
}



MY_DOUBLE my_min_vector(MY_DOUBLE *v, int N, int curr_dim, int max_dim, int a, int b)
{
	int i;
	MY_DOUBLE min_v = v[a*max_dim + curr_dim];

	for (i = a + 1; i <= b; i++)
	{
		if (v[i*max_dim + curr_dim]<min_v)
		{
			min_v = v[i*max_dim + curr_dim];
		}
	}

	return min_v;
}
