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
#include "nn_fastNN.h"


//#ifdef DTW_PROFILE
__int64 num_of_fastNN_adds = 0;
__int64 num_of_fastNN_subs = 0;
__int64 num_of_fastNN_muls = 0;
int num_of_backtrackings = 0;
//#endif



/*************************************************************************************************
**************************************************************************************************
************************ Front-end fastNN functions **********************************************
**************************************************************************************************
*************************************************************************************************/

void NN_initialization_fastNN(MY_DOUBLE *clusters, int num_of_clusters, int dim, struct node **out_root, struct context **out_storage)
{
	struct node *root = NULL;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_DOUBLE));
	int i;
	struct context *storage;

	storage = (struct context *) my_malloc(1*sizeof(struct context),ALIGNMENT);
	if(storage==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			1*sizeof(struct context), "storage");
		exit(-1);
	}
	storage->clusters2 = (MY_DOUBLE *) my_malloc(2*dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if(storage->clusters2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*dim2*sizeof(MY_DOUBLE), "storage->clusters2");
		exit(-1);
	}
	storage->indices2 = (int *) my_malloc(num_of_clusters*sizeof(int),ALIGNMENT);
	if(storage->indices2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*sizeof(MY_DOUBLE), "storage->indices2");
		exit(-1);
	}
	storage->cluster_count2 = (int *) my_malloc(2*sizeof(int),ALIGNMENT);
	if(storage->cluster_count2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*sizeof(int), "storage->cluster_count2");
		exit(-1);
	}
	storage->distances2 = (MY_DOUBLE *) my_malloc(num_of_clusters*sizeof(MY_DOUBLE),ALIGNMENT);
	if(storage->distances2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*sizeof(MY_DOUBLE), "storage->distances2");
		exit(-1);
	}
	storage->new_clusters2 = (double *) my_malloc(2*dim2*sizeof(double),ALIGNMENT);
	if(storage->new_clusters2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*dim2*sizeof(double), "new_clusters2");
		exit(-1);
	}

	root = allocate_node(num_of_clusters,dim);
	root->count = num_of_clusters;
	for(i=0;i<num_of_clusters;i++)
	{
		root->pairs[i].index = i;
	}

	tree_structure(root, clusters, dim, storage);

	*out_root = root;
	*out_storage = storage;
}




int nn_findNN_fastNN(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance)
{
	int max_ind = -1;
	MY_DOUBLE min_dist[2];
	double initial_distance = 0;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_DOUBLE));
	
#ifdef DTW_PROFILE
	num_of_fastNN_adds = 0;
	num_of_fastNN_subs = 0;
	num_of_fastNN_muls = 0;
#endif

	max_ind = fast_NN(query_vector, root, clusters, dim, min_dist);

	*min_distance = (double)(min_dist[0]); //squared
	
	return max_ind;
}



int nn_findNN_fastNN_limited(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance, int TH_BACKTRACKINGS)
{
	int max_ind = -1;
	MY_DOUBLE min_dist[2];
	double initial_distance = 0;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

#ifdef DTW_PROFILE
	num_of_fastNN_adds = 0;
	num_of_fastNN_subs = 0;
	num_of_fastNN_muls = 0;
	num_of_backtrackings = 0;
#endif

	max_ind = fast_NN_limited(query_vector, root, clusters, dim, min_dist, TH_BACKTRACKINGS);

	*min_distance = (double)(min_dist[0]); //squared

	return max_ind;
}



int nn_findNN_fastNN_depth_only(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance)
{
	int max_ind = -1;
	MY_DOUBLE min_dist[2];
	double initial_distance = 0;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_DOUBLE));
	
#ifdef DTW_PROFILE
	num_of_fastNN_adds = 0;
	num_of_fastNN_subs = 0;
	num_of_fastNN_muls = 0;
#endif

	max_ind = fast_NN_depth_only(query_vector, root, clusters, dim, min_dist);

	*min_distance = (double)(min_dist[0]); //squared
	
	return max_ind;
}


int nn_findNN_full_search(MY_DOUBLE *query_vector, MY_DOUBLE *training_vectors, int num_of_training_vectors, int dim, double *min_distance)
{
	//D is MxN, q is 1XN

	int i;
	double min_d = 0;
	int min_i = -1;
	double d = 0;
	MY_DOUBLE *v = training_vectors;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	min_d = distance2(v, query_vector, dim);
	min_i = 0;
	v += dim2;

	for (i = 1; i<num_of_training_vectors; i++)
	{
		d = distance2(v, query_vector, dim);

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

void NN_free_memory_fastNN(struct node **root, struct context **storage)
{
	free_node(*root);
	my_free((*storage)->new_clusters2);
	my_free((*storage)->distances2);
	my_free((*storage)->cluster_count2);
	my_free((*storage)->indices2);
	my_free((*storage)->clusters2);
	my_free(*storage);

	*root = NULL;
	*storage = NULL;
}





/*************************************************************************************************
**************************************************************************************************
************************ Core fastNN functions ***************************************************
**************************************************************************************************
*************************************************************************************************/




/*************************************************************************************************
************************ FastNN main functions for initialization and searching ******************
*************************************************************************************************/

void tree_structure(struct node *root, MY_DOUBLE *clusters, int dim, struct context *storage)
{
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	if (root->count>FAST_NN_THRESHOLD)
	{
		int i;
		int iter = 0;
		double ret_val;
		int mid_i;
		ret_val = kmeans_initialize2_map(storage, clusters, dim, root->count, root->pairs);
		kmeans_cluster2_map(storage, clusters, dim, root->count, root->pairs);
		//printf("TREE: iter=%3d, ret_val= %lf, diff=%7d, H=%lf\n",iter,ret_val,diff_count2,entropy(cluster_count2,2)/((double) dim));
		iter++;
		for (iter = 1; iter<100 && storage->diff_count2>0; iter++)
		{
			ret_val = kmeans_iterate2_map(storage, clusters, dim, root->count, root->pairs);
			//printf("TREE: iter=%3d, ret_val= %lf, diff=%7d, H=%lf\n",iter,ret_val,diff_count2,entropy(cluster_count2,2)/((double) dim));
		}
		root->c0 = calc_hyperplane(&storage->clusters2[0 * dim2], &storage->clusters2[1 * dim2], root->hyperplane, dim);
		for (i = 0; i<root->count; i++)
		{
			root->pairs[i].signed_distance = signed_distance(&clusters[root->pairs[i].index*dim2], root->hyperplane, root->c0, dim);
		}
		qsort((void *)root->pairs, root->count, sizeof(struct paired), increasing);
		mid_i = binary_search(root->pairs, root->count, 0.0);
		root->left = allocate_node(mid_i, dim);
		root->left->count = mid_i;
		for (i = 0; i<mid_i; i++)
		{
			root->left->pairs[i].index = root->pairs[i].index;
		}
		root->right = allocate_node(root->count - mid_i, dim);
		root->right->count = root->count - mid_i;
		for (i = mid_i; i<root->count; i++)
		{
			root->right->pairs[i - mid_i].index = root->pairs[i].index;
		}
		tree_structure(root->left, clusters, dim, storage);
		tree_structure(root->right, clusters, dim, storage);
	}
}


int full_search_NN(MY_DOUBLE *vector, MY_DOUBLE *clusters, int num_training_vectors, int dim, MY_DOUBLE *min_dist2)
{
	int i;
	int min_ind;
	MY_DOUBLE min_dist;
	MY_DOUBLE dist;

	min_dist = distance2(vector, &clusters[0], dim);
	min_ind = 0;

	for (i = 1; i < num_training_vectors; i++){
		dist = distance2(vector, &clusters[i], dim);

		if (dist < min_dist){
			min_dist = dist;
			min_ind = i;
		}
	}
	
	*min_dist2 = min_dist;
	return min_ind;
}


int fast_NN(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2)
{
	int i;
	int min_ind;
	MY_DOUBLE dist;
	MY_DOUBLE test_dist;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	if (root->count>FAST_NN_THRESHOLD)
	{
		double limit;
		struct paired *ptr;
		dist = signed_distance(vector, root->hyperplane, root->c0, dim);

		if (dist <= 0.0)
		{
			min_ind = fast_NN(vector, root->left, clusters, dim, min_dist2);

			limit = min_dist2[1] + dist;
			i = root->left->count;


			for (ptr = &root->pairs[i]; i<root->count && ptr->signed_distance<limit; i++, ptr++)
			{
				test_dist = my_distance2(vector, &clusters[ptr->index*dim2], dim, min_dist2[0]);

				if (test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE)sqrt(test_dist);
#ifdef DTW_PROFILE
					num_of_fastNN_muls++;
#endif
					limit = min_dist2[1] + dist;
					min_ind = ptr->index;
				}
			}
		}
		else
		{
			min_ind = fast_NN(vector, root->right, clusters, dim, min_dist2);

			limit = dist - min_dist2[1];
			i = root->left->count - 1;
			for (ptr = &root->pairs[i]; i >= 0 && ptr->signed_distance>limit; i--, ptr--)
			{

				test_dist = my_distance2(vector, &clusters[ptr->index*dim2], dim, min_dist2[0]);

				if (test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE)sqrt(test_dist);
					limit = dist - min_dist2[1];
					min_ind = ptr->index;
#ifdef DTW_PROFILE
					num_of_fastNN_subs++;
					num_of_fastNN_muls++;
#endif
				}
			}
		}
	}
	else
	{
		min_dist2[0] = distance2(vector, &clusters[root->pairs[0].index*dim2], dim);

		min_ind = root->pairs[0].index;
		if (root->count>1)
		{
			test_dist = my_distance2(vector, &clusters[root->pairs[1].index*dim2], dim, min_dist2[0]);

			if (test_dist<min_dist2[0])
			{
				min_ind = root->pairs[1].index;
				min_dist2[0] = test_dist;
			}
		}

		min_dist2[1] = (MY_DOUBLE)sqrt(min_dist2[0]);
#ifdef DTW_PROFILE
		num_of_fastNN_muls++;
#endif
	}
	return min_ind;
}




int fast_NN_depth_only(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2)
{
	int i;
	int min_ind;
	MY_DOUBLE dist;
	MY_DOUBLE test_dist;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	if (root->count>FAST_NN_THRESHOLD)
	{
		double limit;
		struct paired *ptr;
		dist = signed_distance(vector, root->hyperplane, root->c0, dim);

		if (dist <= 0.0)
		{
			min_ind = fast_NN_depth_only(vector, root->left, clusters, dim, min_dist2);

			return min_ind; // depth only returns here...

			limit = min_dist2[1] + dist;
			i = root->left->count;


			for (ptr = &root->pairs[i]; i<root->count && ptr->signed_distance<limit; i++, ptr++)
			{
				test_dist = my_distance2(vector, &clusters[ptr->index*dim2], dim, min_dist2[0]);

				if (test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE)sqrt(test_dist);
#ifdef DTW_PROFILE
					num_of_fastNN_muls++;
#endif
					limit = min_dist2[1] + dist;
					min_ind = ptr->index;
				}
			}
		}
		else
		{
			min_ind = fast_NN_depth_only(vector, root->right, clusters, dim, min_dist2);

			return min_ind; // depth only returns here...

			limit = dist - min_dist2[1];
			i = root->left->count - 1;
			for (ptr = &root->pairs[i]; i >= 0 && ptr->signed_distance>limit; i--, ptr--)
			{

				test_dist = my_distance2(vector, &clusters[ptr->index*dim2], dim, min_dist2[0]);

				if (test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE)sqrt(test_dist);
					limit = dist - min_dist2[1];
					min_ind = ptr->index;
#ifdef DTW_PROFILE
					num_of_fastNN_subs++;
					num_of_fastNN_muls++;
#endif
				}
			}
		}
	}
	else
	{
		min_dist2[0] = distance2(vector, &clusters[root->pairs[0].index*dim2], dim);

		min_ind = root->pairs[0].index;

		if (root->count>1)
		{
			test_dist = my_distance2(vector, &clusters[root->pairs[1].index*dim2], dim, min_dist2[0]);

			if (test_dist<min_dist2[0])
			{
				min_ind = root->pairs[1].index;
				min_dist2[0] = test_dist;
			}
		}

		min_dist2[1] = (MY_DOUBLE)sqrt(min_dist2[0]);
#ifdef DTW_PROFILE
		num_of_fastNN_muls++;
#endif
	}

	return min_ind;
}





int fast_NN_limited(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2, int TH_BACKTRACKINGS)
{
	int i;
	int min_ind;
	MY_DOUBLE dist;
	MY_DOUBLE test_dist;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	if (root->count>FAST_NN_THRESHOLD)
	{
		double limit;
		struct paired *ptr;
		dist = signed_distance(vector, root->hyperplane, root->c0, dim);

		if (dist <= 0.0)
		{
			min_ind = fast_NN_limited(vector, root->left, clusters, dim, min_dist2, TH_BACKTRACKINGS);
			if (num_of_backtrackings > TH_BACKTRACKINGS) return min_ind;

			limit = min_dist2[1] + dist;
			i = root->left->count;


			for (ptr = &root->pairs[i]; i<root->count && ptr->signed_distance<limit; i++, ptr++)
			{
				test_dist = my_distance2(vector, &clusters[ptr->index*dim2], dim, min_dist2[0]);

				if (test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE)sqrt(test_dist);
#ifdef DTW_PROFILE
					num_of_fastNN_muls++;
#endif
					limit = min_dist2[1] + dist;
					min_ind = ptr->index;
				}
				num_of_backtrackings++;
				if (num_of_backtrackings > TH_BACKTRACKINGS) return min_ind;
			}
		}
		else
		{
			min_ind = fast_NN_limited(vector, root->right, clusters, dim, min_dist2, TH_BACKTRACKINGS);
			if (num_of_backtrackings > TH_BACKTRACKINGS) return min_ind;

			limit = dist - min_dist2[1];
			i = root->left->count - 1;
			for (ptr = &root->pairs[i]; i >= 0 && ptr->signed_distance>limit; i--, ptr--)
			{

				test_dist = my_distance2(vector, &clusters[ptr->index*dim2], dim, min_dist2[0]);
				if (test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE)sqrt(test_dist);
					limit = dist - min_dist2[1];
					min_ind = ptr->index;
#ifdef DTW_PROFILE
					num_of_fastNN_subs++;
					num_of_fastNN_muls++;
#endif
				}

				num_of_backtrackings++;
				if (num_of_backtrackings > TH_BACKTRACKINGS) return min_ind;
			}
		}
	}
	else
	{
		min_dist2[0] = distance2(vector, &clusters[root->pairs[0].index*dim2], dim);
		min_ind = root->pairs[0].index;
		
		num_of_backtrackings++;
		if (num_of_backtrackings > TH_BACKTRACKINGS) return min_ind;

		if (root->count>1)
		{
			test_dist = my_distance2(vector, &clusters[root->pairs[1].index*dim2], dim, min_dist2[0]);

			if (test_dist<min_dist2[0])
			{
				min_ind = root->pairs[1].index;
				min_dist2[0] = test_dist;
			}

			num_of_backtrackings++;
			if (num_of_backtrackings > TH_BACKTRACKINGS) return min_ind;
		}

		min_dist2[1] = (MY_DOUBLE)sqrt(min_dist2[0]);
#ifdef DTW_PROFILE
		num_of_fastNN_muls++;
#endif
	}
	return min_ind;
}






/*************************************************************************************************
************************ FastNN main functions for initialization and searching ******************
*************************************************************************************************/

struct node *allocate_node(int num_of_clusters, int dim)
{
	struct node *root = NULL;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	if (dim>0 && num_of_clusters >= 1)
	{
		root = (struct node *) my_malloc(1 * sizeof(struct node), ALIGNMENT);
		if (root == NULL)
		{
			printf("Not enough memory (%d bytes) for %s - exiting\n",
				1 * sizeof(struct node), "root");
			exit(-1);
		}
		root->pairs = (struct paired *) my_malloc(num_of_clusters*sizeof(struct paired), ALIGNMENT);
		if (root->pairs == NULL)
		{
			printf("Not enough memory (%d bytes) for %s - exiting\n",
				num_of_clusters*sizeof(struct paired), "root->pairs");
			exit(-1);
		}
		if (num_of_clusters>FAST_NN_THRESHOLD)
		{
			root->hyperplane = (MY_DOUBLE *)my_malloc(dim2*sizeof(MY_DOUBLE), ALIGNMENT);
			if (root->hyperplane == NULL)
			{
				printf("Not enough memory (%d bytes) for %s - exiting\n",
					dim2*sizeof(MY_DOUBLE), "root->hyperplane");
				exit(-1);
			}
		}
		else
			root->hyperplane = NULL;
		root->left = root->right = NULL;
	}
	return root;
}

void free_node(struct node *root)
{
	if (root != NULL)
	{
		if (root->pairs != NULL)
			my_free(root->pairs);
		if (root->hyperplane != NULL)
			my_free(root->hyperplane);
		my_free(root);
	}
}

void free_tree(struct node *root)
{
	if (root != NULL)
	{
		free_tree(root->left);
		free_tree(root->right);
		free_node(root);
	}
}


// Returns L2-distance between two vectors (MY_DOUBLE[])
MY_DOUBLE calc_hyperplane_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, MY_DOUBLE *hyperplane, int dim)
{
	int i;
	MY_DOUBLE diff;
	double sum;
	double sum2;
	MY_DOUBLE val;

	sum = 0.0;
	sum2 = 0.0;
	for (i = 0; i<dim; i++)
	{
		diff = vector2[i] - vector1[i];
		hyperplane[i] = (MY_DOUBLE)diff;
		sum += diff*diff;
		sum2 += (vector1[i])*(vector1[i]) - (vector2[i])*(vector2[i]);
	}
	if (sum != 0.0)
	{
		double d;
		d = 1.0 / sqrt((double)sum);
		for (i = 0; i<dim; i++)
		{
			hyperplane[i] = (MY_DOUBLE)(hyperplane[i] * d);
		}
		val = (MY_DOUBLE)(0.5*d*((double)sum2));
	}
	else
	{
		val = 0.0;
	}
	return val;
}




/*************************************************************************************************
************************ Auxilliary functions for data generation ********************************
*************************************************************************************************/
void randomize(MY_DOUBLE *vectors, int dim, int num_of_vectors)
{
	int i;
	int j;
	MY_DOUBLE *p1;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	for (i = 0; i<num_of_vectors; i++)
	{
		p1 = &vectors[i*dim2];
		for (j = 0; j<dim; j++)
		{
			//p1[j] = (MY_DOUBLE) ((rand()&0x1FF)-256);
			p1[j] = gaussian(0.0, 25.0);
		}
		for (; j<dim2; j++)
		{
			p1[j] = (MY_DOUBLE)0;
		}
	}
}



MY_DOUBLE gaussian(MY_DOUBLE mean, MY_DOUBLE std_deviation)
{
	int i;
	int temp;
	int sum;
	MY_DOUBLE dtemp;

	sum = 0;
	for (i = 0; i<12; i++)
	{
		temp = rand(); // uniform distribution, [0..32767] => mean = 16383.5, variance = 32767^2/12
		sum += temp;
	}
	// sum: gaussian, mean = 196602, variance = 32767^2
	dtemp = (MY_DOUBLE)((sum - 196602.0) / 32767.0);

	return dtemp*std_deviation + mean;
}





/*************************************************************************************************
************************ Functions for distance computations *************************************
*************************************************************************************************/

// Returns L2-distance between two vectors (MY_DOUBLE[])
MY_DOUBLE signed_distance_c(MY_DOUBLE *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	double sum;

	sum = c0;
	for (i = 0; i<dim; i++)
	{
		sum += vector[i] * hyperplane[i];
#ifdef DTW_PROFILE
		num_of_fastNN_adds++;
		num_of_fastNN_muls++;
#endif
	}
	return (MY_DOUBLE)sum;
}

// Returns L2-distance between two vectors (MY_DOUBLE[])
MY_DOUBLE distance2_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;
	double sum;
	MY_DOUBLE diff;

	sum = 0.0;
	for (i = 0; i<dim; i++)
	{
		diff = vector1[i] - vector2[i];
		sum += diff*diff;
#ifdef DTW_PROFILE
		num_of_fastNN_adds++;
		num_of_fastNN_subs++;
		num_of_fastNN_muls++;
#endif
	}
	return (MY_DOUBLE)sum;
}

// Returns partial L2-distance between two vectors (MY_DOUBLE[]).
// When return value is > min_dist, it is not guaranteed to be the full distance
// since there could be early termination
MY_DOUBLE partial_distance2_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim, MY_DOUBLE min_dist)
{
	int i;
	double sum;
	MY_DOUBLE diff;

	sum = (double)min_dist;
	for (i = 0; i<dim && sum>0; i++)
	{
		diff = vector1[i] - vector2[i];
		sum -= diff*diff;

#ifdef DTW_PROFILE
		num_of_fastNN_subs += 2;
		num_of_fastNN_muls++;
#endif
	}

#ifdef DTW_PROFILE
	num_of_fastNN_subs++;
#endif

	return (MY_DOUBLE)(min_dist - sum);
}




/*************************************************************************************************
************************ Functions for vector operations******************************************
*************************************************************************************************/

MY_DOUBLE* my_normalize_vector(MY_DOUBLE *v, int N)
{
	int i;
	MY_DOUBLE d = (MY_DOUBLE)my_norm(v, N);

	for (i = 0; i<N; i++)
	{
		v[i] /= d;
	}

	return v;
}


MY_DOUBLE* my_normalize_matrix_rows(MY_DOUBLE *p, int M, int dim)
{
	MY_DOUBLE *tmp;
	int i, j;
	MY_DOUBLE d = 0;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	tmp = p;
	for (i = 0; i<M; i++)
	{
		d = (MY_DOUBLE)my_norm(tmp, dim);
		for (j = 0; j<dim; j++)
		{
			tmp[j] /= d;
		}
		tmp += dim;

		for (; j<dim2; j++)
		{
			tmp++;
		}
	}

	return p;
}


double my_norm(MY_DOUBLE *v, int N)
{
	double d = 0;
	int i;

	for (i = 0; i<N; i++)
	{
		d += v[i] * v[i];
	}

	return sqrt(d);
}

void accumulate_vector_c(double *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;

	for (i = 0; i<dim; i++)
	{
		vector1[i] += vector2[i];

#ifdef DTW_PROFILE
		num_of_fastNN_adds++;
#endif
	}
}

/*************************************************************************************************
************************ Functions for K-Means algorithm *****************************************
*************************************************************************************************/

double kmeans_initialize2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i;
	MY_DOUBLE *p1;
	MY_DOUBLE *p2;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	MY_DOUBLE max_dist;
	int max_ind;
	double total_sum;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	*diff_count = num_of_vectors;
	if (num_of_vectors<2)
	{
		return -1.0;
	}
	memset(new_clusters, 0, dim2*sizeof(double));
	memset(cluster_count, 0, 2 * sizeof(int));
	memset(indices, 0, num_of_vectors*sizeof(int));
	for (i = 0; i<num_of_vectors; i++)
	{
		p1 = &training[map[i].index*dim2];
		accumulate_vector(new_clusters, p1, dim);
	}
	cluster_count[0] = num_of_vectors;
	for (i = 0; i<dim; i++)
	{
		clusters[i] = (MY_DOUBLE)new_clusters[i] / (MY_DOUBLE)num_of_vectors;
	}
	for (; i<dim2; i++)
	{
		clusters[i] = (MY_DOUBLE)0;
	}
	// clusters[0] is the overall centroid
	max_dist = -1.0;
	max_ind = 0;
	for (i = 0; i<num_of_vectors; i++)
	{
		min_dist = distance2(&training[map[i].index*dim2], clusters, dim);
		if (min_dist>max_dist)
		{
			max_dist = min_dist;
			max_ind = i;
		}
	}
	p1 = &training[map[max_ind].index*dim2];
	for (i = 0; i<dim; i++)
	{
		clusters[i] = p1[i];
	}
	// clusters[0] is the first centroid
	max_dist = -1.0;
	for (i = 0; i<num_of_vectors; i++)
	{
		min_dist = distance2(&training[map[i].index*dim2], clusters, dim);
		distances[i] = min_dist;
		if (min_dist>max_dist)
		{
			max_dist = min_dist;
			max_ind = i;
		}
	}
	p2 = &clusters[dim2];
	p1 = &training[map[max_ind].index*dim2];
	for (i = 0; i<dim; i++)
	{
		p2[i] = p1[i];
	}
	for (; i<dim2; i++)
	{
		p2[i] = (MY_DOUBLE)0;
	}
	// p2=clusters[1] is the second centroid
	total_sum = 0.0;
	for (i = 0; i<num_of_vectors; i++)
	{
		p1 = &training[map[i].index*dim2];
		min_dist = distances[i];
		dist = my_distance2(p1, p2, dim, min_dist);
		if (dist<min_dist)
		{
			min_dist = dist;
			distances[i] = dist;
			indices[i] = 1;
		}
		total_sum += min_dist;
	}
	return ((double)total_sum) / ((double)num_of_vectors);
}

void kmeans_cluster2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i, j;
	MY_DOUBLE *p1;
	int min_ind;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	if (num_of_vectors<2)
	{
		return;
	}
	memset(new_clusters, 0, 2 * dim2*sizeof(double));
	memset(cluster_count, 0, 2 * sizeof(int));
	for (i = 0; i<num_of_vectors; i++)
	{
		min_ind = indices[i];
		accumulate_vector(&new_clusters[min_ind*dim2], &training[map[i].index*dim2], dim);
		cluster_count[min_ind]++;
	}
	for (j = 0; j<2; j++)
	{
		if ((min_ind = cluster_count[j])>0)
		{
			double *p2;
			p1 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for (i = 0; i<dim; i++)
			{
				p1[i] = (MY_DOUBLE)p2[i] / (MY_DOUBLE)min_ind;
			}
		}
	}
}

double kmeans_iterate2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i, j;
	MY_DOUBLE *p1;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	int min_ind;
	double total_sum;
	int dim2 = ALIGN(dim, ALIGNMENT / sizeof(MY_DOUBLE));

	if (num_of_vectors<2)
	{
		*diff_count = -1;
		return -1.0;
	}
	memset(new_clusters, 0, 2 * dim2*sizeof(double));
	memset(cluster_count, 0, 2 * sizeof(int));
	*diff_count = 0;
	total_sum = 0.0;
	for (i = 0; i<num_of_vectors; i++)
	{
		p1 = &training[map[i].index*dim2];
		min_ind = indices[i];
		min_dist = distance2(p1, &clusters[min_ind*dim2], dim);
		dist = my_distance2(p1, &clusters[(1 - min_ind)*dim2], dim, min_dist);
		if (dist<min_dist)
		{
			min_dist = dist;
			min_ind = 1 - min_ind;
			(*diff_count)++;
			indices[i] = min_ind;
		}
		accumulate_vector(&new_clusters[min_ind*dim2], p1, dim);
		cluster_count[min_ind]++;
		distances[i] = min_dist;
		total_sum += min_dist;
	}
	for (j = 0; j<2; j++)
	{
		if ((min_ind = cluster_count[j])>0)
		{
			double *p2;
			MY_DOUBLE *p3;
			p3 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for (i = 0; i<dim; i++)
			{
				p3[i] = (MY_DOUBLE)p2[i] / (MY_DOUBLE)min_ind;
			}
		}
	}
	return ((double)total_sum) / ((double)(num_of_vectors*dim));
}





/*************************************************************************************************
************************ Functions for binary search and sorting algorithm ***********************
*************************************************************************************************/

int binary_search(struct paired *pairs, int count, MY_DOUBLE mid_point)
{
	int min_i, max_i, mid_i;
	min_i = 0;
	max_i = count - 1;
	for (; max_i>min_i + 1;)
	{
		mid_i = (max_i + min_i + 1) / 2;
		if (pairs[mid_i].signed_distance <= mid_point)
			min_i = mid_i;
		else
			max_i = mid_i;
	}
	return max_i;
}


int increasing(const void *iptr1, const void *iptr2)
{
	struct paired *ptr1 = (struct paired *) iptr1;
	struct paired *ptr2 = (struct paired *) iptr2;

	if (ptr1->signed_distance<ptr2->signed_distance)
		return -1;
	else
		return 1;
}


