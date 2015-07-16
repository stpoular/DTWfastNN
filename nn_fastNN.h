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

#include "my_common.h"

#define DTW_PROFILE

#ifdef DTW_PROFILE
extern __int64 num_of_fastNN_adds;
extern __int64 num_of_fastNN_subs;
extern __int64 num_of_fastNN_muls;
#endif


#ifndef _NN_FASTNN_H
#define _NN_FASTNN_H


#define KKZ
#define FAST_NN_THRESHOLD 2

struct paired {
	int index;
	MY_DOUBLE signed_distance;
};

struct node
{
	struct node *left;
	struct node *right;
	MY_DOUBLE c0;
	MY_DOUBLE *hyperplane;
	int count;
	struct paired *pairs;
};

struct context {
	MY_DOUBLE *clusters2;
	int *indices2;
	int *cluster_count2;
	MY_DOUBLE *distances2;
	double *new_clusters2;
	int diff_count2;
};





#define calc_hyperplane(v1,v2,h,dim) calc_hyperplane_c(v1,v2,h,dim)
#define signed_distance(v,h,c0,dim) signed_distance_c(v,h,c0,dim)
#define distance2(v1,v2,dim) distance2_c(v1,v2,dim)
#define partial_distance2(v1,v2,dim,min_dist) partial_distance2_c(v1,v2,dim,min_dist)
#define my_distance2(v1,v2,dim,min_dist) partial_distance2(v1,v2,dim,min_dist)
#define accumulate_vector(v1,v2,dim) accumulate_vector_c(v1,v2,dim)



/*************************************************************************************************
**************************************************************************************************
************************ Front-end fastNN functions **********************************************
**************************************************************************************************
*************************************************************************************************/
void NN_initialization_fastNN(MY_DOUBLE *clusters, int num_of_clusters, int dim, struct node **out_root, struct context **out_storage);
int nn_findNN_fastNN(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance);
int nn_findNN_fastNN_depth_only(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance);
int nn_findNN_full_search(MY_DOUBLE *query_vector, MY_DOUBLE *training_vectors, int num_of_training_vectors, int dim, double *min_distance);
void NN_free_memory_fastNN(struct node **root, struct context **storage);




/*************************************************************************************************
**************************************************************************************************
************************ Core fastNN functions ***************************************************
**************************************************************************************************
*************************************************************************************************/

/*************************************************************************************************
************************ FastNN main functions for initialization and searching ******************
*************************************************************************************************/
void tree_structure(struct node *root, MY_DOUBLE *clusters, int dim, struct context *storage);
int fast_NN(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2);
int fast_NN_depth_only(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2);


/*************************************************************************************************
************************ Auxilliary functions for fastNN *****************************************
*************************************************************************************************/
struct node *allocate_node(int num_of_clusters, int dim);
void free_node(struct node *root);
MY_DOUBLE calc_hyperplane_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, MY_DOUBLE *hyperplane, int dim);


/*************************************************************************************************
************************ Auxilliary functions for data generation ********************************
*************************************************************************************************/
void randomize(MY_DOUBLE *vectors, int dim, int num_of_vectors);
MY_DOUBLE gaussian(MY_DOUBLE mean, MY_DOUBLE std_deviation);

/*************************************************************************************************
************************ Functions for distance computations *************************************
*************************************************************************************************/
MY_DOUBLE signed_distance_c(MY_DOUBLE *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim);
MY_DOUBLE distance2_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim);
MY_DOUBLE partial_distance2_c(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim, MY_DOUBLE min_dist);


/*************************************************************************************************
************************ Functions for vector operations******************************************
*************************************************************************************************/
MY_DOUBLE* my_normalize_vector(MY_DOUBLE *v, int N);
MY_DOUBLE* my_normalize_matrix_rows(MY_DOUBLE *p, int M, int dim);
double my_norm(MY_DOUBLE *v, int N);
void accumulate_vector_c(double *vector1, MY_DOUBLE *vector2, int dim);

/*************************************************************************************************
************************ Functions for K-Means algorithm *****************************************
*************************************************************************************************/
double kmeans_initialize2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map);
void kmeans_cluster2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map);
double kmeans_iterate2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map);


/*************************************************************************************************
************************ Functions for binary search and sorting algorithm ***********************
*************************************************************************************************/
int binary_search(struct paired *pairs, int count, MY_DOUBLE mid_point);
int increasing(const void *iptr1, const void *iptr2);


#endif
