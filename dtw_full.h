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
#include "kmeans_lib3.h"

#ifndef _MCS_FULL_H
#define _MCS_FULL_H

extern __int64 num_of_adds;
extern __int64 num_of_subs;
extern __int64 num_of_muls;
extern __int64 dtw_computations;


/*************************************************************************************************
****************************** DTW  demonstration function ***************************************
*************************************************************************************************/
void test_full_search(MY_DOUBLE *training_vectors, int *training_labels, int num_of_training_vectors, MY_DOUBLE *query_vectors, int *query_labels, int num_of_query_vectors, int dim, int num_of_categories, int r, int *confusion_matrix, double *total_search_time, int LOOP_ITERATIONS, MY_DOUBLE *Ls, MY_DOUBLE *Us, double *fastNN_initialization_time, __int64 *total_num_of_adds, __int64 *total_num_of_subs, __int64 *total_num_of_muls, __int64 *total_dtw_computations);


/*************************************************************************************************
**************************************************************************************************
****************************** DTW search functions **********************************************
**************************************************************************************************
*************************************************************************************************/
int dtw_full_search_initial_estimate(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r, MY_DOUBLE *Ls, MY_DOUBLE *Us, int initial_indx, struct paired *all_LBKeoghs);
int dtw_search_full(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance);
int dtw_search_sakoe(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r);
int dtw_search_sakoe_LB_Keogh(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r, MY_DOUBLE *Ls, MY_DOUBLE *Us, struct paired *all_LBKeoghs);


/*************************************************************************************************
**************************************************************************************************
****************************** DTW core functions ************************************************
**************************************************************************************************
*************************************************************************************************/
double compute_dtw_full_sakoe(MY_DOUBLE *X, MY_DOUBLE *Y, int m, int n, double *Cost, int r);
double compute_dtw_full(MY_DOUBLE *X, MY_DOUBLE *Y, int m, int n, double *Cost);
double apply_LB_Keogh_multi(MY_DOUBLE *v, int num_of_points, int max_dim, MY_DOUBLE *L, MY_DOUBLE *U);
double val_at(double p[], int i, int j, int m, int n);
double DTW_C(MY_DOUBLE x[], int i, MY_DOUBLE y[], int j);
double my_min(double a, double b, double c);
void init_Cost_matrix(double *Cost, int n);


/*************************************************************************************************
************************ Auxilliary functions for DTW ********************************************
*************************************************************************************************/
void create_L_U_signals_multi(MY_DOUBLE *q, int num_of_points, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U);
void create_L_U_signals(MY_DOUBLE *q, int num_of_points, int curr_dim, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U);
MY_DOUBLE my_max_vector(MY_DOUBLE *v, int N, int curr_dim, int max_dim, int a, int b);
MY_DOUBLE my_min_vector(MY_DOUBLE *v, int N, int curr_dim, int max_dim, int a, int b);

#endif

