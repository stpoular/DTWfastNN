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

#ifndef _IO_H
#define _IO_H

void read_info_from_file(char *filename, int *num_of_users, int *num_of_categories, int *num_of_examples, int *num_of_points);
void read_data_from_file(char *data_folder, int *persons, int num_of_persons,  int num_of_categories, int num_of_examples, int num_of_points, int dim, MY_DOUBLE *data_vectors, int *class_labels, int *total_num_of_vectors, int max_number_of_vectors);
void read_vector_from_file(char *filename, int dim, int dim2, MY_DOUBLE *v);


void prepare_tmp_training_persons_vector(int *tmp_training_persons, int num_of_users, int person_id);
char* find_program_name(char *argv0);


void initialize_confusion_matrix(int *confusion_matrix, int n);
void print_confusion_matrix(int *confusion_matrix, int n);
void fprint_confusion_matrix(int *confusion_matrix, int n, char *filename);





#endif