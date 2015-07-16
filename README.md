# DTWfastNN
*********************************************************************************************************
*********************************************************************************************************
******************************** Dynamic Time Warping (DTW) *********************************************
********************* using Tree-based fast Nearest Neighbor (fastNN) ***********************************
******************************** Version: 0.4 ***********************************************************
*********************************************************************************************************
*********************************************************************************************************
**
** This code shows an example of performing Dynamic Time Warping using the tree-based fast Nearest Neighbor
** algorithm of Katsavounidis et al. [1].
** In case you use this code for research purposes, please cite [1].
**
** [1] I. Katsavounidis, C.-C.J. Kuo, and Zhen Zhang. Fast tree-structured nearest neighbor
** encoding for vector quantization. IEEE Transactions on Image Processing, 5 (2):398 - 404, 1996.
**
*********************************************************************************************************
*********************************************************************************************************
**
** This code was jointly written by Stergios Poularakis (stpoular@inf.uth.gr) and
** Prof. Ioannis Katsavounidis (ioannis.k@inf.uth.gr), VideoTeam, University of Thessaly, Greece.
** For further information please contact Prof. Ioannis Katsavounidis.
** Volos, November 2014
**
**
*********************************************************************************************************
*********************************************************************************************************
************************************** C API for DTW *************************************************
*********************************************************************************************************
** Step 1. DTW structure initialization for LB Keogh
** ** void create_L_U_signals_multi(MY_DOUBLE *q, int num_of_points, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U);
** ** void create_L_U_signals(MY_DOUBLE *q, int num_of_points, int curr_dim, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U);
** ** max_dim: dimension of each sequence data point (currently only max_dim=2 is supported)
** ** Computes L and U signals for query sequence q. This operation is useful when LB Keogh is used.
** ******************************************************************************************************
** Step 2.a. DTW search (full search - brute force)
** ** int dtw_search_full(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance);
** ** Returns the index corresponding to the DTW--NN of query_vector, as well as the cost matrix and the corresponding DTW distance (min_distance).
** ******************************************************************************************************
** Step 2.b. DTW search (full search - Sakoe Chiba band)
** ** int dtw_search_sakoe(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r)
** ** r: the Sakoe Chiba band parameter
** ** Returns the index corresponding to the DTW--NN of query_vector, as well as the cost matrix and the corresponding DTW distance (min_distance).
** ** Uses Sakoe Chiba band to reduce the cost of each DTW computation.
** ******************************************************************************************************
** Step 2.c. DTW search (partial search - Sakoe Chiba band - LB Keogh)
** ** int dtw_search_sakoe_LB_Keogh(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r, MY_DOUBLE *Ls, MY_DOUBLE *Us, struct paired *all_LBKeoghs);
** ** r: the Sakoe Chiba band parameter
** ** Returns the index corresponding to the DTW--NN of query_vector, as well as the cost matrix and the corresponding DTW distance (min_distance).
** ** Uses Sakoe Chiba band to reduce the cost of each DTW computation.
** ** Uses the LB Keogh lower bound to perform partial search.
** ******************************************************************************************************
**
**
*********************************************************************************************************
*********************************************************************************************************
************************************** C API for fastNN *************************************************
*********************************************************************************************************
** Step 1. FastNN structure initialization
** ** void NN_initialization_fastNN(MY_DOUBLE *clusters, int num_of_clusters, int dim, struct node **out_root, struct context **out_storage);
** ** clusters: matrix of training data, containing num_of_clusters row vectors of dimension dim
** ** out_root: root of the constructed binary tree
** ** out_storage: auxiliary memory storage
** ******************************************************************************************************
** Step 2.a. FastNN search (exact search)
** ** int nn_findNN_fastNN(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance);
** ** Assumes root and storage already created by NN_initialization_fastNN .
** ** flag_print_info can be set to 1 for debugging purposes.
** ** Returns the index corresponding to the NN of query_vector, as well as the corresponding Euclidean distance (min_distance).
** ******************************************************************************************************
** Step 2.b. FastNN search (Depth only search -- DOS, approximate search)
** ** int nn_findNN_fastNN_depth_only(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance);
** ** Performs a fastNN search, returning the training vector found in the first leaf node.
** ** Input/Output similar to exact search.
** ******************************************************************************************************
** Step 2.d. Full search (Standard brute-force algorithm, examining all training examples)
** ** int nn_findNN_full_search(MY_DOUBLE *query_vector, MY_DOUBLE *training_vectors, int num_of_training_vectors, int dim, double *min_distance);
** ** Performs a standard brute-force algorithm, examining all training examples. This version is useful
** ** for correctness verification and as a reference to estimate execution speedups.
** ******************************************************************************************************
** Step 3. Free all fastNN memory
** ** void NN_free_memory_fastNN(struct node **root, struct context **storage);
** ******************************************************************************************************
**
**
*********************************************************************************************************
*********************************************************************************************************
********************************** HOW TO CONTROL THE DEMO PROGRAM **************************************
*********************************************************************************************************
** The demo program (main.c) demonstrates a possible scenario of using the DTW API, focusing on 
** estimation of execution speedups over the standard Full Search algorithm (brute force) and an ideal
** initialization method, where the true NN is assumed as known in advance.
**
** Some demo options can be controlled by command line parameters, containing information about 
** training/testing data and profiling parameters. The usage scenario is as:
** DTWfastNN [training_dataset_name] [query_dataset_name] [LOOP_ITERATIONS] [TARGET_NUMBER_OF_EXAMPLES] [NUM_OF_EXPERIMENT_ITERATIONS]
** where LOOP_ITERATIONS denotes the number of DTW searches for each query sequence (to guarantee 
** time measurement stability), TARGET_NUMBER_OF_EXAMPLES is the number of training examples per user 
** per gesture and NUM_OF_EXPERIMENT_ITERATIONS is the number of experiment repetitions
** (to guarantee robustness of the time measurement process).
**
** The dataset parameters are controlled by the file dataset_info.info. The dataset filenames
** correspond to the pattern data_[UserID]_[GestureID]_[ExampleID]
**
** This program outputs a .csv file (DTW_fastNN_time_profile_[TARGET_NUMBER_OF_EXAMPLES].csv) showing the:
** 1. total number of additions (totalAdds)
** 2. total number of subtractions (totalSubs)
** 3. total number of multiplications (totalMuls)
** 4. total number of query searches performed (totalSearches)
** 5. total number of training examples in each search (numTrainingExamples)
** 6. total number of DTW computations (totalDTWcomputations)
** 7. total execution time of all DTW searches (timeDTW)
** 8. total execution time of fastNN initialization (timeInitFastNN)
** For example, after executing: digits6D_gestures digits6D_gestures 1 2 1 we get:
** 7620000 7680000 7680000 600 100 60000 60 10.0
** where we notice that the 7680000 multiplications correspond to 600 query searches and 100 training examples,
** i.e. 128 multiplications per computation. Since the gestures are of dimension 8x2,
** each DTW computation occupies 8x8=64 cells of 2 multiplications each, which corresponds
** to the case of the FullSearch DTW scheme.
** On the other hand, when running the same experiment for the fastNN initialization DTW scheme, 
** we get: 1119076 1132144 1155631 600 100 1324 10 0.0
** i.e. 19.26 multiplications per computation (which is a significant improvement).
**
** The main function of the demo is: void test_full_search(MY_DOUBLE *training_vectors, int *training_labels, int num_of_training_vectors, MY_DOUBLE *query_vectors, int *query_labels, int num_of_query_vectors, int dim, int num_of_categories, int r, int *confusion_matrix, double *total_search_time, int LOOP_ITERATIONS, MY_DOUBLE *Ls, MY_DOUBLE *Us, double *fastNN_initialization_time, __int64 *total_num_of_adds, __int64 *total_num_of_subs, __int64 *total_num_of_muls, __int64 *total_dtw_computations);
** in which one can choose the desired DTW search scheme (lines 138-158 in file dtw_full.c)
**
*********************************************************************************************************
*********************************************************************************************************
*********************************************************************************************************
*********************************************************************************************************
