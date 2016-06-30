## DTWfastNN: Dynamic Time Warping (DTW) using Tree-based fast Nearest Neighbor (fastNN)
This code shows an example of performing Dynamic Time Warping using the tree-based fast Nearest Neighbor algorithm of Katsavounidis et al. [1].  <br/>
[1] I. Katsavounidis, C.-C.J. Kuo, and Zhen Zhang. Fast tree-structured nearest neighbor encoding for vector quantization. 
IEEE Transactions on Image Processing, 5 (2):398 - 404, 1996.

This code was jointly written by Stergios Poularakis (stpoular@inf.uth.gr) and
Prof. Ioannis Katsavounidis (ioannis.k@inf.uth.gr), VideoTeam, University of Thessaly, Greece. <br\> <br/>
For further information please contact Prof. Ioannis Katsavounidis. <br\> <br/>
Volos, November 2014 <br\> <br\>


## How to control the demo program
The demo program (main.c) demonstrates a possible scenario of using the DTW API, focusing on estimation of execution speedups over 
the standard Full Search algorithm (brute force) and an ideal initialization method, where the true NN is assumed as known in advance.
<br />
Some demo options can be controlled by command line parameters, containing information about training/testing data and profiling parameters. 
<br /><br />
**The usage scenario is as:**<br />
DTWfastNN [training_dataset_name] [query_dataset_name] [LOOP_ITERATIONS] [TARGET_NUMBER_OF_EXAMPLES] [NUM_OF_EXPERIMENT_ITERATIONS] [DIM] [METHOD_CHOOSER_ID]
<br />
where LOOP_ITERATIONS denotes the number of DTW searches for each query sequence (to guarantee time measurement stability), 
TARGET_NUMBER_OF_EXAMPLES is the number of training examples per user per gesture and NUM_OF_EXPERIMENT_ITERATIONS is the number of experiment 
repetitions (to guarantee robustness of the time measurement process).
<br /> DIM is the dimensionality of each trajectory point.
<br />
METHOD_CHOOSER_ID selects the desired variation: <br />
0. dtw_full_examples <br />
1. dtw_full_sakoe_examples <br />
2. dtw_full_sakoe_LB_Keogh_precomputed_examples <br />
3. dtw8_fastNN_init_DOS_examples <br />
4. dtw8_fastNN_init_examples <br />
5. dtw8_ideal_init_examples <br />
6. dtw8_fastNN_limited_init_examples <br />
7. fastNN_examples <br />
8. fastNN_DOS_examples <br />
9. fastNN_limited_examples <br />
<br /><br />
The dataset parameters are controlled by the file dataset_info.info. The dataset filenames correspond to the pattern data_[UserID]_[GestureID]_[ExampleID]
<br /><br />
**This program outputs a .csv file (DTW_fastNN_time_profile_[TARGET_NUMBER_OF_EXAMPLES].csv) showing the:**<br />
1. total number of additions (totalAdds)<br />
2. total number of subtractions (totalSubs)<br />
3. total number of multiplications (totalMuls)<br />
4. total number of query searches performed (totalSearches)<br />
5. total number of training examples in each search (numTrainingExamples)<br />
6. total number of DTW computations (totalDTWcomputations)<br />
7. total execution time of all DTW searches (timeDTW)<br />
8. total execution time of fastNN initialization (timeInitFastNN)<br /><br />
For example, after executing: digits6D_gestures digits6D_gestures 1 2 1 we get:<br />
7620000 7680000 7680000 600 100 60000 60 10.0<br />
where we notice that the 7680000 multiplications correspond to 600 query searches and 100 training examples,
i.e. 128 multiplications per computation. Since the gestures are of dimension 8x2,
each DTW computation occupies 8x8=64 cells of 2 multiplications each, which corresponds
to the case of the FullSearch DTW scheme. <br /><br />
On the other hand, when running the same experiment for the fastNN initialization DTW scheme, 
we get: 1119076 1132144 1155631 600 100 1324 10 0.0
i.e. 19.26 multiplications per computation (which is a significant improvement).
<br /><br />
**The main function of the demo is:** <br />
void test_full_search(MY_DOUBLE *training_vectors, int *training_labels, int num_of_training_vectors, MY_DOUBLE *query_vectors, int *query_labels, int num_of_query_vectors, int dim, int num_of_categories, int r, int *confusion_matrix, double *total_search_time, int LOOP_ITERATIONS, MY_DOUBLE *Ls, MY_DOUBLE *Us, double *fastNN_initialization_time, __int64 *total_num_of_adds, __int64 *total_num_of_subs, __int64 *total_num_of_muls, __int64 *total_dtw_computations);
<br />
in which one can choose the desired DTW search scheme (lines 138-158 in file dtw_full.c) <br />
To run the demo, you will need to download some of the example datasets: <br />
https://www.dropbox.com/s/950gbypkcwdbgaw/digits6D_gestures8.zip?dl=0  <br />
https://www.dropbox.com/s/2umrehdn9x4gefd/lower6D_gestures8.zip?dl=0 <br />
https://www.dropbox.com/s/n0qs2f1dezf3uc5/upper6D_gestures8.zip?dl=0 <br />
https://www.dropbox.com/s/98sbxwpbooe17ht/character_trajectories16.zip?dl=0 <br />


## C API for DTW
- **Step 1. DTW structure initialization for LB Keogh** <br/>
- void create_L_U_signals_multi(MY_DOUBLE *q, int num_of_points, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U); <br/>
- void create_L_U_signals(MY_DOUBLE *q, int num_of_points, int curr_dim, int max_dim, int r, MY_DOUBLE *L, MY_DOUBLE *U); <br/>
- max_dim: dimension of each sequence data point (currently only max_dim=2 is supported) <br/>
- Computes L and U signals for query sequence q. This operation is useful when LB Keogh is used. <br/>
<br/>
**Step 2.a. DTW search (full search - brute force)** <br/>
- int dtw_search_full(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance); <br/>
- Returns the index corresponding to the DTW--NN of query_vector, as well as the cost matrix and the corresponding DTW distance (min_distance). <br/>
<br/>
**Step 2.b. DTW search (full search - Sakoe Chiba band)** <br/>
- int dtw_search_sakoe(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r) <br/>
- r: the Sakoe Chiba band parameter <br/>
- Returns the index corresponding to the DTW--NN of query_vector, as well as the cost matrix and the corresponding DTW distance (min_distance). <br/>
- Uses Sakoe Chiba band to reduce the cost of each DTW computation. <br/>
<br/>
**Step 2.c. DTW search (partial search - Sakoe Chiba band - LB Keogh)** <br/>
- int dtw_search_sakoe_LB_Keogh(MY_DOUBLE *q, MY_DOUBLE *D, int M, int num_of_points, int dim, double *Cost, double *min_distance, int r, MY_DOUBLE *Ls, MY_DOUBLE *Us, struct paired *all_LBKeoghs); <br/>
- r: the Sakoe Chiba band parameter <br/>
- Returns the index corresponding to the DTW--NN of query_vector, as well as the cost matrix and the corresponding DTW distance (min_distance). <br/>
- Uses Sakoe Chiba band to reduce the cost of each DTW computation. <br/>
- Uses the LB Keogh lower bound to perform partial search. <br/>


## C API for fastNN
- **Step 1. FastNN structure initialization** <br/>
- void NN_initialization_fastNN(MY_DOUBLE *clusters, int num_of_clusters, int dim, struct node **out_root, struct context **out_storage); <br/>
- clusters: matrix of training data, containing num_of_clusters row vectors of dimension dim <br/>
- out_root: root of the constructed binary tree <br/>
- out_storage: auxiliary memory storage <br/>
 <br/>
**Step 2.a. FastNN search (exact search)** <br/>
- int nn_findNN_fastNN(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance); <br/>
- Assumes root and storage already created by NN_initialization_fastNN . <br/>
- flag_print_info can be set to 1 for debugging purposes. <br/>
- Returns the index corresponding to the NN of query_vector, as well as the corresponding Euclidean distance (min_distance). <br/>
 <br/>
**Step 2.b. FastNN search (Depth only search -- DOS, approximate search)** <br/>
- int nn_findNN_fastNN_depth_only(MY_DOUBLE *query_vector, MY_DOUBLE *clusters, struct node *root, struct context *storage, int dim, double *min_distance); <br/>
- Performs a fastNN search, returning the training vector found in the first leaf node. <br/>
- Input/Output similar to exact search. <br/>
<br/>
**Step 2.d. Full search (Standard brute-force algorithm, examining all training examples)** <br/>
- int nn_findNN_full_search(MY_DOUBLE *query_vector, MY_DOUBLE *training_vectors, int num_of_training_vectors, int dim, double *min_distance); <br/>
- Performs a standard brute-force algorithm, examining all training examples. This version is useful
- for correctness verification and as a reference to estimate execution speedups. <br/>
 <br/>
**Step 3. Free all fastNN memory** <br/>
- void NN_free_memory_fastNN(struct node **root, struct context **storage); <br/>

## License
This code is provided "as is", please use it at your own risk. You can freely use this code for research purposes, citing [1]. <br/>
[1] S. Poularakis and I. Katsavounidis, "Initialization of dynamic time warping using tree-based fast Nearest Neighbor," Pattern Recognition Letters, Vol.79, pp.31–37, 2016. <br/>
Link: http://www.sciencedirect.com/science/article/pii/S016786551630068X  <br />

Other relevant publications: <br />
[2] S. Poularakis and I. Katsavounidis, "Low–complexity hand gesture recognition system for continuous streams of digits and letters," IEEE Transactions on Cybernetics PP, 1–1, 2015. <br />
[3] I. Katsavounidis, C.-C.J. Kuo, and Zhen Zhang, "Fast tree-structured nearest neighbor encoding for vector quantization," IEEE Transactions on Image Processing, 5 (2):398 - 404, 1996. <br />

