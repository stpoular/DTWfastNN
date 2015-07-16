#ifndef _COMMON_H
#define _COMMON_H

#define USE_FAST_NN_FOR_INIT

#define MY_DOUBLE float

#define FAST_NN
#define ALIGNMENT 1
#define ALIGN(x,y) x
#define my_malloc(size,d) malloc(size)
#define my_free(ptr) free(ptr)

#endif