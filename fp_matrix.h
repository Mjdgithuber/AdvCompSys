#ifndef __FP__MATRIX__
#define __FP__MATRIX__

typedef unsigned char BOOL;

/* Returns a 2D square matrix with length and
 * width = size.  If randomize is asserted
 * the matrix will be initialized with random
 * data, otherwise it will remain uninitialized
 * (NOTE: aligned on 32 byte boundary) */
float** gen_fp_mat(size_t sz, BOOL randomize);

/* Randomizes each cell in 2D matrix. NOTE
 * this function returns the same pointer
 * passed in as the first argument*/
float** fp_mat_rand(float** mat, size_t sz);

/* Returns the resultant 2D matrix of the
 * multiplication of a and b both of size
 * sz */
float** mult_fp_mat(float** a, float** b, size_t sz);

/* Frees the 2D matrix allocation. NOTE: due
 * to aligned allocation, rows must not be
 * freed via free() */
void free_fp_mat(float** mat, size_t sz);

/* Prints floating point matrix with given
 * size to stdout */
void print_fp_mat(float** mat, size_t sz);

/* Generates a checksum by returning the sum
 * of all cells.  Intended to be used to ensure
 * correct output without having to print
 * the entire matrix */
float fp_mat_checksum();

#endif
