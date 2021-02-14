#include <stdio.h>
#include <stdlib.h>

/* Access to AVX C wrappers */
#include <immintrin.h>

#include "fp_matrix.h"

/* Used to align memory on boundary for
 * AVX SIMD instructions */
#define BOUNDARY_ALIGNMENT 32

/* Used to maximize AVX register
 * usage */
#define AVX_REG_A_MAX 3
#define AVX_REG_B_MAX 4

typedef float simd_f8 __attribute__ ((vector_size (32)));

float** gen_fp_mat(size_t sz, BOOL randomize) {
	size_t i;
	
	/* Many AVX instructions are only valid
         * for aligned data. _mm_malloc returns
         * aligned data.  NOTE: this cannot be
         * freed safely with free(), _mm_free()
         * must be used! */
	float** ptr = malloc(sz * sizeof(float*));
	for(i = 0; i < sz; i++)
		ptr[i] = _mm_malloc(sz * sizeof(float), BOUNDARY_ALIGNMENT);

	return randomize ? fp_mat_rand(ptr, sz) : ptr;
}

float** fp_mat_rand(float** mat, size_t sz) {
	size_t i, j;
	
	/* Randomize each floating point cell
         * with random value [0,1] */
	for(i = 0; i < sz; i++) {
		for(j = 0; j < sz; j++)
			mat[i][j] = (rand() / (float) RAND_MAX) * (1);
	}

	return mat;
}

void free_fp_mat(float** mat, size_t sz) {
	size_t i;
	
	/* Note the use of _mm_free() instead of 
         * free() */
	for(i = 0; i < sz; i++)
		_mm_free(mat[i]);
	free(mat);
}

void print_fp_mat(float** mat, size_t sz) {
	size_t i, j;

	printf("Matrix:\n");
	for(i = 0; i < sz; i++) {
		for(j = 0; j < sz; j++)
			printf("%f ", mat[i][j]);
		printf("\n");
	}
}

float fp_mat_checksum(float** mat, size_t sz) {
	size_t i, j;
	float ret = 0.f;

	for(i = 0; i < sz; i++) {
		for(j = 0; j < sz; j++)
			ret += mat[i][j];
	}

	return ret;
}

static void calc_block(float** a, float** b, float** c, size_t a_block_size, size_t b_block_size, size_t a_block_index, size_t b_block_index, size_t size) {
	/* local simd scratch temps */
	simd_f8 a_r, b_r;

	/* row count and block index offsets */
	size_t row_count, a_bi, b_bi;

	/* local cache for faster storage while computing block */
	simd_f8 c_local[AVX_REG_A_MAX][AVX_REG_B_MAX] = {{{0.0}}};

	/* must go through entire matrix to calculate a block of c */
	for(row_count = 0; row_count < size; row_count++) {
		for(b_bi = 0; b_bi < b_block_size; b_bi++) {
			/* load 8 floats (simd) from matrix b on this particular row and block offset.
                         * NOTE: this is going down matrix b */
			b_r = _mm256_load_ps(b[row_count] + 8*(b_bi + b_block_index * AVX_REG_B_MAX));

			for(a_bi = 0; a_bi < a_block_size; a_bi++) {
				/* load simd reg by broadcasting a value from a. NOTE: this is going across
                                 * matrix a. */
				a_r = _mm256_broadcast_ss(a[a_bi + a_block_index*AVX_REG_A_MAX] + row_count);

				/* add to local cache at block offsets */
				c_local[a_bi][b_bi] += _mm256_mul_ps(a_r, b_r);
			}
		}
	}

	/* store local sum into c */
	for(a_bi = 0; a_bi < a_block_size; a_bi++) {
		for(b_bi = 0; b_bi < b_block_size; b_bi++) {
			_mm256_store_ps(&c[a_bi + a_block_index*AVX_REG_A_MAX][(b_bi + b_block_index*AVX_REG_B_MAX)*8], c_local[a_bi][b_bi]);
		}
	}
}

float** mult_fp_mat(float** a, float** b, size_t sz) {
	float** c = gen_fp_mat(sz, 0);
	
	/* block count, max, remaining for a and b  matrices */
	size_t bc_a = 0, bc_b = 0;
	size_t bc_a_max = sz / (AVX_REG_A_MAX);
	size_t bc_b_max = sz / (AVX_REG_B_MAX * 8);
	size_t bc_a_r = sz % (AVX_REG_A_MAX);
	size_t bc_b_r = (sz % (AVX_REG_B_MAX * 8)) / 8;
	
	for(bc_a = 0; bc_a < bc_a_max + (bc_a_r ? 1 : 0); bc_a++) {
		for(bc_b = 0; bc_b < bc_b_max + (bc_b_r ? 1 : 0); bc_b++) {
			int b_block_size = bc_b < bc_b_max ? AVX_REG_B_MAX : bc_b_r;
			int a_block_size = bc_a < bc_a_max ? AVX_REG_A_MAX : bc_a_r;

			/* calculate one "block" of c at a time to maximize
                         * cache efficiency.  a_block_size and b_block_size
                         * are used to specify the size the "block" that is
                         * calculated both measured in simd width.  The size
                         * of the block calculated is a_block_size*b_block_size.
                         * a_block_index and b_block_index are used to specify
                         * the location of where the block starts in c
                         *
                         * Params:
                         * a - matrix a
                         * b - matrix b
                         * c - matrix c 
                         * a_block_size - */
			calc_block(a, b, c, a_block_size, b_block_size, bc_a, bc_b, sz);
		}
	}

	return c;
}

float** mult_fp_mat_naive(float** a, float** b, size_t sz) {
	float** c = gen_fp_mat(sz, 0);
	size_t i, j, k;

	/* Compute dot product */
	for(i = 0; i < sz; i++) {
		for(j = 0; j < sz; j++) {
			for(k = 0; k < sz; k++)
				c[i][j] += a[i][k] * b[k][j];
		}
	}
	return c;
}
