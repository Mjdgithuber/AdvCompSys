#include <stdio.h>
#include <stdlib.h>

/* Access to AVX C wrappers */
#include <immintrin.h>

#include "si_matrix.h"

/* Used to align memory on boundary for
 * AVX SIMD instructions */
#define BOUNDARY_ALIGNMENT 16

/* Used to maximize AVX register
 * usage */
#define AVX_REG_A_MAX 3
#define AVX_REG_B_MAX 4

short int** gen_si_mat(size_t sz, BOOL randomize) {
	size_t i;
	
	/* Many AVX instructions are only valid
         * for aligned data. _mm_malloc returns
         * aligned data.  NOTE: this cannot be
         * freed safely with free(), _mm_free()
         * must be used! */
	short int** ptr = malloc(sz * sizeof(short int*));
	for(i = 0; i < sz; i++)
		ptr[i] = _mm_malloc(sz * sizeof(short int), BOUNDARY_ALIGNMENT);

	return randomize ? si_mat_rand(ptr, sz) : ptr;
}

short int** si_mat_rand(short int** mat, size_t sz) {
	size_t i, j;
	
	/* Randomize each floating point cell
         * with random value [0,1] */
	for(i = 0; i < sz; i++) {
		for(j = 0; j < sz; j++)
			mat[i][j] = (short int)((rand() / (float) RAND_MAX) * (100));
	}

	return mat;
}

void free_si_mat(short int** mat, size_t sz) {
	size_t i;
	
	/* Note the use of _mm_free() instead of 
         * free() */
	for(i = 0; i < sz; i++)
		_mm_free(mat[i]);
	free(mat);
}

void print_si_mat(short int** mat, size_t sz) {
	size_t i, j;

	printf("Matrix:\n");
	for(i = 0; i < sz; i++) {
		for(j = 0; j < sz; j++)
			printf("%hd ", mat[i][j]);
		printf("\n");
	}
}

short int si_mat_checksum(short int** mat, size_t sz) {
	size_t i, j;
	short int ret = 0;

	for(i = 0; i < sz; i++) {
		for(j = 0; j < sz; j++)
			ret += mat[i][j];
	}

	return ret;
}

static void calc_block(short int** a, short int** b, short int** c, size_t a_block_size, size_t b_block_size, size_t a_block_index, size_t b_block_index, size_t size) {
	/* local simd scratch temps */
	__m128i a_r, b_r;

	/* row count and block index offsets */
	size_t row_count, a_bi, b_bi;

	/* local cache for faster storage while computing block */
	__m128i c_local[AVX_REG_A_MAX][AVX_REG_B_MAX] = {{{0.0}}};

	/* must go through entire matrix to calculate a block of c */
	for(row_count = 0; row_count < size; row_count++) {
		for(b_bi = 0; b_bi < b_block_size; b_bi++) {
			/* load 8 short ints (simd) from matrix b on this particular row and block offset.
                         * NOTE: this is going down matrix b */
			b_r = _mm_loadu_si128((__m128i*) (b[row_count] + 8*(b_bi + b_block_index * AVX_REG_B_MAX)));
			for(a_bi = 0; a_bi < a_block_size; a_bi++) {
				/* load simd reg by broadcasting a value from a. NOTE: this is going across
                                 * matrix a. */
				a_r = _mm_set1_epi16(a[a_bi + a_block_index*AVX_REG_A_MAX][row_count]);

				/* add to local cache at block offsets */
				c_local[a_bi][b_bi] = _mm_add_epi16(c_local[a_bi][b_bi], _mm_mullo_epi16(a_r, b_r));
			}
		}
	}

	/* store local sum into c */
	for(a_bi = 0; a_bi < a_block_size; a_bi++) {
		for(b_bi = 0; b_bi < b_block_size; b_bi++) {
			_mm_store_si128((__m128i*) (&c[a_bi + a_block_index*AVX_REG_A_MAX][(b_bi + b_block_index*AVX_REG_B_MAX)*8]), c_local[a_bi][b_bi]);
		}
	}
}

short int** mult_si_mat(short int** a, short int** b, size_t sz) {
	short int** c = gen_si_mat(sz, 0);
	
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
                         * the location of where the block starts in c */
			calc_block(a, b, c, a_block_size, b_block_size, bc_a, bc_b, sz);
		}
	}

	return c;
}

short int** mult_si_mat_naive(short int** a, short int** b, size_t sz) {
	short int** c = gen_si_mat(sz, 0);
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
