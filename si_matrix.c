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

typedef float simd_si8 __attribute__ ((vector_size (16)));

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

static void print_num(__m128i num) {
	int i;

	for(i = 0; i < 8; i++)
		printf("%hd ", *( ((short int*)(&num)) + i) );
	printf("\n");
}

static void calc_block(short int** a, size_t a_block_size, size_t a_block_index, size_t b_block_index, size_t b_block_size, short int** b, short int** c, size_t size) {
	__m128i a_r, b_r;
	__m128i c_local[AVX_REG_A_MAX][AVX_REG_B_MAX] = {{{0.0}}};
	size_t row_count, a_bi, b_bi;

	
	int i;

	printf("Starting block (b block size: %u):\n", b_block_size);
	for(row_count = 0; row_count < size; row_count++) {
		for(b_bi = 0; b_bi < b_block_size; b_bi++) {
			b_r = _mm_loadu_si128((__m128i*) (b[row_count] + 8*(b_bi + b_block_index * AVX_REG_B_MAX)));
			//b_r = _mm256_load_ps(b[row_count] + 8*(b_bi + b_block_index * AVX_REG_B_MAX));
			//print_num(b_r);if(b_bi > 1) return;
			for(a_bi = 0; a_bi < a_block_size; a_bi++) {
				a_r = _mm_set1_epi16(a[a_bi + a_block_index*AVX_REG_A_MAX][row_count]);
				//a_r = _mm256_broadcast_ss(a[a_bi + a_block_index*AVX_REG_A_MAX] + row_count);
				c_local[a_bi][b_bi] += _mm_mullo_epi16(a_r, b_r);
				//c_local[a_bi][b_bi] += _mm256_mul_ps(a_r, b_r);
			}
		}
	}

	for(a_bi = 0; a_bi < a_block_size; a_bi++) {
		for(b_bi = 0; b_bi < b_block_size; b_bi++) {
			//for(i = 0; i < 8; i++)
				//printf("%hd\n", *(((short int*)(&c_local[a_bi][b_bi]))+i) );
			//printf("\n");
			_mm_store_si128((__m128i*) (&c[a_bi + a_block_index*AVX_REG_A_MAX][(b_bi + b_block_index*AVX_REG_B_MAX)*8]), c_local[a_bi][b_bi]);
			//_mm256_store_ps(&c[a_bi + a_block_index*AVX_REG_A_MAX][(b_bi + b_block_index*AVX_REG_B_MAX)*8], c_local[a_bi][b_bi]);
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
			calc_block(a, a_block_size, bc_a, bc_b, b_block_size, b, c, sz);
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
