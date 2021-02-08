/*#include <x86intrin.h>*/
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#define L_SF8(ADDR) *((const simd_f8 *)(ADDR))

typedef int simd_f __attribute__ ((vector_size (16)));

typedef float simd_f8 __attribute__ ((vector_size (32)));

float** gen_mat_f(size_t size) {
	size_t i, j;

	//float (*a)[4][4] = _aligned_malloc(sizeof(*a));
	//float *aab = _mm_malloc(sizeof(float) * 8, 4);	
	
	float** ptr = malloc(size * sizeof(float*));
	for(i = 0; i < size; i++) {
		ptr[i] = _mm_malloc(size * sizeof(float), 32);
		for(j = 0; j < size; j++)
			ptr[i][j] = (rand() / (float) RAND_MAX) * (1);
	}

	/*float** ptr = malloc(size * sizeof(float*));
	for(i = 0; i < size; i++) {
		ptr[i] = malloc(size * sizeof(float));
		for(j = 0; j < size; j++)
			ptr[i][j] = (rand() / (float) RAND_MAX) * (1);
	}*/


	return ptr;
}

void free_mat_f(float** ptr, size_t size) {
	size_t i;

	for(i = 0; i < size; i++)
		_mm_free(ptr[i]);
	free(ptr);
}

void print_mat(float** m, size_t size) {
	size_t i, j;

	printf("Matrix:\n");
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
}

void print_f8(simd_f8 num) {
	int i = 0;

	for(i = 0; i < 8; i++)
		printf("%f ", ((float*)(&num))[i]);

	printf("\n");
}

void mult_f(float** a, float** b) {
	size_t row, col;

	float** c = gen_mat_f(8);
	for(row = 0; row < 8; row++) {
		for(col = 0; col < 8; col++)
			c[row][col] = 0.f;
	}

	for(row = 0; row < 8; row++) {
		simd_f8 a_r;
		simd_f8 b_r;

		for(col = 0; col < 8; col++) {
			a_r = _mm256_broadcast_ss(a[row] + col);
			b_r = _mm256_load_ps(b[col]);
			
			_mm256_store_ps(c[row], _mm256_load_ps(c[row]) +  _mm256_mul_ps(a_r, b_r));
		}
	}

	print_mat(c, 8);
}



void run() {
	float **m_1;
	float **m_2;
	int i, j;
	int size = 8;

	m_1 = gen_mat_f(size);
	m_2 = gen_mat_f(size);

	print_mat(m_1, size);
	print_mat(m_2, size);

	mult_f(m_1, m_2);

	free_mat_f(m_1, size);
	free_mat_f(m_2, size);
}

int main() {
	int x = 5;
	
	simd_f8 f1 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
	simd_f8 f2 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
	simd_f8 res;

	res = _mm256_mul_ps(f1, f2);

	for(x = 0; x < 8; x++)
		printf("%f ", ((float*)(&res))[x] );
	printf("\nStart:\n\n");
	run();
	/*float xy[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

	ff = L_SF8(&xy);

	printf("It Worked!\n");*/
}
