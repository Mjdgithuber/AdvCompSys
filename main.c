/*#include <x86intrin.h>*/
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#define L_SF8(ADDR) *((const simd_f8 *)(ADDR))

typedef int simd_f __attribute__ ((vector_size (16)));

typedef float simd_f8 __attribute__ ((vector_size (32)));

float** fill_mat_f(float** mat, size_t size) {
	size_t i, j;

	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++)
			mat[i][j] = (rand() / (float) RAND_MAX) * (1);
	}

	return mat;
}

float** gen_mat_f(size_t size, char fill) {
	size_t i, j;

	//float (*a)[4][4] = _aligned_malloc(sizeof(*a));
	//float *aab = _mm_malloc(sizeof(float) * 8, 4);	
	
	float** ptr = malloc(size * sizeof(float*));
	for(i = 0; i < size; i++) {
		ptr[i] = _mm_malloc(size * sizeof(float), 32);
		//for(j = 0; j < size; j++)
			//ptr[i][j] = (rand() / (float) RAND_MAX) * (1);
	}

	/*float** ptr = malloc(size * sizeof(float*));
	for(i = 0; i < size; i++) {
		ptr[i] = malloc(size * sizeof(float));
		for(j = 0; j < size; j++)
			ptr[i][j] = (rand() / (float) RAND_MAX) * (1);
	}*/


	return fill ? fill_mat_f(ptr, size) : ptr;
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

	printf("Printing f8:\n");
	for(i = 0; i < 8; i++)
		printf("%f ", ((float*)(&num))[i]);

	printf("\n");
}

void naive_mult(float** a, float** b, int size) {
	int i, j, k;

	float** res = gen_mat_f(size, 0);

	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			for(k = 0; k < size; k++)
				res[i][j] += a[i][k] * b[k][j];
		}
	}

	print_mat(res, size);
}

#define AZ 2
#define BZ 2
void mult_f_t2(float** a, float** b, int size) {
	size_t row, col;
	
	simd_f8 csum[AZ][BZ] = {{0.0}};
	int ai, bi, ii;

	simd_f8 a_r, b_r;
	
	float** c = gen_mat_f(size, 0);

	for(ii = 0; ii < size; ii++) {
		for(bi = 0; bi < BZ; bi++) {
			b_r = _mm256_load_ps(b[ii] + 8*bi);
			for(ai = 0; ai < AZ; ai++) {
				a_r = _mm256_broadcast_ss(a[ai] + ii);
				//print_f8(csum[ai][bi] + _mm256_mul_ps(a_r, b_r));
				csum[ai][bi] += _mm256_mul_ps(a_r, b_r);
			}
		}
	}
	
	for(ai = 0; ai < AZ; ai++) {
		for(bi = 0; bi < BZ; bi++) {
			_mm256_store_ps(&c[ai][bi*8], csum[ai][bi]);
		}
	}
	
	printf("Done!\n");
	
	//for(ai = 0; ai < 3; ai++
	print_f8(csum[0][0]);
	print_f8(csum[0][1]);

	printf("\n\nFinal:\n");
	print_mat(c, 16);


	//printf("\nEmma:%f\n", *((float*)&csum[0][0]));
	//print_mat((float**)csum, 1);

	/*for(row = 0; row < 8; row++) {
		for(col = 0; col < 8; col++)
			c[row][col] = 0.f;
	}

	for(row = 0; row < 8; row++) {
		simd_f8 a_r;
		simd_f8 b_r;


		b_r = _mm256_load_ps(b[row]);
		for(col = 0; col < 8; col++) {
			a_r = _mm256_broadcast_ss(a[col] + row);
			//a_r = _mm256_broadcast_ss(a[row] + col);
			//b_r = _mm256_load_ps(b[col]);
			
			_mm256_store_ps(c[col], _mm256_load_ps(c[col]) +  _mm256_mul_ps(a_r, b_r));
		}
	}

	print_mat(c, 8);*/
}

void mult_f(float** a, float** b) {
	size_t row, col;

	float** c = gen_mat_f(8, 1);
	for(row = 0; row < 8; row++) {
		for(col = 0; col < 8; col++)
			c[row][col] = 0.f;
	}

	for(row = 0; row < 8; row++) {
		simd_f8 a_r;
		simd_f8 b_r;


		b_r = _mm256_load_ps(b[row]);
		for(col = 0; col < 8; col++) {
			a_r = _mm256_broadcast_ss(a[col] + row);
			//a_r = _mm256_broadcast_ss(a[row] + col);
			//b_r = _mm256_load_ps(b[col]);
			
			_mm256_store_ps(c[col], _mm256_load_ps(c[col]) +  _mm256_mul_ps(a_r, b_r));
		}
	}

	print_mat(c, 8);
}



void run() {
	float **m_1;
	float **m_2;
	int i, j;
	int size = 32;

	m_1 = gen_mat_f(size, 1);
	m_2 = gen_mat_f(size, 1);

	print_mat(m_1, size);
	print_mat(m_2, size);

	naive_mult(m_1, m_2, size);
	mult_f_t2(m_1, m_2, size);

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
