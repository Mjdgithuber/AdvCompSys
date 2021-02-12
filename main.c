/*#include <x86intrin.h>*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fp_matrix.h"

#define PRINT_MAT 0

void run() {
	struct timeval t1, t2;
	float **m_1, **m_2, **m_res;
	int i, j;
	int size = 5500;

	m_1 = gen_fp_mat(size, 1);
	m_2 = gen_fp_mat(size, 1);

	//print_fp_mat(m_1, size);
	//print_fp_mat(m_2, size);

	if(PRINT_MAT) {
		print_mat(m_1, size);
		print_mat(m_2, size);
	}

	//naive_mult(m_1, m_2, size);
	
	//printf("A Checksum: %f\n", fp_mat_checksum(m_1, size));
	//printf("B Checksum: %f\n", fp_mat_checksum(m_2, size));

	gettimeofday(&t1, NULL);
	m_res = mult_fp_mat(m_1, m_2, size);
	gettimeofday(&t2, NULL);
	printf("SIMD  Checksum: %f in %u seconds\n", fp_mat_checksum(m_res, size), t2.tv_sec - t1.tv_sec);
	//print_fp_mat(m_res, size);

	free_fp_mat(m_1, size);
	free_fp_mat(m_2, size);
	free_fp_mat(m_res, size);
}

int main() {
	run();
}
