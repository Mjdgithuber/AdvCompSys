#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "fp_matrix.h"
#include "si_matrix.h"

#define PRINT_MAT 0

void run_fixed_point() {
	struct timeval t1, t2;
	short int **m_1, **m_2, **m_res;
	int size = 8;

	m_1 = gen_si_mat(size, 1);
	m_2 = gen_si_mat(size, 1);

	print_si_mat(m_1, size);
	print_si_mat(m_2, size);

	gettimeofday(&t1, NULL);
	m_res = mult_si_mat_naive(m_1, m_2, size);
	gettimeofday(&t2, NULL);
	print_si_mat(m_res, size);
	printf("NAIVE Checksum Fixed Point: %hd in %u seconds\n", si_mat_checksum(m_res, size), (unsigned int)(t2.tv_sec - t1.tv_sec));

	free_si_mat(m_1, size);
	free_si_mat(m_2, size);
	free_si_mat(m_res, size);
}

void run_floating_point() {
	struct timeval t1, t2;
	float **m_1, **m_2, **m_res;
	int size = 1000;

	m_1 = gen_fp_mat(size, 1);
	m_2 = gen_fp_mat(size, 1);

	//print_fp_mat(m_1, size);
	//print_fp_mat(m_2, size);

	if(PRINT_MAT) {
		print_fp_mat(m_1, size);
		print_fp_mat(m_2, size);
	}

	gettimeofday(&t1, NULL);
	m_res = mult_fp_mat_naive(m_1, m_2, size);
	gettimeofday(&t2, NULL);
	printf("NAIVE Checksum: %f in %u seconds\n", fp_mat_checksum(m_res, size), (unsigned int)(t2.tv_sec - t1.tv_sec));
	free_fp_mat(m_res, size);

	
	//printf("A Checksum: %f\n", fp_mat_checksum(m_1, size));
	//printf("B Checksum: %f\n", fp_mat_checksum(m_2, size));

	gettimeofday(&t1, NULL);
	m_res = mult_fp_mat(m_1, m_2, size);
	gettimeofday(&t2, NULL);
	printf("SIMD  Checksum: %f in %u seconds\n", fp_mat_checksum(m_res, size), (unsigned int)(t2.tv_sec - t1.tv_sec));
	//print_fp_mat(m_res, size);

	free_fp_mat(m_1, size);
	free_fp_mat(m_2, size);
	free_fp_mat(m_res, size);
}

int main() {
	run_fixed_point();

	return 0;
}
