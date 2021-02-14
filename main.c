#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include "fp_matrix.h"
#include "si_matrix.h"

void run_fixed_point(size_t sz, BOOL naive, BOOL simd, BOOL print) {
	struct timeval t1, t2;
	short int **m_1, **m_2, **m_res;

	m_1 = gen_si_mat(sz, 1);
	m_2 = gen_si_mat(sz, 1);

	if(print) {
		print_si_mat(m_1, sz);
		print_si_mat(m_2, sz);
	}

	if(naive) {
		gettimeofday(&t1, NULL);
		m_res = mult_si_mat_naive(m_1, m_2, sz);
		gettimeofday(&t2, NULL);
		if(print) print_si_mat(m_res, sz);
		printf("NAIVE Checksum Fixed Point: %hd in %u seconds\n", si_mat_checksum(m_res, sz), (unsigned int)(t2.tv_sec - t1.tv_sec));
		free_si_mat(m_res, sz);
	}

	if(simd) {
		gettimeofday(&t1, NULL);
		m_res = mult_si_mat(m_1, m_2, sz);
		gettimeofday(&t2, NULL);
		if(print) print_si_mat(m_res, sz);
		printf("SIMD  Checksum Fixed Point: %hd in %u seconds\n", si_mat_checksum(m_res, sz), (unsigned int)(t2.tv_sec - t1.tv_sec));
		free_si_mat(m_res, sz);
	}

	free_si_mat(m_1, sz);
	free_si_mat(m_2, sz);
}

void run_floating_point(size_t sz, BOOL naive, BOOL simd, BOOL print) {
	struct timeval t1, t2;
	float **m_1, **m_2, **m_res;

	m_1 = gen_fp_mat(sz, 1);
	m_2 = gen_fp_mat(sz, 1);

	if(print) {
		print_fp_mat(m_1, sz);
		print_fp_mat(m_2, sz);
	}
	
	if(naive) {
		gettimeofday(&t1, NULL);
		m_res = mult_fp_mat_naive(m_1, m_2, sz);
		gettimeofday(&t2, NULL);
		if(print) print_fp_mat(m_res, sz);
		printf("NAIVE Checksum: %f in %u seconds\n", fp_mat_checksum(m_res, sz), (unsigned int)(t2.tv_sec - t1.tv_sec));
		free_fp_mat(m_res, sz);
	}

	if(simd) {
		gettimeofday(&t1, NULL);
		m_res = mult_fp_mat(m_1, m_2, sz);
		gettimeofday(&t2, NULL);
		if(print) print_fp_mat(m_res, sz);
		printf("SIMD  Checksum: %f in %u seconds\n", fp_mat_checksum(m_res, sz), (unsigned int)(t2.tv_sec - t1.tv_sec));
		free_fp_mat(m_res, sz);
	}

	free_fp_mat(m_1, sz);
	free_fp_mat(m_2, sz);
}

int main(int argc, char** argv) {
	if(argc < 3) {
		printf("Must be in the format: ./prog size_of_matrices flags\nFlags:\n");
		printf("s - run simd algorithms\nn- run naive algorithms\n");
		printf("i - run fixed point\nf - run floating point\n");
		printf("p - print matrices\n");
		printf("No delimiter between flags.  Example to run all:\n");
		printf("./prog 1000 snif\n");
		printf("Flag order doesn't matter\n");
		return 0;
	}
	
	BOOL simd = strchr(argv[2], 's') ? 1 : 0;
	BOOL naive = strchr(argv[2], 'n') ? 1 : 0;
	BOOL fix = strchr(argv[2], 'i') ? 1 : 0;
	BOOL flo = strchr(argv[2], 'f') ? 1 : 0;
	BOOL pri = strchr(argv[2], 'p') ? 1 : 0;
	size_t sz = atoi(argv[1]);

	if(sz % 8) {
		printf("Input size must be a multiple of 8!\n");
		return 0;
	}
	
	/* if method not specified run both */
	if(!simd && !naive) {
		simd = 1;
		naive = 1;
	}
	
	/* if type not specified run both */
	if(!fix && !flo) {
		fix = 1;
		flo = 1;
	}

	if(fix)
		run_fixed_point(sz, naive, simd, pri);
	if(flo)
		run_floating_point(sz, naive, simd, pri);

	return 0;
}
