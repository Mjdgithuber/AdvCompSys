# Advanced Computer Systems Repository

## Project 1: SIMD matrix multiplication:
This project involves matrix multiplication using intel intrinsics for both floating and fixed (2 byte) point data.

### Code Structure
As I have decided to do this in C which doesn't support template syntax, there are two pairs of files that contain the implementation of fixed and floating point matrix multiplication.  The floating point implementation is contained within fp_matrix.c file while the fixed point is implemented in si_matrix.c (si stands for short int).  Their respective header files contain common functions to generate, free, randomize, perform a checksum, print, and multiply matrices (both with and without SIMD & optimizations).  The naive multiplication functions are standard and don't require a detailed description. 

#### SIMD Functions
At the top of both fp_matrix.c and si_matrix.c you will see `#include <immintrin.h>` which will allow use of Intel Intrinsics.  The following is a description of the wrappers used to perform the multiplication.

##### Fixed Point
All fixed point instructions used here require SSE2.

`_mm_loadu_si128(__m128i const* mem_addr)` - Load 128-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.

`_mm_set1_epi16(short a)` - Broadcast 16-bit integer a to all all elements of dst. This intrinsic may generate vpbroadcastw.

`_mm_add_epi16(__m128i a, __m128i b)` - Add packed 16-bit integers in a and b, and store the results in dst.

`_mm_store_si128(__m128i* mem_addr, __m128i a)` - Store 128-bits of integer data from a into memory. mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.

##### Floating Point
All floating point instructions used here require AVX.

`_mm256_load_ps(float const * mem_addr)` - Load 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) from memory into dst. mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.

`_mm256_broadcast_ss(float const * mem_addr)` - Broadcast a single-precision (32-bit) floating-point element from memory to all elements of dst.

`_mm256_mul_ps(__m256 a, __m256 b)` - Multiply packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.

`_mm256_store_ps(float * mem_addr, __m256 a)` - Store 256-bits (composed of 8 packed single-precision (32-bit) floating-point elements) from a into memory. mem_addr must be aligned on a 32-byte boundary or a general-protection exception may be generated.

#### Common
Many of the Intel intrinsic instructions require boundary alignment in order to work properly, in cases where it isn't aligned (and alignment is required) it could cause a seg fault.  At the top of both files there is a macro definition for `BOUNDARY_ALIGNMENT` that will ensure that the matrix's cells will be properly aligned with the use of `_mm_malloc(int size, int align)` (OS independent) and `_mm_free (void *p)`.

### Build & Requirements
In order to run this code your CPU must support the following flags: SSE2 (for fixed point), and AVX (for fixed point).

A simple way to check is to use regular expressions to search for the flags for example:

`$ grep -E '(avx.*sse2)|(sse2.*avx)' /proc/cpuinfo -c`

Your CPU supports both AVX and SSE2 if the above line returns the number of physical cores you have (0 if unsupported).

To build the code run the top line in the comp file:
```
$ cat comp
gcc main.c fp_matrix.c si_matrix.c -o prog -mavx -Wall
gcc -S -masm=intel -mavx main.c
$ gcc main.c fp_matrix.c si_matrix.c -o prog -mavx -Wall
```
Note that the second line in comp can be used to produce an assembly dump (in Intel syntax of a given file).  This can be used to make micro-optimizations by seeing what the compiler is outputting and tweaking your source to produce better assembly.

**NOTE:** My code uses the sys/time.h included on POSIX compliant operating systems which is not included int the Windows C libraries.  This only applies to compilers native to Windows such as mingw, most systems that provide a Unix-like command line (or full emulation) will have this file such as WSL, Git Bash, and Cygwin.  If this isn't available on your system, the timing code can be removed/altered to allow the program to build.

### Running
To run the program it needs two arguments, the size of the square matrices and flags:  `$ ./prog size flags`

Description of flags:
* n - Run naive implementation
* s - Run SIMD implementation
* i - Run with integer (2 byte) matrices
* f - Run floating point (4 byte) matrices
* p - Print out matrices (this should only be used for small matrices)

Example to run SIMD & naive implementation for floating and fixed point matrices with size of 128.
```
$ ./prog 128 snif
NAIVE Checksum Fixed Point: -28805 in 0 seconds
SIMD  Checksum Fixed Point: -28805 in 0 seconds
NAIVE Checksum: 524038.656250 in 0 seconds
SIMD  Checksum: 524038.656250 in 0 seconds
```
The checksums can be used to verify that both the naive and SIMD implementations return the same result.

**NOTE:** Flags have no delimiters so they should be entered all in one string (order doesn't matter).  Also if neither n -or- s is specified then both with be run, the same goes for the i and f flags.

### Results
