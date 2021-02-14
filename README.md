# Advanced Computer Systems Repository

## Project 1: SIMD matrix multiplication:
This project involves matrix multiplication using intel intrinsics for both floating and fixed (2 byte) point data.

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
`$ ./prog 128 snif`

**NOTE:** Flags have no delimiters so they should be entered all in one string (order doesn't matter).  Also if neither n -or- s is specified then both with be run, the same goes for the i and f flags.

### Results
