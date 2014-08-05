#include <stdio.h>
//#include <nmmintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

float *pad_m(float *A, size_t old_n, size_t new_n) {
    float *new_A = malloc(new_n*new_n*sizeof(float));  
    int i, j;
    for(j = 0; j < old_n; j++) {
	for(i = 0; i < old_n; i++) {
	    memcpy(&new_A[i+j*new_n], &A[i+j*old_n], sizeof(float));
	}
    }
    return new_A;
}

void transpose(int n, int blocksize, float *dst, float *src) {
    int i,j,l,k;
    for(l = 0; l < n/blocksize; l++) {
	for(k = 0; k < n/blocksize; k++) {
	    for(i = 0; i < blocksize && i+l*blocksize < n; i++ ) {
		for(j = 0; j < blocksize && j+k*blocksize < n; j++ ) {
		    dst[j+k*blocksize+(l*blocksize+i)*n] = src[l*blocksize+i+(k*blocksize+j)*n];
		}
	    }
	}
    }
}


float *unpad_m(float *A, size_t old_n, size_t new_n) {
    float *new_A = calloc(old_n*old_n, sizeof(float));  
    int i, j;
    for(j = 0; j < old_n; j++) {
	for(i = 0; i < old_n; i++) {
	    memcpy(&new_A[i+j*old_n], &A[i+j*new_n], sizeof(float));
	}
    }
    return new_A;
}

void mmul(float *v, float *A, float *u, size_t n, unsigned iters) {
    for (size_t l = 0; l < n; l += 1) {
	for (size_t i = 0; i < 2; i += 1) {
	    for (size_t j = 0; j < n; j += 1) {
		v[j + l*n] += u[i + l*n] * A[j + n*i];
	    }
	}
    }
}

void eig_naive(float *v, float *A, float *u, size_t n, unsigned iters) {
    for (size_t k = 0; k < iters; k += 1) {
        /* v_k = Au_{k-1} */
        memset(v, 0, n * n * sizeof(float));
        for (size_t l = 0; l < n; l += 1) {
            for (size_t i = 0; i < n; i += 1) {
                for (size_t j = 0; j < n; j += 1) {
                    v[i + l*n] += u[j + l*n] * A[i + n*j];
                }
            }
        }

    }
}

int main() {
    float A1[9] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; //, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    float *A = A1;
    float *B = calloc(9, sizeof(float));
    
    // printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8], A[9], A[10], A[11], A[12], A[13], A[14], A[15]);

    printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]);
    
    printf("\n");

    transpose(3, 2, B, A);

    printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8]);

    // printf("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n", B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7], B[8], B[9], B[10], B[11], B[12], B[13], B[14], B[15]);
    
    
    return 0;  
}
/*
    for(size_t k = 0; k < 3; k += 1) {
	for (size_t l = 0; l < old_n; l += 1) {
	    for (size_t i = 0; i < old_n; i += 1) {
		for (size_t j = 0; j < old_n; j += 1) {
		    v[i + l*new_n] += u[j + l*new_n] * A[i + new_n*j];
		}
	    }
	}


	float mu[old_n];
	memset(mu, 0, old_n * sizeof(float));
	for (size_t l = 0; l < old_n; l += 1) {
	    for (size_t i = 0; i < old_n; i += 1) {
		mu[l] += v[i + l*new_n] * v[i + l*new_n];
	    }
	    mu[l] = sqrt(mu[l]);
	}
    

	for (size_t l = 0; l < old_n; l += 1) {
	    for (size_t i = 0; i < old_n; i += 1) {
		u[i + l*new_n] = v[i + l*new_n] / mu[l];
	    }
	}
	//printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]);
	//printf("\n");
    }
*/

/*

void mmul(float *v, float *A, float *u, size_t n) {
    __m128 v_line1, v_line2, v_line3, v_line4, A_line1, A_line2, A_line3, A_line4, u_line;
    size_t i, j, l;
    #pragma omp parallel for private(l, i, j, v_line1, v_line2, v_line3, v_line4, A_line1, A_line2, A_line3, A_line4, u_line)
    for(l=0; l<n; l++) {
	for(i=0; i<n; i+=16) {	    
	    u_line = _mm_set1_ps(u[0+l*n]);
	    A_line1 = _mm_load_ps(&A[i]);
	    v_line1 = _mm_mul_ps(A_line1, u_line);	    
	    A_line2 = _mm_load_ps(&A[i+4]);
	    v_line2 = _mm_mul_ps(A_line2, u_line);
	    A_line3 = _mm_load_ps(&A[i+8]);
	    v_line3 = _mm_mul_ps(A_line3, u_line);
	    A_line4 = _mm_load_ps(&A[i+12]);
	    v_line4 = _mm_mul_ps(A_line4, u_line);	    
	    for(j=1; j<n; j++) {
		u_line = _mm_set1_ps(u[j+l*n]);
		A_line1 = _mm_load_ps(&A[j*n+i]);
		v_line1 = _mm_add_ps(_mm_mul_ps(A_line1, u_line), v_line1);		
		A_line2 = _mm_load_ps(&A[j*n+i+4]);
		v_line2 = _mm_add_ps(_mm_mul_ps(A_line2, u_line), v_line2);
		A_line3 = _mm_load_ps(&A[j*n+i+8]);
		v_line3 = _mm_add_ps(_mm_mul_ps(A_line3, u_line), v_line3);
		A_line4 = _mm_load_ps(&A[j*n+i+12]);
		v_line4 = _mm_add_ps(_mm_mul_ps(A_line4, u_line), v_line4);		
	    }
	    _mm_store_ps(&v[i+l*n], v_line1);
	    _mm_store_ps(&v[i+4+l*n], v_line2);
	    _mm_store_ps(&v[i+8+l*n], v_line3);
	    _mm_store_ps(&v[i+12+l*n], v_line4);
	}
    }
}

void euc_norm(float *mu, float *v, size_t n) {
    __m128 sum, tmp;
    size_t l, i;
    #pragma omp parallel for private(l, i, sum, tmp)
    for(l=0; l<n; l++) {
	sum = _mm_setzero_ps();
	for(i=0; i<n; i=i+16) {
	    tmp = _mm_loadu_ps(v+i+l*n);
	    tmp = _mm_mul_ps(tmp, tmp);
	    sum = _mm_add_ps(sum,tmp);
	    tmp = _mm_loadu_ps(v+i+4+l*n);
	    tmp = _mm_mul_ps(tmp, tmp);
	    sum = _mm_add_ps(sum,tmp);                   
	    tmp = _mm_loadu_ps(v+i+8+l*n);
	    tmp = _mm_mul_ps(tmp, tmp);
	    sum = _mm_add_ps(sum,tmp);                   
	    tmp = _mm_loadu_ps(v+i+12+l*n);
	    tmp = _mm_mul_ps(tmp, tmp);
	    sum = _mm_add_ps(sum,tmp);                   
	}
	float MU[4] = {0.0, 0.0, 0.0, 0.0};  
	_mm_storeu_ps(MU, sum);
	mu[l] += sqrt(MU[0]+MU[1]+MU[2]+MU[3]);
    }
}

void div_norm(float *u, float *v, float *mu, size_t n) {
    __m128 tmp, MU;
    int i, l;
    #pragma omp parallel for private(l, i, tmp, MU)
    for(l=0; l<n; l++) {
	MU = _mm_set1_ps(mu[l]);
	for(i=0; i<n; i+=16) {
	    tmp = _mm_loadu_ps(v+i+l*n);
	    tmp = _mm_div_ps(tmp, MU);	    
	    _mm_store_ps(u+i+l*n, tmp);
	    tmp = _mm_loadu_ps(v+i+4+l*n);
	    tmp = _mm_div_ps(tmp, MU);	    
	    _mm_store_ps(u+i+4+l*n, tmp);
	    tmp = _mm_loadu_ps(v+i+8+l*n);
	    tmp = _mm_div_ps(tmp, MU);	    
	    _mm_store_ps(u+i+8+l*n, tmp);
	    tmp = _mm_loadu_ps(v+i+12+l*n);
	    tmp = _mm_div_ps(tmp, MU);	    
	    _mm_store_ps(u+i+12+l*n, tmp);
	}
    }
}
*/

/*
float *pad_v(float *v, size_t old_n, size_t new_n) {
    float *new_v = calloc(new_n, sizeof(float));
    for(int i = 0; i < old_n; i++) {
	new_v[i] = v[i]; 
    }
    return new_v;
}

float *pad_m(float *A, size_t old_n, size_t new_n) {
    float *new_A = calloc(new_n*new_n, sizeof(float));  
    int i, j;
    for(j = 0; j < old_n; j++) {
	for(i = 0; i < old_n; i++) {
	    new_A[i+j*new_n] = A[i+j*old_n];
	}
    }
    return new_A;
}

float *unpad_v(float *v, size_t n) {
    float *new_v = calloc(n, sizeof(float));
    for(int i = 0; i < n; i++) {
	new_v[i] = v[i]; 
    }
    return new_v;
}

float *unpad_m(float *A, size_t old_n, size_t new_n) {
    float *new_A = calloc(old_n*old_n, sizeof(float));  
    int i, j;
    for(j = 0; j < old_n; j++) {
	for(i = 0; i < old_n; i++) {
	    new_A[i+j*old_n] = A[i+j*new_n];
	}
    }
    return new_A;
}
*/
