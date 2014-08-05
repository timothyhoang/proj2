#include <stdio.h>
#include <nmmintrin.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include "benchmark.h"

float *pad(float *A, size_t old_n, size_t new_n) {
    float *new_A = calloc(new_n*new_n, sizeof(float));  
    int i, j;
    for(j = 0; j < old_n; j++) {
	for(i = 0; i < old_n; i++) {
	    memcpy(&new_A[i+j*new_n], &A[i+j*old_n], sizeof(float));
	}
    }
    return new_A;
}

float *unpad(float *A, size_t old_n, size_t new_n) {
    float *new_A = calloc(old_n*old_n, sizeof(float));  
    int i, j;
    for(j = 0; j < old_n; j++) {
	for(i = 0; i < old_n; i++) {
	    memcpy(&new_A[i+j*old_n], &A[i+j*new_n], sizeof(float));
	}
    }
    return new_A;
}

void mmul(float *v, float *A, float *u, size_t n) {
    __m128 v_line1, v_line2, v_line3, v_line4, A_line1, A_line2, A_line3, A_line4, u_line;
    size_t i, j, l, jn;
    #pragma omp parallel for private(l, i, j, v_line1, v_line2, v_line3, v_line4, A_line1, A_line2, A_line3, A_line4, u_line)
    for(l=0; l<n; l++) {
	for(i=0; i<n; i+=16) {
	    v_line1 = _mm_set1_ps(0.0);
	    v_line2 = _mm_set1_ps(0.0);
	    v_line3 = _mm_set1_ps(0.0);
	    v_line4 = _mm_set1_ps(0.0);
	    for(j=0; j<n; j++) {
		jn = j*n;
		u_line = _mm_set1_ps(u[j+l*n]);
		A_line1 = _mm_load_ps(&A[jn+i]);
		v_line1 = _mm_add_ps(_mm_mul_ps(A_line1, u_line), v_line1);		
		A_line2 = _mm_load_ps(&A[jn+i+4]);
		v_line2 = _mm_add_ps(_mm_mul_ps(A_line2, u_line), v_line2);
		A_line3 = _mm_load_ps(&A[jn+i+8]);
		v_line3 = _mm_add_ps(_mm_mul_ps(A_line3, u_line), v_line3);
		A_line4 = _mm_load_ps(&A[jn+i+12]);
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

void mmul_n16(float *v, float *A, float *u, size_t n) {
    size_t n_16 = (n/16)*16;
    __m128 v_line1, v_line2, v_line3, v_line4, A_line1, A_line2, A_line3, A_line4, u_line;
    size_t i, j, l, ln, jn;
    #pragma omp parallel for private(l, i, j, v_line1, v_line2, v_line3, v_line4, A_line1, A_line2, A_line3, A_line4, u_line)
    for(l=0; l<n; l++) {
	for(i=0; i<n_16; i+=16) {
	    v_line1 = _mm_set1_ps(0.0);
	    v_line2 = _mm_set1_ps(0.0);
	    v_line3 = _mm_set1_ps(0.0);
	    v_line4 = _mm_set1_ps(0.0);
	    for(j=0; j<n; j++) {
		u_line = _mm_set1_ps(u[j+l*n]);
		A_line1 = _mm_loadu_ps(&A[j*n+i]);
		v_line1 = _mm_add_ps(_mm_mul_ps(A_line1, u_line), v_line1);		
		A_line2 = _mm_loadu_ps(&A[j*n+i+4]);
		v_line2 = _mm_add_ps(_mm_mul_ps(A_line2, u_line), v_line2);
		A_line3 = _mm_loadu_ps(&A[j*n+i+8]);
		v_line3 = _mm_add_ps(_mm_mul_ps(A_line3, u_line), v_line3);
		A_line4 = _mm_loadu_ps(&A[j*n+i+12]);
		v_line4 = _mm_add_ps(_mm_mul_ps(A_line4, u_line), v_line4);		
	    }
	    _mm_storeu_ps(&v[i+l*n], v_line1);
	    _mm_storeu_ps(&v[i+4+l*n], v_line2);
	    _mm_storeu_ps(&v[i+8+l*n], v_line3);
	    _mm_storeu_ps(&v[i+12+l*n], v_line4);
	}
	/*
	if(n-n_16 >= 4) {
	    size_t n_4 = (n/4)*4;
	    for(i=n_16; i<n_4; i+=4) {
		v_line1 = _mm_set1_ps(0.0);
		for(j=0; j<n; j++) {
		    u_line = _mm_set1_ps(u[j+l*n]);
		    A_line1 = _mm_loadu_ps(&A[j*n+i]);
		    v_line1 = _mm_add_ps(_mm_mul_ps(A_line1, u_line), v_line1);				
		}
		_mm_storeu_ps(&v[i+l*n], v_line1);
	    }	    
	    for (size_t i = n_4; i < n; i += 1) {
		for (size_t j = 0; j < n; j += 1) {
		    v[i + l*n] += u[j + l*n] * A[i + n*j];
		}
	    }
	    } else { */
	    for (size_t i = n_16; i < n; i += 1) {
		for (size_t j = 0; j < n; j += 1) {
		    v[i + l*n] += u[j + l*n] * A[i + n*j];
		}
	    }
	    //	}
    }
}


void euc_norm_n16(float *mu, float *v, size_t n) {
    size_t n_16 = (n/16)*16;
    __m128 sum, tmp;
    size_t l, i;
    #pragma omp parallel for private(l, i, sum, tmp)
    for(l=0; l<n; l++) {
	sum = _mm_setzero_ps();
	for(i=0; i<n_16; i=i+16) {
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
	mu[l] += MU[0]+MU[1]+MU[2]+MU[3];

	for(i=n_16; i < n; i++) {
	    mu[l] += v[i + l*n] * v[i + l*n];
	}
	mu[l] = sqrt(mu[l]);
    }
}

void div_norm_n16(float *u, float *v, float *mu, size_t n) {
    size_t n_16 = (n/16)*16;
    __m128 tmp, MU;
    int i, l;
    #pragma omp parallel for private(l, i, tmp, MU)
    for(l=0; l<n; l++) {
	MU = _mm_set1_ps(mu[l]);
	for(i=0; i<n_16; i+=4) {
	    u[i + l*n] = v[i + l*n] / mu[l];
	    u[i + 1 + l*n] = v[i + 1 + l*n] / mu[l];
	    u[i + 2 + l*n] = v[i + 2 + l*n] / mu[l];
	    u[i + 3 + l*n] = v[i + 3 + l*n] / mu[l];
	}
	for (size_t i = n_16; i < n; i += 1) {
	    u[i + l*n] = v[i + l*n] / mu[l];
	}
    }
}

void eig(float *v, float *A, float *u, size_t n, unsigned iters) {          
    if(n%16 > 0) {
	size_t k;
	for(size_t k = 0; k < iters; k +=1) {
	    /* v_k = Au_{k-1} */
	    memset(v, 0, n * n * sizeof(float));
	    mmul_n16(v, A, u, n);
	
	    /* mu_k = ||v_k|| */
	    float mu[n];
	    memset(mu, 0, n * sizeof(float));
	    euc_norm_n16(mu, v, n);

	    /* u_k = v_k / mu_k */
	    div_norm_n16(u, v, mu, n);
	}
    } else {
	size_t k;
	for(size_t k = 0; k < iters; k +=1) {
	    /* v_k = Au_{k-1} */
	    memset(v, 0, n * n * sizeof(float));
	    mmul(v, A, u, n);
	
	    /* mu_k = ||v_k|| */
	    float mu[n];
	    memset(mu, 0, n * sizeof(float));
	    euc_norm(mu, v, n);

	    /* u_k = v_k / mu_k */
	    div_norm(u, v, mu, n);
	}
    }
}
