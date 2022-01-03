#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails. Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows < 1 || cols < 1) {
        // fail if the rows and cols are non positive
        // invalid rows and cols
        // remember to set error messages in numc.c
        return -1;
    } 
    *mat = (matrix *) calloc(1, sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->ref_cnt = 1;
    int matSize = rows * cols;
    (*mat)->data = (double *) calloc(matSize, sizeof(double));
    if ((*mat)->data == NULL) {
        return -2;
    }
    (*mat)->parent = NULL;
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Remember to set the error messages in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows < 1 || cols < 1) {
        // invalid rows and cols
        // remember to set error messages in numc.c
        return -1;
    } 
    *mat = (matrix *) calloc(1, sizeof(matrix));
    if (*mat == NULL) {
        return -2;
    }
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    // *** add ref_cnt for mat and maybe from
    from->ref_cnt += 1; // parent-ref_cnt ++
    (*mat)->data = &(from->data[offset]); // from->data[offset]
    // from->data --> double pointer 
    (*mat)->parent = from;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (mat == NULL) {
        // You cannot assume that mat is not NULL.
        return;
    }
    if (mat->parent == NULL && mat->ref_cnt == 1) {
        // ref_cnt == 1 only the current matrix is using the data
        // no other matrix are referencing data from current matrix
        free(mat->data);
    } 
    if (mat->ref_cnt == 1 && mat->parent != NULL && mat->parent->ref_cnt == 2) {
        // parent matrix has no other reference --> parent->ref_cnt == 2
        free(mat->parent->data);
    }
    // Q: do i need to free matrix as well???
    free(mat);
    return;
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    // implementation for row major
    int width = mat->cols;
    int index = row * width + col;
    double val = (mat->data)[index];
    return val;
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    int width = mat->cols;
    int index = row * width + col;
    (mat->data)[index] = val;
    return;  
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix_naive(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    int matSize = (mat->cols) * (mat->rows);
    for (int offset = 0; offset < matSize; offset++) {
        (mat->data)[offset] = val;
    }
    return;
}


void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    // this particular version implement both avx and parallel
    int matSize = (mat->cols) * (mat->rows);
    # pragma omp parallel for 
    for (int offset = 0; offset < matSize / 4 * 4; offset+=4) {
        __m256d setVal = _mm256_set1_pd (val); // set all four foubles in the mm256d to val
        _mm256_storeu_pd (&(mat->data[offset]), setVal); 
    }
    # pragma omp parallel for   
    for(int offset = matSize / 4 * 4; offset < matSize; offset++) {
        // tail case
        (mat->data)[offset] = val;
    }
    return;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix_naive(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return -1; // incorrect dimension
    }
    result->rows = mat1->rows;
    result->cols = mat1->cols;
    int matSize = (result->rows) * (result->cols);
    result->data = (double *) malloc(matSize * sizeof(double)); 
    // double check whether result->data need creation
    if (result->data == NULL) {
        return -2; // fail to create double array
    }
    result->ref_cnt = 1;
    result->parent = NULL;

    for (int offset = 0; offset < matSize; offset++) {
        (result->data)[offset] = (mat1->data)[offset] + (mat2->data)[offset];
    }
    return 0;
}


int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    // this particular version implement avx and parallel
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return -1; // incorrect dimension
    }
    int matSize = (mat1->rows) * (mat1->cols);
    #pragma omp parallel for 
    for (int offset = 0; offset < matSize / 4 * 4; offset+=4) {
        __m256d mat1Vec = _mm256_loadu_pd(&(mat1->data[offset]));
        __m256d mat2Vec = _mm256_loadu_pd(&(mat2->data[offset]));
        __m256d resultVec = _mm256_add_pd (mat1Vec, mat2Vec);
        _mm256_storeu_pd (&(result->data[offset]), resultVec); 
    }
    #pragma omp parallel for 
    for(int offset = matSize / 4 * 4; offset < matSize; offset++) {
        // tail case
        (result->data)[offset] = (mat1->data)[offset] + (mat2->data)[offset];
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix_naive(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return -1; // incorrect dimension
    }
    result->rows = mat1->rows;
    result->cols = mat1->cols;
    int matSize = (result->rows) * (result->cols);
    result->data = (double *) malloc(matSize * sizeof(double));
    if (result->data == NULL) {
        return -2; // fail to create double array
    }
    result->ref_cnt = 1;
    result->parent = NULL;

    for (int offset = 0; offset < matSize; offset++) {
        (result->data)[offset] = (mat1->data)[offset] - (mat2->data)[offset];
    }
    return 0;
}


int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    // this particular version implement both avx and parallel
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        return -1; // incorrect dimension
    }
    int matSize = (mat1->rows) * (mat1->cols);
    #pragma omp parallel for 
    for (int offset = 0; offset < matSize / 4 * 4; offset+=4) {
        __m256d mat1Vec = _mm256_loadu_pd(&(mat1->data[offset]));
        __m256d mat2Vec = _mm256_loadu_pd(&(mat2->data[offset]));
        __m256d resultVec = _mm256_sub_pd (mat1Vec, mat2Vec);
        _mm256_storeu_pd (&(result->data[offset]), resultVec); 
    }
    #pragma omp parallel for 
    for(int offset = matSize / 4 * 4; offset < matSize; offset++) {
        // tail case
        (result->data)[offset] = (mat1->data)[offset] - (mat2->data)[offset];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix_naive(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */

    if (mat1->cols != mat2-> rows) {
        return -1; // incorrect dimension
    }
    int matSize = (mat1->rows) * (mat2->cols);
    // best performance jki order
    int resultWidth = mat2->cols;
    int resultHeight = mat1->rows;
    int mat1Width = mat1->cols;
    int mat1Height = mat1->rows;
    int mat2Width = mat2->cols;
    int mat2Height = mat2->rows;
    /* This is ijk loop order. */
    #pragma omp parallel for 
    for(int i = 0; i < mat1Height; i++) {
        // mat1rows
        for(int j = 0; j < mat2Width; j++) {
            double cij = 0;
            for(int k = 0; k < mat1Width; k++) {
                cij += (mat1->data)[i*mat1Width + k] * (mat2->data)[k*mat2Width + j];
            }
            (result->data)[i*resultWidth + j] = cij;
        }
    }
    return 0;   
}

int mul_matrix_cache_blocking(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    void transpose_blocking(int rows, int cols, double *dst, double *src);
    if (mat1->cols != mat2-> rows) {
        return -1; // incorrect dimension
    }
    int mat2Size = (mat2->rows) * (mat2->cols);
    double transposeMat2[mat2Size];
    transpose_blocking(mat2->rows, mat2->cols, transposeMat2, mat2->data); // mat2 to transpose mat2
    int mat1Width = mat1->cols;
    int mat1Height = mat1->rows;
    int mat2Width = mat2->cols;
    int mat2Height = mat2->rows;


    int blockSize = 8;
    int block_num_mat1 = mat1Height / blockSize;
    int block_num_mat2 = mat2Width / blockSize;
    int block_remain_mat1 = mat1Height % blockSize;
    int block_remain_mat2 = mat2Width % blockSize;
    for (int x = 0; x <= block_num_mat1; x++) {
        int xEnd = blockSize;
        if (x == block_num_mat1) {
            xEnd = block_remain_mat1;
        }
        for (int y = 0; y <= block_num_mat2; y++) {
            int yEnd = blockSize;
            if (y == block_num_mat1) {
                yEnd = block_remain_mat2;
            }
            for (int bx = 0; bx < xEnd; bx++) {
                for (int by = 0; by < yEnd; by++) {
                    int i = x * blockSize + bx;
                    int j = y * blockSize + by;
                    double cij = 0;
                    for(int k = 0; k < mat1Width; k++) {
                        cij += (mat1->data)[i*mat1Width + k] * transposeMat2[j*mat2Height + k];
                    }
                    (result->data)[i*mat2Width + j] = cij;
                }
            }
        }
    }
    return 0;   
}

int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    void transpose_blocking(int rows, int cols, double *dst, double *src);
    if (mat1->cols != mat2-> rows) {
        return -1; // incorrect dimension
    }
    int mat2Size = (mat2->rows) * (mat2->cols);
    double *transposeMat2 = (double *) calloc(mat2Size, sizeof(double));
    transpose_blocking(mat2->rows, mat2->cols, transposeMat2, mat2->data); // mat2 to transpose mat2
    int mat1Width = mat1->cols;
    int mat1Height = mat1->rows;
    int mat2Width = mat2->cols;
    int mat2Height = mat2->rows;

    int blockSize = 16;
    int block_num_mat1 = mat1Height / blockSize;
    int block_num_mat2 = mat2Width / blockSize;
    int block_remain_mat1 = mat1Height % blockSize;
    int block_remain_mat2 = mat2Width % blockSize;
    #pragma omp parallel for 
    for (int x = 0; x <= block_num_mat1; x++) {
        int xEnd = blockSize;
        if (x == block_num_mat1) {
            xEnd = block_remain_mat1;
        }
        for (int y = 0; y <= block_num_mat2; y++) {
            int yEnd = blockSize;
            if (y == block_num_mat2) {
                yEnd = block_remain_mat2;
            }
            for (int bx = 0; bx < xEnd; bx++) {
                for (int by = 0; by < yEnd; by++) {
                    int i = x * blockSize + bx;
                    int j = y * blockSize + by;
                    __m256d cijVec = _mm256_setzero_pd();
                    for(int k = 0; k < mat1Width / 16 * 16; k+=16) {
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&((mat1->data)[i*mat1Width + k])), _mm256_loadu_pd(&(transposeMat2[j*mat2Height + k])), cijVec);
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&((mat1->data)[i*mat1Width + (k + 4)])), _mm256_loadu_pd(&(transposeMat2[j*mat2Height + (k + 4)])), cijVec);
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&((mat1->data)[i*mat1Width + (k + 8)])), _mm256_loadu_pd(&(transposeMat2[j*mat2Height + (k + 8)])), cijVec);
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&((mat1->data)[i*mat1Width + (k + 12)])), _mm256_loadu_pd(&(transposeMat2[j*mat2Height + (k + 12)])), cijVec);
                    }
                    double tmp_arr[4];
                    _mm256_storeu_pd (tmp_arr, cijVec); 
                    double cij = tmp_arr[0] + tmp_arr[1] + tmp_arr[2] + tmp_arr[3];
                    for(int k = mat1Width / 16 * 16; k < mat1Width; k++) {
                        // tail case
                        cij += (mat1->data)[i*mat1Width + k] * transposeMat2[j*mat2Height + k];
                    }
                    (result->data)[i*mat2Width + j] = cij;
                }
            }
        }
    }
    free(transposeMat2);
    return 0;  
}

void transpose_naive(int rows, int cols, double *dst, double *src) {
    #pragma omp parallel for 
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            dst[y + x * rows] = src[x + y * cols];
        }
    }
    return;
}

void transpose_blocking(int rows, int cols, double *dst, double *src) {
    // YOUR CODE HERE
    // concise
    int blocksize = 16; // find a number to optimize
    int block_count_horizontal = cols / blocksize;
    int block_count_vertical = rows / blocksize;
    int block_remain_horizontal = cols % blocksize;
    int block_remain_vertical = rows % blocksize;

    #pragma omp parallel for 
    for(int x = 0; x <= block_count_horizontal; ++x){
        int xEnd = blocksize;
        if (x == block_count_horizontal) {
            xEnd = block_remain_horizontal;
        }
        int block_x = x * blocksize;
        for(int y = 0; y <= block_count_vertical; ++y){
            int yEnd = blocksize;
            if (y == block_count_vertical) {
                yEnd = block_remain_vertical;
            }
            int block_y = y * blocksize;
            for(int i = 0; i < xEnd; ++i) {
                int val_x = block_x + i;
                for(int j = 0; j < yEnd; ++j) {
                    int val_y = block_y + j;
                    dst[val_y  + val_x * rows] = src[val_x + val_y * cols];
                }
            }
        }
    }
}

// need more optimize
double *matrixSquareMul(int n, double *resultMat, double *matMatrix1, double *matMatrix2) {
    int blockSize = 16;
    int block_num = n / blockSize;
    int block_remain = n % blockSize;
    #pragma omp parallel for 
    for (int x = 0; x <= block_num; ++x) {
        int xEnd = blockSize;
        if (x == block_num) {
            xEnd = block_remain;
        }
        int block_x = x * blockSize;
        for (int y = 0; y <= block_num; ++y) {
            int yEnd = blockSize;
            if (y == block_num) {
                yEnd = block_remain;
            }
            int block_y = y * blockSize;
            for (int bx = 0; bx < xEnd; ++bx) {
                int i = block_x + bx;
                int in = i * n;
                for (int by = 0; by < yEnd; ++by) {
                    int j = block_y + by;
                    int jn = j * n;
                    __m256d cijVec = _mm256_setzero_pd();
                    for(int k = 0; k < n / 16 * 16; k+=16) {
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&(matMatrix1[jn + k])), _mm256_loadu_pd(&(matMatrix2[in + k])), cijVec);
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&(matMatrix1[jn + (k + 4)])), _mm256_loadu_pd(&(matMatrix2[in + (k + 4)])), cijVec);
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&(matMatrix1[jn + (k + 8)])), _mm256_loadu_pd(&(matMatrix2[in + (k + 8)])), cijVec);
                        cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&(matMatrix1[jn + (k + 12)])), _mm256_loadu_pd(&(matMatrix2[in + (k + 12)])), cijVec);
                    }
                    double tmp_arr[4];
                    _mm256_storeu_pd (tmp_arr, cijVec); 
                    double cij = tmp_arr[0] + tmp_arr[1] + tmp_arr[2] + tmp_arr[3];
                    for (int k = n / 16 * 16; k < n; ++k) {
                        // tail case
                        cij += matMatrix1[jn + k] * matMatrix2[in + k];
                    }
                    resultMat[jn + i] = cij;
                }
            }
        }
    }
    return resultMat;
}

double *matrixSquareMul_xyz(int n, double *resultMat, double *matMatrix1, double *matMatrix2) {
    int blockSize = 32;
    int block_num = n / blockSize;
    int block_remain = n % blockSize;
    #pragma omp parallel for 
    for (int index = 0; index < (n * n); index++) {
        resultMat[index] = 0;
    }
    #pragma omp parallel for 
    for (int x = 0; x <= block_num; x++) {
        int xEnd = blockSize;
        if (x == block_num) {
            xEnd = block_remain;
        }
        for (int y = 0; y <= block_num; y++) {
            int yEnd = blockSize;
            if (y == block_num) {
                yEnd = block_remain;
            }
            for (int z = 0; z <= block_num; z++) {
                int zEnd = blockSize;
                if (z == block_num) {
                    zEnd = block_remain;
                }
                for (int bx = 0; bx < xEnd; bx++) {
                    for (int by = 0; by < yEnd; by++) {
                        int i = x * blockSize + bx;
                        int j = y * blockSize + by;
                        __m256d cijVec = _mm256_setzero_pd();
                        for(int bz = 0; bz < zEnd / 4 * 4; bz+=4) {
                            int k = z * blockSize + bz;
                            cijVec = _mm256_fmadd_pd (_mm256_loadu_pd(&(matMatrix1[j*n + k])), _mm256_loadu_pd(&(matMatrix2[i*n + k])), cijVec);
                        }
                        double tmp_arr[4];
                        _mm256_storeu_pd (tmp_arr, cijVec); 
                        double cij = tmp_arr[0] + tmp_arr[1] + tmp_arr[2] + tmp_arr[3];
                        for (int bz = zEnd - (zEnd % 4); bz < zEnd; bz++) {
                            // tail case
                            int k = z * blockSize + bz;
                            cij += matMatrix1[j*n + k] * matMatrix2[i*n + k];
                        }
                        resultMat[j*n + i] += cij; // need += but will need calloc
                    }
                }
            }
        }
    }
    return resultMat;
}

/* Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // https://stackoverflow.com/questions/30253662/raising-a-2d-array-to-a-power-in-c
    /* TODO: YOUR CODE HERE */
    double *matrixSquareMul(int n, double *resultMat, double *matMatrix1, double *matMatrix2);
    if (mat->cols != mat->rows) {
        return -1; // incorrect dimension
    }
    int matSize = (mat->rows) * (mat->cols);
    int n =  mat->rows;
    double *power2 = (double *) calloc(matSize, sizeof(double));
    double *power22 = (double *) calloc(matSize, sizeof(double));
    double *matPre = (double *) calloc(matSize, sizeof(double));
    double *matAfter = (double *) calloc(matSize, sizeof(double));
    double *transpose = (double *) calloc(matSize, sizeof(double));
    #pragma omp parallel for 
    for (int index = 0; index < matSize; ++index) {
        power2[index] = mat->data[index];
    }
    // identity matrix 
    #pragma omp parallel for 
    for(int i=0; i < n; ++i) {
        matPre[i * n + i] = 1;
    }
    double *tmp;
    transpose_blocking(n, n, transpose, power2);
    while(pow != 0) {
        if(pow & 1) {
            tmp = matrixSquareMul(n, matAfter, matPre, transpose);
            matAfter = matPre;
            matPre = tmp;
        }
        tmp = matrixSquareMul(n, power22, power2, transpose);
        power22 = power2;
        power2 = tmp;
        transpose_blocking(n, n, transpose, power2);
        pow >>= 1;
    }
    #pragma omp parallel for
    for (int index = 0; index < matSize; ++index) {
        result->data[index] = matPre[index];
    }
    free(power2);
    free(power22);
    free(matAfter);
    free(matPre);
    free(transpose);
    return 0;   
}
/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix_naive(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    result->rows = mat->rows;
    result->cols = mat->cols;
    int matSize = (result->rows) * (result->cols);
    result->data = (double *) malloc(matSize * sizeof(double)); 
    if (result->data == NULL) {
        return -2; // fail to create double array
    }
    result->ref_cnt = 1;
    result->parent = NULL;

    for (int offset = 0; offset < matSize; offset++) {
        (result->data)[offset] = -(mat->data)[offset];
    }
    return 0;
}

int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    // this particular version implement both avx and parallel
    int matSize = (result->rows) * (result->cols);
    #pragma omp parallel for 
    for (int offset = 0; offset < matSize / 4 * 4; offset+=4) {
        __m256d matVec = _mm256_loadu_pd(&(mat->data[offset]));
        __m256d zeroVec = _mm256_setzero_pd();
        __m256d resultVec = _mm256_sub_pd (zeroVec, matVec);
        _mm256_storeu_pd (&(result->data[offset]), resultVec); 
    }
    #pragma omp parallel for 
    for(int offset = matSize / 4 * 4; offset < matSize; offset++) {
        // tail case
        (result->data)[offset] = -(mat->data)[offset];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix_naive(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    result->rows = mat->rows;
    result->cols = mat->cols;
    int matSize = (result->rows) * (result->cols);
    result->data = (double *) malloc(matSize * sizeof(double)); 
    if (result->data == NULL) {
        return -2; // fail to create double array
    }
    result->ref_cnt = 1;
    result->parent = NULL;
    
    for (int offset = 0; offset < matSize; offset++) {
        (result->data)[offset] = fabs((mat->data)[offset]);
    }
    return 0;
}


int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    // this particular version implement both avx and parallel
    int matSize = (mat->rows) * (mat->cols);     
    __m256d zeroVec = _mm256_setzero_pd();
    #pragma omp parallel for 
    for (int offset = 0; offset <  matSize / 4 * 4; offset+=4) {
        __m256d matVec = _mm256_loadu_pd(&(mat->data[offset]));
        __m256d zeroSubVec = _mm256_sub_pd (zeroVec, matVec);
        __m256d absVec = _mm256_max_pd (matVec, zeroSubVec);
        _mm256_storeu_pd (&(result->data[offset]), absVec); 
    }
    #pragma omp parallel for 
    for(int offset = matSize / 4 * 4; offset < matSize; offset++) {
    // tail case
        if ((mat->data)[offset] >= 0) {
            (result->data)[offset] = (mat->data)[offset];
        } else {
            (result->data)[offset] = -(mat->data)[offset];
        }
    }
    return 0;
}