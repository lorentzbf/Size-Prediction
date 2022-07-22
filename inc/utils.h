#ifndef _Z_UTILS_H_
#define _Z_UTILS_H_
#include <scalable_allocator.h>
#include "common.h"
#include "CSR.h"

namespace util{

template <typename T>
T sum(T *a, int n){
    T res = 0;
    for(int i = 0; i < n; i++){
        res += a[i];
    }
    return res;
}

template <typename T>
T mean(T *a, int n){
    return sum(a,n)/n;
}

template <typename T>
T max(T *a, int n){
    T res = a[0];
    for(int i = 1; i < n; i++){
        if(res < a[i]){
            res = a[i];
        }
    }
    return res;
}

template <typename T>
T min(T *a, int n){
    T res = a[0];
    for(int i = 1; i < n; i++){
        if(res > a[i]){
            res = a[i];
        }
    }
    return res;
}

}

template <typename T>
inline T* my_malloc(unsigned long array_size)
{
#ifdef CPP
    return (T *)(new T[array_size]);
#elif defined IMM
    return (T *)_mm_malloc(sizeof(T) * array_size, 64);
#elif defined TBB
    return (T *)scalable_malloc(sizeof(T) * array_size);
#else
    return (T *)scalable_malloc(sizeof(T) * array_size);
#endif
}

template <typename T>
inline void my_free(T *a)
{
#ifdef CPP
    delete[] a;
#elif defined IMM
    _mm_free(a);
#elif defined TBB
    scalable_free(a);
#else
    scalable_free(a);
#endif
}

long compute_flop(mint *arpt, mint *acol, mint *brpt, mint M, mint* nnzrA, mint *floprC){
    long total_flop = 0;
#pragma omp parallel
{
    int thread_flop = 0;
#pragma omp for
    for(mint i = 0; i < M; i++){
        int local_sum = 0;
        nnzrA[i] = arpt[i+1] - arpt[i];
        for(mint j = arpt[i]; j < arpt[i+1]; j++){
            local_sum += brpt[acol[j]+1] - brpt[acol[j]];
        }
        floprC[i] = local_sum;
        thread_flop += local_sum;
    }
#pragma omp critical
{
    total_flop += thread_flop;
}
}
    return total_flop;
}

long compute_flop(mint *arpt, mint *acol, mint *brpt, mint M, mint* nnzrA, mint *floprC, mint *num_threads){
    long total_flop = 0;
#pragma omp parallel
{
    int thread_flop = 0;
#pragma omp for
    for(mint i = 0; i < M; i++){
        int local_sum = 0;
        nnzrA[i] = arpt[i+1] - arpt[i];
        for(mint j = arpt[i]; j < arpt[i+1]; j++){
            local_sum += brpt[acol[j]+1] - brpt[acol[j]];
        }
        floprC[i] = local_sum;
        thread_flop += local_sum;
    }
#pragma omp critical
{
    total_flop += thread_flop;
}
#pragma omp single
{
    *num_threads = omp_get_num_threads();
}
}
    return total_flop;
}


long compute_flop(const CSR& A, const CSR& B){
    mint *nnzrA = new mint [A.M];
    mint *row_flop = new mint [A.M];
    long flop = compute_flop(A.rpt, A.col, B.rpt, A.M, nnzrA, row_flop);
    delete [] row_flop;
    delete [] nnzrA;
    return flop;
}


inline int upper_log(int num){
    // assert(num > 0);
    int a = num;
    int base = 0;
    while(a = a>>1){
        base++;
    }
    if(likely(num != 1<<base))
        return 1 << (base+1);
    else
        return 1 << base;
}

// Prefix sum (Sequential)
template <typename T1, typename T2>
void seq_scan(T1 *in, T2 *out, int N)
{
    out[0] = 0;
    for (int i = 0; i < N - 1; ++i) {
        out[i + 1] = out[i] + in[i];
    }
}

// Prefix sum (Thread parallel)
template <typename T1, typename T2>
void scan(T1 *in, T2 *out, int N)
{
    if (N < (1 << 17)) {
        seq_scan(in, out, N);
    }
    else {
        int tnum;
        // my modify, different from yusuke, not much difference in performance
        #pragma omp parallel
        {
            #pragma omp single
            {
                tnum = omp_get_num_threads();
            }
        }
        int each_n = N / tnum;
        //T2 *partial_sum = (T *)scalable_malloc(sizeof(T) * (tnum));
        T2 *partial_sum = new T2 [tnum];
#pragma omp parallel num_threads(tnum)
        {
            int tid = omp_get_thread_num();
            int start = each_n * tid;
            int end = (tid < tnum - 1)? start + each_n : N;
            out[start] = 0;
            for (int i = start; i < end - 1; ++i) {
                out[i + 1] = out[i] + in[i];
            }
            partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier

            int offset = 0;
            for (int i = 0; i < tid; ++i) {
                offset += partial_sum[i];
            }
            for (int i = start; i < end; ++i) {
                out[i] += offset;
            }
        }
        //out[N] = out[N-1] + in[N-1];
        //scalable_free(partial_sum);
        delete [] partial_sum;
    }
}

template <typename T1, typename T2>
void seq_prefix_sum(T1 *in, T2 *out, int N)
{
    out[0] = 0;
    for (int i = 0; i < N; ++i) {
        out[i + 1] = out[i] + in[i];
    }
}

template <typename T1, typename T2>
void para_prefix_sum(T1 *in, T2 *out, int N, int tnum){

    //printf("num threads %d\n", tnum);
    int each_n = N / tnum;
    //T2 *partial_sum = (T *)scalable_malloc(sizeof(T) * (tnum));
    T2 *partial_sum = new T2 [tnum];
#pragma omp parallel num_threads(tnum)
    {
        int tid = omp_get_thread_num();
        int start = each_n * tid;
        int end = (tid < tnum - 1)? start + each_n : N;
        out[start] = 0;
        for (int i = start; i < end - 1; ++i) {
            out[i + 1] = out[i] + in[i];
        }
        partial_sum[tid] = out[end - 1] + in[end - 1];
#pragma omp barrier
        int offset = 0;
        for (int i = 0; i < tid; ++i) {
            offset += partial_sum[i];
        }
        for (int i = start; i < end; ++i) {
            out[i] += offset;
        }
        if(tid == tnum - 1)
            out[N] = out[N-1] + in[N-1];
    }
    //scalable_free(partial_sum);
    delete [] partial_sum;
}

template <typename T1, typename T2>
inline void opt_prefix_sum(T1 *in, T2 *out, int N, int tnum = 64){
    if( N < 1 << 13){
        seq_prefix_sum(in, out, N);
    }
    else{
        para_prefix_sum(in, out, N, tnum);
    }
}

template <typename T>
int binary_search_approx(T *arr, int N, T elem){
    int s = 0;
    int e = N-1;
    assert(elem >= arr[0] && "find approx elem < smallest");
    if(elem >= arr[N-1]){
        return N-1;
    }
    while(true){
        int m = (s+e)/2;
        if(arr[m] == elem){
            return m;
        }
        else if(arr[m] < elem){
            if(elem < arr[m+1]){
                return m;
            }
            s = m;
        }
        else{ // elem < arr[m]
            if(arr[m-1] < elem){
                return m-1;
            }
            e = m;
        }
    }
}




double estimate_sample_CR(const int *row_pointer_A, const int* col_index_A, const int* row_pointer_B, const int* col_index_B, const int * floprC, int M, int num_threads, int sample_num  = 300){
    int *rand_index = new int [sample_num];
    srand(0);
    int max_flopr = 0;
    int total_flop = 0;
    for(int i = 0; i < sample_num; i++){
        //rand_index[i] = int(double(rand())*(1.0/double(RAND_MAX))*M);
        rand_index[i] = rand() % M;
        total_flop += floprC[rand_index[i]];
        if(unlikely(max_flopr < floprC[rand_index[i]])){
            max_flopr = floprC[rand_index[i]];
        }
    }
    float jobs = float(sample_num)/num_threads;
    max_flopr = upper_log(max_flopr);

    int total_nnz = 0;
#pragma omp parallel
{
    int rid = omp_get_thread_num();

    int start = rid * jobs;
    int end_t = (rid + 1) * jobs;
    int end = end_t < sample_num ? end_t : sample_num;

    int *ht = new int [max_flopr];
    int hash_size;
    int local_nnz = 0;
    for(int i = start; i < end; i++){
        int row = rand_index[i];
        hash_size = upper_log(floprC[row]);
        for(int i = 0; i < hash_size; i++){
            ht[i] = -1;
        }
        for(int index = row_pointer_A[row]; index < row_pointer_A[row + 1]; index++){// each element of A row
            int B_row = col_index_A[index];
            //local_flop += row_pointer_B[B_row + 1] - row_pointer_B[B_row];
            for(int B_index = row_pointer_B[B_row]; B_index <  row_pointer_B[B_row + 1]; B_index++){
                // insert col_index_B[B_index] to hash_table
                int hash = (col_index_B[B_index] * 107) & (hash_size - 1);
                while(1){
                    if(ht[hash] == col_index_B[B_index]) // already inserted
                        break;
                    else if(ht[hash] == -1) { // not inserted
                        ht[hash] = col_index_B[B_index];
                        local_nnz++;
                        break;
                    }
                    else{ // hash conflict
                        hash = (hash + 1) & (hash_size - 1);
                    }
                }
            }
        }
    }
    #pragma omp critical
    {
        //total_flop += local_flop;
        total_nnz += local_nnz;
    }
    delete [] ht;
}
    delete [] rand_index;
    return double(total_flop)/total_nnz;
}

double estimate_sample_density(const int *row_pointer_A, const int* col_index_A, const int* row_pointer_B, const int* col_index_B, const int * floprC, int M, int num_threads, int sample_num, int N){
    int *rand_index = new int [sample_num];
    srand(0);
    int max_flopr = 0;
    for(int i = 0; i < sample_num; i++){
        //rand_index[i] = int(double(rand())*(1.0/double(RAND_MAX))*M);
        rand_index[i] = rand() % M;
        if(unlikely(max_flopr < floprC[rand_index[i]])){
            max_flopr = floprC[rand_index[i]];
        }
    }
    float jobs = float(sample_num)/num_threads;
    max_flopr = upper_log(max_flopr);

    int total_nnz = 0;
#pragma omp parallel
{
    int rid = omp_get_thread_num();

    int start = rid * jobs;
    int end_t = (rid + 1) * jobs;
    int end = end_t < sample_num ? end_t : sample_num;

    int *ht = new int [max_flopr];
    int hash_size;
    int local_nnz = 0;
    for(int i = start; i < end; i++){
        int row = rand_index[i];
        hash_size = upper_log(floprC[row]);
        for(int i = 0; i < hash_size; i++){
            ht[i] = -1;
        }
        for(int index = row_pointer_A[row]; index < row_pointer_A[row + 1]; index++){// each element of A row
            int B_row = col_index_A[index];
            //local_flop += row_pointer_B[B_row + 1] - row_pointer_B[B_row];
            for(int B_index = row_pointer_B[B_row]; B_index <  row_pointer_B[B_row + 1]; B_index++){
                // insert col_index_B[B_index] to hash_table
                int hash = (col_index_B[B_index] * 107) & (hash_size - 1);
                while(1){
                    if(ht[hash] == col_index_B[B_index]) // already inserted
                        break;
                    else if(ht[hash] == -1) { // not inserted
                        ht[hash] = col_index_B[B_index];
                        local_nnz++;
                        break;
                    }
                    else{ // hash conflict
                        hash = (hash + 1) & (hash_size - 1);
                    }
                }
            }
        }
    }
    #pragma omp critical
    {
        //total_flop += local_flop;
        total_nnz += local_nnz;
    }
    delete [] ht;
}
    delete [] rand_index;
    return double(total_nnz)/(double(sample_num)*N);
}


class Two_long{
    public:
    int flop;
    int nnz;
    Two_long(int a, int b):flop(a), nnz(b){}
} ;

Two_long compute_sample(const int *row_pointer_A, const int* col_index_A, const int* row_pointer_B, const int* col_index_B, const int * floprC, int M, int num_threads, int sample_num  = 300){
    int *rand_index = new int [sample_num];
    srand(0);
    int max_flopr = 0;
    int total_flop = 0;
    for(int i = 0; i < sample_num; i++){
        //rand_index[i] = int(double(rand())*(1.0/double(RAND_MAX))*M);
        rand_index[i] = rand() % M;
        total_flop += floprC[rand_index[i]];
        if(unlikely(max_flopr < floprC[rand_index[i]])){
            max_flopr = floprC[rand_index[i]];
        }
    }
    float jobs = float(sample_num)/num_threads;
    max_flopr = upper_log(max_flopr);

    int total_nnz = 0;
#pragma omp parallel
{
    int rid = omp_get_thread_num();

    int start = rid * jobs;
    int end_t = (rid + 1) * jobs;
    int end = end_t < sample_num ? end_t : sample_num;

    //int *ht = new int [max_flopr];
    int *ht = my_malloc<int>(max_flopr);
    int hash_size;
    int local_nnz = 0;
    for(int i = start; i < end; i++){
        int row = rand_index[i];
        hash_size = upper_log(floprC[row]);
        for(int i = 0; i < hash_size; i++){
            ht[i] = -1;
        }
        for(int index = row_pointer_A[row]; index < row_pointer_A[row + 1]; index++){// each element of A row
            int B_row = col_index_A[index];
            //local_flop += row_pointer_B[B_row + 1] - row_pointer_B[B_row];
            for(int B_index = row_pointer_B[B_row]; B_index <  row_pointer_B[B_row + 1]; B_index++){
                // insert col_index_B[B_index] to hash_table
                int hash = (col_index_B[B_index] * 107) & (hash_size - 1);
                while(1){
                    if(ht[hash] == col_index_B[B_index]) // already inserted
                        break;
                    else if(ht[hash] == -1) { // not inserted
                        ht[hash] = col_index_B[B_index];
                        local_nnz++;
                        break;
                    }
                    else{ // hash conflict
                        hash = (hash + 1) & (hash_size - 1);
                    }
                }
            }
        }
    }
    #pragma omp critical
    {
        //total_flop += local_flop;
        total_nnz += local_nnz;
    }
    //delete [] ht;
    my_free(ht);
}
    delete [] rand_index;
    Two_long two_long(total_flop, total_nnz);
    return two_long;
    // return double(total_flop)/total_nnz;
}


#endif
