#include "spgemm.h"
#include "mkl_spgemm.h"
int main(int argc, char **argv)
{
    std::string mat1, mat2;
    mat1 = "can_24";
    mat2 = "can_24";
    if(argc == 2){
        mat1 = argv[1];
        mat2 = argv[1];
    }
    if(argc >= 3){
        mat1 = argv[1];
        mat2 = argv[2];
    }
    std::string mat1_file;
    if(mat1.find("ER") != std::string::npos){
        mat1_file = "./matrix/ER/" + mat1 +".mtx";
    }
    else if(mat1.find("G500") != std::string::npos){
        mat1_file = "./matrix/G500/" + mat1 +".mtx";
    }
    else{
        mat1_file = "./matrix/suite_sparse/" + mat1 + "/" + mat1 +".mtx";
    }
    std::string mat2_file;
    if(mat2.find("ER") != std::string::npos){
        mat2_file = "./matrix/ER/" + mat2 +".mtx";
    }
    else if(mat2.find("G500") != std::string::npos){
        mat2_file = "./matrix/G500/" + mat2 +".mtx";
    }
    else{
        mat2_file = "./matrix/suite_sparse/" + mat2 + "/" + mat2 +".mtx";
    }
	
    CSR A, B;
    A.construct(mat1_file);
    if(mat1 == mat2){
        B = A;
    }
    else{
        B.construct(mat2_file);
        if(A.N == B.M){
            // do nothing
        }
        else if(A.N < B.M){
            CSR tmp(B, A.N, B.N, 0, 0);
            B = tmp;
        }
        else{
            CSR tmp(A, A.M, B.M, 0, 0);
            A = tmp;
        }
    }

    mint num_threads = 64;
    if(argc >= 4){
        num_threads = atoi(argv[3]);
    }
    omp_set_num_threads(num_threads);

    mint M = A.M;
    mint N = B.N;
    mint *nnzrA = new int [M];
    mint *floprC = new int [M];


    double t0, t1, t2, t3;
    int iter = 10;
    t0 = t1 = fast_clock_time();
    
    long total_flop = compute_flop(A.rpt, A.col, B.rpt, M, nnzrA, floprC);

    t1 = fast_clock_time();
    for(int i = 0; i < iter; i++){
        total_flop = compute_flop(A.rpt, A.col, B.rpt, M, nnzrA, floprC);
    }
    t1 = (fast_clock_time() - t1)/iter;
    //printf("compute flop %le\n", t1);

    mint sample_num = std::min(int(0.003*M), 300);
    Two_long two_long = compute_sample(A.rpt, A.col, B.rpt, B.col, floprC, M, num_threads, sample_num);
    t2 = fast_clock_time();
    for(int i = 0; i < iter; i++){
        two_long = compute_sample(A.rpt, A.col, B.rpt, B.col, floprC, M, num_threads, sample_num);
    }
    t2 = (fast_clock_time() - t2)/iter;
    //printf("compute sample %le\n", t2);


    CSR C;
    brmerge_precise(A, B, C);
    t3 = 0;
    for(int i = 0; i < iter; i++){
        t0 = fast_clock_time();
        brmerge_precise(A, B, C);
        t3 += fast_clock_time() - t0;
        C.release();
    }
    t3 /= iter;
    //printf("spgemm %le\n", t3);

    //printf("GFLOPS %lf\n", double(total_flop)*2/1000000000/t3);
    printf("%s %s %le %le %le %lf %lf %lf\n", mat1.c_str(), mat2.c_str(), t1, t2, t3, double(total_flop)*2/1000000000/t3, t1/t3*100, t2/t3*100);

    
    mint total_nnz = C.nnz;
    assert(total_nnz > 0 && "main line:78, nnz out of range");

}


