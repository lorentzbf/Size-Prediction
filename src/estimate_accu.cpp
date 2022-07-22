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
    long total_flop = compute_flop(A.rpt, A.col, B.rpt, M, nnzrA, floprC);

    mint sample_num = std::min(int(0.003*M), 300);

    CSR C;
    brmerge_precise(A, B, C);
    
    mint total_nnz = C.nnz;
    assert(total_nnz > 0 && "main line:75, nnz out of range");

    Two_long two_long = compute_sample(A.rpt, A.col, B.rpt, B.col, floprC, M, num_threads, sample_num);
    long sample_flop = two_long.flop;
    long sample_nnz = two_long.nnz;

    long estimate_nnz = double(sample_nnz) * M / sample_num;
    long estimate_flop = double(sample_flop) * M / sample_num;
    double rela_nnz = 100*(estimate_nnz - total_nnz)/double(total_nnz);
    double rela_flop = 100*(estimate_flop - total_flop)/double(total_flop);

    long estimate_nnz2 = double(sample_nnz)/double(sample_flop) * total_flop;
    double rela_nnz2 = 100*(estimate_nnz2 - total_nnz)/double(total_nnz);

    double CR = double(total_flop)/total_nnz;

    printf("%s %s %d %lf %d %lf %lf %lf %lf %lf %lf\n", mat1.c_str(), mat2.c_str(), sample_num, CR, total_nnz, rela_nnz, rela_flop, rela_nnz2, std::abs(rela_nnz), std::abs(rela_flop), std::abs(rela_nnz2));



}


