#include<iostream>
#include "sycl_ludecomp.hpp"
#define N 5
#define NRHS 1
#define LDA N
#define LDB N

/* DGESV prototype */
extern "C" void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv,
                double* b, int* ldb, int* info );

void lapack(double *matA, double *vec, int cols, int rows){


     /* Locals */
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
    /* Local arrays */
    int ipiv[N];

    /* Solve the equations A*X = B */
    dgesv_( &n, &nrhs, matA, &lda, ipiv, vec, &ldb, &info );
    /* Check for the exact singularity */
    if( info > 0 ) {
            printf( "The diagonal element of the triangular factor of A,\n" );
            printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
            printf( "the solution could not be computed.\n" );
            exit( 1 );
    }
    printf("\x1B[33m\n\n******* Solution vector LAPACK: *******\033[0m\n");
     printf("\n");
    for(int i=0; i<rows; i++){
        printf("%lf  ",vec[i]);}
    printf("\x1B[32m\n\n********************************\033[0m\n");

}


int main (){

    int rows = 5; 
    int cols = 5; 
    double matrix[] = {3,2,4,2,-3,1,1,1,2};
    double vec[3] = {4,2,3};
    double matA[25]={2.,1.,1.,3.,2.,
    1.,2.,2.,1.,1.,
    1.,2.,9.,1.,5.,
    3.,1.,1.,7.,1.,
    2.,1.,5.,1.,8.};
    double bvec[5]={-2.,4.,3.,-5.,1.};
    //double *matrix = new double [rows*cols];

   /*   for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            //matrix[i*cols+j] = 0.0;  
            printf("%lf ",matrix[i*cols+j]);
        }
        printf("\n");
    } */
    double lu_mat[rows*(cols+1)];
    double sol[rows];

    printf("\x1B[32m\n\n******** SYCL DEVICE INFO *******\033[0m\n");
    printf("\n");
    auto Q = cl::sycl::queue{cl::sycl::gpu_selector{}};
    printf("\x1B[33mChosen device:\033[0m");
    std::cout << " "  << Q.get_device().get_info<cl::sycl::info::device::name>()<<std::endl;
    std::cout<< "Max Work Group Size: "<< Q.get_device().get_info<cl::sycl::info::device::max_work_group_size>()<<std::endl;
    std::cout<< "Max Compute Units: "<< Q.get_device().get_info<cl::sycl::info::device::max_compute_units>()<<std::endl;
    std::cout<< "Max work-group size: "<< Q.get_device().get_info<cl::sycl::info::device::max_work_group_size>()<<std::endl;
    auto d = Q.get_device();
    auto p = d.get_platform();
    std::cout << "SYCL Platform: " << p.get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "SYCL Device:   " << d.get_info<sycl::info::device::name>() << std::endl;

    SYCL_ENABLE::GaussLU::buildLinearSystem(Q, matA,lu_mat,bvec,rows,cols);
    printf("\x1B[32m\n\n********** Linear System *******\033[0m\n");
    printf("\n");
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols+1; j++){  
            printf("% lf ",lu_mat[i*(cols+1)+j]);
        }
        printf("\n");
    }
    SYCL_ENABLE::GaussLU::solve(Q, lu_mat, rows, cols); 

    SYCL_ENABLE::GaussLU::getXvec(Q,lu_mat,sol,rows, cols); 
    printf("\x1B[32m\n\n********** SYCL LU DECOMP *******\033[0m\n");
    printf("\n");
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols+1; j++){  
            printf("% lf ",lu_mat[i*(cols+1)+j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\x1B[33m\n\n***** Solution vector funQC::SYCL: *****\033[0m\n");
    printf("\n");
    for(int i=0; i<rows; i++){
        printf("%lf  ",sol[i]);}
    printf("\n");
    printf("\x1B[32m\n\n********************************\033[0m\n");
    //delete []  matrix ; 
    lapack(matA,bvec,rows,cols);

}