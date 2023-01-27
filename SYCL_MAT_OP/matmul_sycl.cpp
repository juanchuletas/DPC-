#include<CL/sycl.hpp>
#include<iostream>
#include<cmath>
//using namespace sycl;
//void matMult(double *A, int rowsA,int colsA, double *B,int rowsB, int colsB,double *C);
typedef cl::sycl::accessor<double,1,cl::sycl::access::mode::write,cl::sycl::access::target::global_buffer> syclAcc;

class matMul{
    syclAcc cmat,amat,bmat;
    int rowsB,rowsA, colsA,colsB;

    public:
        matMul(syclAcc inputA, int _rowsA, int _colsA,syclAcc inputB, int _rowsB, int _colsB, syclAcc inputC)
        : amat{inputA}, bmat{inputB},cmat{inputC},rowsB{_rowsB},rowsA{_rowsA},colsA{_colsA},colsB{_colsB}
        {

        }
        void operator()(cl::sycl::nd_item<2> it)const{
            int i = it.get_global_id(0); //i
            int j = it.get_global_id(1); //j
            cmat[i*colsA+j] = 0.0;
             for(int k=0; k<rowsB; k++){
                cmat[i*colsA+j] += amat[i*colsA + k]*bmat[j*colsB+k];
            }
        }


}; 
void syclMatMul(double *A, int rowsA,int colsA, double *B,int rowsB, int colsB,double *C){

    auto Q = cl::sycl::queue{cl::sycl::gpu_selector{}};
    std::cout << "Chosen device: "  << Q.get_device().get_info<cl::sycl::info::device::name>()<<std::endl;
    std::cout<< "Max Work Group Size: "<< Q.get_device().get_info<cl::sycl::info::device::max_work_group_size>()<<std::endl;
    {
        cl::sycl::buffer<double, 1> buffc{C,cl::sycl::range<1>{static_cast<size_t>(rowsA*colsB)}};
        cl::sycl::buffer<double, 1> buffa{A,cl::sycl::range<1>{static_cast<size_t>(rowsA*colsA)}};
        cl::sycl::buffer<double, 1> buffb{B,cl::sycl::range<1>{static_cast<size_t>(rowsB*colsB)}};
        int *data = cl::sycl::malloc_shared<int>(3,Q);
         Q.submit([&](cl::sycl::handler &cgh){
            
            auto acc_matC = buffc.get_access<cl::sycl::access::mode::write>(cgh);
            auto acc_matA = buffa.get_access<cl::sycl::access::mode::write>(cgh);
            auto acc_matB = buffb.get_access<cl::sycl::access::mode::write>(cgh);
           cgh.parallel_for<matMul>(cl::sycl::nd_range<2>(cl::sycl::range<2> {static_cast<size_t>(rowsA),static_cast<size_t>(colsB)},cl::sycl::range<2> {1,1}), matMul(acc_matA,rowsA,colsA,acc_matB,rowsB,colsB,acc_matC)
           
           
           
           );
        });

    }
}
int main(){
    //SYCL_DEVICE_FILTER=PI_CUDA
    /* auto Q = cl::sycl::queue{cl::sycl::gpu_selector{}};
    std::cout << "Chosen device: "  << Q.get_device().get_info<cl::sycl::info::device::name>()<<std::endl;
    std::cout<< "Max Work Group Size: "<< Q.get_device().get_info<cl::sycl::info::device::max_work_group_size>()<<std::endl; */
    int M=5,N=5,P=3;	
    double B[]={1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1,
                1,1,1,1,1};
    double A[]={1,1,1,1,1,
              2,2,2,2,2,
              1,1,1,1,1};
    double C[P*M];
	
    syclMatMul(A,P,N,B,N,M,C);



    
    for(int i=0; i<P; i++){
        for(int j=0; j<N; j++){
            printf("%lf ",C[i*N+j]);
        }
        printf("i\n");
    }


}   