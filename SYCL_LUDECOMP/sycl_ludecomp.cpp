#include "sycl_ludecomp.hpp"

void SYCL_ENABLE::GaussLU::solve(sycl::queue Q, double *matIn,int rows, int cols){


    {// SYCL_SCOPE

        sycl::buffer<double, 1> buffIn{matIn,cl::sycl::range<1>{static_cast<size_t>(rows*(cols+1))}};

        for(int k=0; k<rows-1; k++){
            Q.submit([&](sycl::handler &handler){
                auto mat_acc = buffIn.get_access<sycl::access::mode::read_write>(handler);
                handler.parallel_for<decompose>(sycl::nd_range<2>(sycl::range<2> {static_cast<size_t>(rows-(k+1)),static_cast<size_t>(cols-k)},sycl::range<2> {1,1}),decompose(k,mat_acc, rows, cols)           
                );

            });
        }

    }//END SYCL SCOPE
}
void SYCL_ENABLE::GaussLU::buildLinearSystem(sycl::queue Q, double *matIn,double *matTot,double *sourceVec,int rows, int cols){
    
    {//SYCL SCOPE

        sycl::buffer<double, 1> buffIn{matIn,cl::sycl::range<1>{static_cast<size_t>(rows*cols)}};
        sycl::buffer<double, 1> buffbvec{sourceVec,cl::sycl::range<1>{static_cast<size_t>(rows)}};
        //Buffer for the augmented matrix
        sycl::buffer<double, 1> buffOut{matTot,cl::sycl::range<1>{static_cast<size_t>(rows*(cols+1))}};

        Q.submit([&](sycl::handler &handler){

            auto inMatAcc =  buffIn.get_access<sycl::access::mode::read>(handler);
            auto outMatAcc =  buffOut.get_access<sycl::access::mode::write>(handler);
            auto bvecAcc=  buffbvec.get_access<sycl::access::mode::read>(handler);


            handler.parallel_for<assamble>(cl::sycl::nd_range<2>(cl::sycl::range<2> {static_cast<size_t>(rows),static_cast<size_t>(cols)},cl::sycl::range<2> {1,1}), assamble(inMatAcc,outMatAcc,bvecAcc,rows,cols)



            );


        });





        
    }


}
void SYCL_ENABLE::GaussLU::getXvec(sycl::queue Q, double *matIn,double *xvec,int rows, int cols){
    
    // SYCL Gauss Elimination 
    for(int i=0; i<rows; i++){
        xvec[i] = matIn[i*(cols+1) + rows];
    }
    //SYCL_ENABLE::GaussLU::solve(Q,matIn,rows, cols);
    
    xvec[rows-1] = xvec[rows-1]/matIn[((cols+1)*rows)-2];
    {//SYCL SCOPE
         sycl::buffer<double, 1> buffMat{matIn,cl::sycl::range<1>{static_cast<size_t>(rows*(cols+1))}};
         sycl::buffer<double, 1> buffVec{xvec,cl::sycl::range<1>{static_cast<size_t>(rows)}};
        for(int k=rows-2; k>=0; k--){
            int amount = rows-(k+1);
            double aux[amount]; 
            sycl::buffer<double, 1> buffAux{aux,cl::sycl::range<1>{static_cast<size_t>(amount)}};
            Q.submit([&](sycl::handler &handler){
                auto mat_acc = buffMat.get_access<sycl::access::mode::read>(handler);
                auto vec_acc = buffVec.get_access<sycl::access::mode::read_write>(handler);
                auto aux_acc = buffAux.get_access<sycl::access::mode::discard_write>(handler);  
                
                handler.parallel_for<backsub>(sycl::range<1>{static_cast<size_t>(amount)}, backsub(k,vec_acc,aux_acc, mat_acc, rows, cols)
                );

            });
            

        }
        
    }// END SYCL SCOPE
    /* for(int i=0; i<amount; i++){
        printf("mult = %lf\n", aux[i]);
    } */
}
