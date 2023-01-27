#include "backsub.hpp"
//*****************************************************************************
// *********** BACK SUBSTITUTION CLASS ****************************************


backsub::backsub(int index, syclAccRW vecAcc,syclAccDW auxAcc, syclAccR matAcc, int _rows, int _cols)
:k{index},xvec{vecAcc},store{auxAcc},mat{matAcc} ,rows{_rows},cols{_cols+1}{

}
void backsub::operator()(sycl::item<1> it) const{

    // THE VALUE OF N IS THE AMOUNT OF WORK
    int j=it[0];
    int start = it[0];
    int end = it.get_range(0);
    store[j] =  mat[k*cols + (k + j+1)]*xvec[(k+j+1)];
    //store[j] =  mat[k*cols + (k + j+1)];
    double sum = xvec[k];
    if(it[0]==0){
        for(int l = start; l<end; l++ ){
            xvec[k]  = xvec[k] - store[l]; 
        }
        xvec[k] = xvec[k]/mat[k*cols+k];
    }
    
   
    //xvec[indx] = indx;


}