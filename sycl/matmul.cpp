#include<iostream>


void MatrixProduct(double *mat_A,double *mat_B, double *mat_C,int N,int M, int P)
{
  int i, j, l;
  double suma;
  int renglones_1 = N;
  int renglones_2 = M;
  int columnas_2 = P;
  for (i = 0; i < renglones_1; i++){
    for (j = 0; j < columnas_2; j++) {
      suma = 0.f;
      for (l = 0; l < renglones_2; l++){
        suma = suma + mat_A[i*renglones_1 + l]*mat_B[j*N + l];
	      printf("sum[%d]= A[%d]*B[%d]\n",i+j*N,i*N + l,j*N+l);
      }
      mat_C[i + j*N] = suma;
     printf("C[%d]=%lf\n",i+j*N,suma);
    }
   }
}
void matMult(double *A, int rowsA,int colsA, double *B,int rowsB, int colsB,double *C){

  double sum;
  for(int i=0; i<rowsA; i++){
    for(int j=0; j<colsB; j++){
      C[i*colsA+j] = 0.0;
      printf("C[%d]\n",i*colsA+j);
      for(int k=0; k<rowsB; k++){
        printf("sum[%d]= A[%d]*B[%d]\n",i*colsA+j,  i*colsA + k,   j*colsB+k);
        C[i*colsA+j] += A[i*colsA + k]*B[j*colsB+k];
      }
      
    }
  }


}
int main (){
	
	int M=4,N=4,P=4;	
	double B[]={1,1,1,1,
              1,1,1,1,
              1,1,1,1,
              1,1,1,1};
	double A[]={1,1,1,1,
              2,2,2,2,
              1,1,1,1,
              2,2,2,2};
	double C[P*M];
	
	//MatrixProduct(A,B,C,N,M,P);
  matMult(A,P,N,B,M,N,C);
	for(int i=0; i<P; i++){
		for(int j=0; j<N; j++){
			printf("%lf ",C[i*N+j]);
		}
		printf("\n");
	}

	return 0;


}
