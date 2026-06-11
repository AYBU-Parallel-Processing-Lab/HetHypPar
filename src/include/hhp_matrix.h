#ifndef B07E3622_7549_4990_91E6_440BF5799438
#define B07E3622_7549_4990_91E6_440BF5799438

#include <hhp_common.h>

CSC ReadSparseMatrix(char *fname);
int *CalcPartVec(int nparts, const CSC *cscmatrix, const CSR *csrmatrix,char * fName, double final_imbal, int seed , char* resultFName);

CSR buReadSparseMatrix(char* fname);

// Assume the COO is Column Sorted.
//Note that this function allocates memory.
//It can be freed using freeSparseMatrix.
CSC COO_to_CSC(const COO *in);

COO CSR_to_COO(const CSR *in);

CSC CSR_to_CSC(CSR csr);

// Convert a SHARD-convention CSC (.I = column pointers [n+1], .J = row
// indices [nnz]) to CSR (.I = row pointers [m+1], .J = column indices [nnz]).
// Allocates; free with freeSparseMatrix. Used to enable row-parallel local SpMV.
CSR SHARD_loc_CSC_to_CSR(CSC in);

SHARD_CSC MPI_CSR_split_row(CSR big, iVector partvec);

void SHARD_CSC_destroy(SHARD_CSC *A);

#endif /* B07E3622_7549_4990_91E6_440BF5799438 */
