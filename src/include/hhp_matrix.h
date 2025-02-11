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

CSC CSR_to_CSC(const CSR *in);


SPLIT_CSR* cleanSplit(CSR in, int* partvec);
SPLIT_CSR* sparseSplit(CSR big, int* partvec);
SPLIT_CSR combineSplit(const SPLIT_CSR in);

#endif /* B07E3622_7549_4990_91E6_440BF5799438 */
