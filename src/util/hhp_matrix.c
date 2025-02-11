#include "mmio.h"
#include "patoh.h"
#include "bora_spmxv.h"
#include "stdio.h"
#include <stdlib.h>
#include <time.h> 
#include "math.h"
#include "stdint.h"
#include <sys/types.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_wrap_bora.h"
#include "hhp_util.h"

// improved via claude.ai
static inline CSR CSC_to_CSR(const CSC *in) {
    CSR out = {
        .I = (int *)malloc(sizeof(int) * (in->m + 1)),
        .J = (int *)malloc(sizeof(int) * in->nnz),
        .val = (double *)malloc(sizeof(double) * in->nnz),
        .nnz = in->nnz,
        .m = in->m,
        .n = in->n
    };
    if (out.I == NULL || out.J == NULL || out.val == NULL) {
        freeSparseMatrix(&out);
        return (CSR){0};
    }

    // Initialize row counts
    int *row_counts = (int *)calloc(in->m, sizeof(int));
    if (row_counts == NULL) {
        freeSparseMatrix(&out);
        return (CSR){0};
    }

    // Count entries in each row
    for (int i = 0; i < in->nnz; i++) {
        row_counts[in->I[i]]++;
    }

    // Calculate row pointers
    out.I[0] = 0;
    for (int i = 0; i < in->m; i++) {
        out.I[i + 1] = out.I[i] + row_counts[i];
        row_counts[i] = 0; // Reset for use as current index in each row
    }

    // Fill in column indices and values
    for (int col = 0; col < in->n; col++) {
        for (int j = in->J[col]; j < in->J[col + 1]; j++) {
            int row = in->I[j];
            int index = out.I[row] + row_counts[row];
            out.J[index] = col;
            out.val[index] = in->val[j];
            row_counts[row]++;
        }
    }

    free(row_counts);
    return out;
}

CSC CSR_to_CSC(const CSR *in) {
    CSC out = {
        .I = (int *)malloc(sizeof(int) * in->nnz),
        .J = (int *)malloc(sizeof(int) * (in->n + 1)),
        .val = (double *)malloc(sizeof(double) * in->nnz),
        .nnz = in->nnz,
        .m = in->m,
        .n = in->n
    };
    if (out.I == NULL || out.J == NULL || out.val == NULL) {
        freeSparseMatrix(&out);
        return (CSC){0};
    }

    // Initialize column counts
    int *col_counts = NULL;
    CALLOC_ARRAY(col_counts, in->n);
    if (col_counts == NULL) {
        freeSparseMatrix(&out);
        return (CSC){0};
    }

    // Count entries in each column
    for (int i = 0; i < in->nnz; i++) {
        col_counts[in->J[i]]++;
    }

    // Calculate column pointers
    out.J[0] = 0;
    for (int i = 0; i < in->n; i++) {
        out.J[i + 1] = out.J[i] + col_counts[i];
        col_counts[i] = 0; // Reset for use as current index in each column
    }

    // Fill in row indices and values
    for (int row = 0; row < in->m; row++) {
        for (int j = in->I[row]; j < in->I[row + 1]; j++) {
            int col = in->J[j];
            int index = out.J[col] + col_counts[col];
            out.I[index] = row;
            out.val[index] = in->val[j];
            col_counts[col]++;
        }
    }

    FREE_AND_NULL(col_counts);
    return out;
}

COO CSR_to_COO(const CSR *in) {
  COO out = {
    .nnz = in->nnz,
    .m = in->m,
    .n = in->n
  };
  ALLOC_ARRAY(out.I, in->nnz);
  ALLOC_ARRAY(out.J, in->nnz);
  ALLOC_ARRAY(out.val, in->nnz);

  for (int i =0; i < out.m; i++){
    for (int j =in->I[i]; j < in->I[i+1]; j++) {
      out.I[j] = i;
      out.J[j] = in->J[j];

      if ((out.J[j] >= out.m) || (out.J[j] > out.n)){
        DEBUGLOG("out.J[j (%d)] (%d) is bigger than out.m or out.n", j, out.J[j]);
      }

      out.val[j] = in->val[j];
    }
  }

  return out;
}

CSC COO_to_CSC(const COO *in) {
  CSC out = {.I = (int *)malloc(sizeof(int) * in->nnz),
             .J = (int *)malloc(sizeof(int) * (in->n + 1)),
             .val = (double *)malloc(sizeof(double) * in->nnz),
             .nnz = in->nnz,
             .m = in->m,
             .n = in->n};
  if (out.I == NULL || out.J == NULL || out.val == NULL) {
    return (CSC){0};
  }
  out.J[0] = 0;
  out.J[out.n] = out.nnz;
  int i, col;
  for (i = 0, col = 0; i < out.nnz; i++) {
    if (in->J[i] != col)
      out.J[col++ + 1] = i;
    out.I[i] = in->I[i];
    out.val[i] = in->val[i];
  }
  int last = out.J[out.n];
  if (last != out.nnz) {
    out.J[out.n] = out.nnz;
  }
  return out;
}

CSR buReadSparseMatrix(char* fname){
  buMatrix bigA;

  readMatrixMarketFromFile(&bigA, fname);

  CSR out = {
    bigA.ia,
    bigA.ja,
    bigA.val,
    bigA.nnz,
    bigA.gm,
    bigA.gn
  };

  free(bigA.inPart);
  free(bigA.outPart);

  return out;
}

CSC ReadSparseMatrix(char *fname) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int M, N, nz;

  if ((f = fopen(fname, "r")) == NULL) {
    fprintf(stderr, "Could not open matrix file\n");
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  if (mm_is_complex(matcode) | !mm_is_matrix(matcode) | !mm_is_sparse(matcode) |
      mm_is_symmetric(matcode)) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
    exit(1);
  }

  /* reseve memory for matrices */

  COO smatrix = {.I = (int *)malloc(nz * sizeof(int)),
                 .J = (int *)malloc(nz * sizeof(int)),
                 .val = (double *)malloc(nz * sizeof(double)),
                 .nnz = nz,
                 .m = M,
                 .n = N};

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (int i = 0; i < nz; i++) {
    if (fscanf(f, "%d %d %lg\n", &smatrix.I[i], &smatrix.J[i],
               &smatrix.val[i]) != 3) {
      fprintf(stderr, "Error reading matrix\n");
      exit(EXIT_FAILURE);
    }
    smatrix.I[i]--; /* adjust from 1-based to 0-based */
    smatrix.J[i]--;
  }
  fclose(f);

  CSC cscmatrix = {0};
  if ((cscmatrix = COO_to_CSC(&smatrix)).nnz == 0)
    exit(EXIT_FAILURE);
  freeSparseMatrix(&smatrix);

  return cscmatrix;
}

static void nnzDistribution(int *cweights, int numpart, int nrow, int *partvec,char* resultFName) {

  int nnzCount[numpart];
  for (int i = 0; i < numpart; i++)
    nnzCount[i] = 0;

  for (int i = 0; i < nrow; i++) {

    nnzCount[partvec[i]] += cweights[i];
  }

 //-----------------------Result file ----------------------------------   
  FILE* fptr = fopen(resultFName,"a") ; 

  if (fptr == NULL) {
     printf("Result file cannot opened!!") ;
     exit(-1) ;
  }
  else  {

    time_t mytime;
    mytime = time(NULL) ;
    
    fprintf(fptr,"%s\n",ctime(&mytime)) ;
    fprintf(fptr,"Number of Non-zero Count : \n") ;
    for (int i = 0; i < numpart; i++) {
        printf("[%d] %d ", i, nnzCount[i]);
        fprintf(fptr," [%d] : %d ",i,nnzCount[i]) ;
    }
  }
  fclose(fptr) ;
  //--------------------- Ending Result file -----------------------------
  printf("\n");
}

int *CalcPartVec(int nparts, const CSC *cscmatrix, const CSR *csrmatrix,char * fName, double final_imbal, int seed , char* resultFName) {

  CSC csctmatrix = SparseTranspose(*cscmatrix); /* test needed */
  int *cweights = CalcWeights(&csctmatrix);

  // PATOH STARTS HERE
  // =========================================================================================================

  int nweights[csrmatrix->n];
  for (int i = 0; i < csrmatrix->n; i++)
    nweights[i] = 1;

  PaToH_Parameters args = {0};
  PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);
  args._k = nparts;
  
  // =========================================================================================================
  // Final imbalance and counstant seed  
  args.final_imbal = final_imbal ; 
  args.seed = seed ;
  // =========================================================================================================

  int *partvec =
      malloc(sizeof(int) * csrmatrix->n); // contains the resulting partition
  int partweights[args._k]; // contains the weights of the partition
  int cut;

  float targetweigths[nparts] ; 


  // Open a file in read mode
  FILE *fptr;
  if((fptr = fopen(fName, "r")) == NULL) {
    fprintf(stderr, "file can't be opened %s file \n",fName) ;
    exit(-1) ;
  }
  
  int i=0 ;
  while (fscanf(fptr, "%f ",targetweigths + i++)  == 1) ;
        
  fclose(fptr) ;
  
  
  PaToH_Alloc(&args, csrmatrix->n, csrmatrix->m, 1, cweights, nweights,
              csrmatrix->I, csrmatrix->J);

  PaToH_Part(&args, csrmatrix->n, csrmatrix->m, 1, 0, cweights, nweights,
             csrmatrix->I, csrmatrix->J, targetweigths, partvec, partweights,
             &cut);

  nnzDistribution(cweights, nparts, cscmatrix->m, partvec,resultFName); // print nnz per part 
                                                        
  
  
  printf("cut : %d",cut) ;
  
  // --------------------------- open result file for write cut size -----------------------
   FILE* file = fopen(resultFName,"a") ; 

  if (file == NULL) {
     printf("Result file cannot opened!!") ;
     exit(-1) ;
  }
  else  {
    fprintf(file,"\n\ncut  : %d \n",cut) ;
  }
  fclose(file) ; 
   
  // --------------------------- closing file ---------------------------------------------- 




  free(cweights);
  PaToH_Free();
  // PATOH ENDS HERE
  // ============================================================================================
  return partvec;
}


// ==============================================================
// ==============================================================
// Split matrix
// ==============================================================
// ==============================================================




//Taken From G2G
// A iterative binary search function. It returns location
// of x in given array arr[l..r] if present, otherwise -1
static int binarySearch(int arr[], int l, int r, int x)
{
    // the loop will run till there are elements in the
    // subarray as l > r means that there are no elements to
    // consider in the given subarray
    while (l <= r) {
 
        // calculating mid point
        int m = l + (r - l) / 2;
 
        // Check if x is present at mid
        if (arr[m] == x) {
            return m;
        }
 
        // If x greater than ,, ignore left half
        if (arr[m] < x) {
            l = m + 1;
        }
 
        // If x is smaller than m, ignore right half
        else {
            r = m - 1;
        }
    }
 
    // if we reach here, then element was not present
    return -1;
}


static inline uint32_t* boolArray_new(int nItems){
  uint32_t* res;
  int nelems = ceil(nItems/32.0);
  CALLOC_ARRAY(res, nelems);
  return res;
}

static inline int boolArray_get(uint32_t* array, int index){
  return (array[index/32]) & (1 << (index % 32));
}

static inline void boolArray_set(uint32_t* array, int index, int val){
  uint32_t mask = -1;
  mask ^= 1 << (index % 32);
  array[index/32] &= mask;
  array[index/32] |= (val) << (index % 32);
}

// find ~a1 & a2
// store result into a2
static inline void boolArray_diff(uint32_t* a1, uint32_t* a2, int nItems){
  int nelems = ceil(nItems/32.0);
  for (size_t i = 0; i < nelems; i++)
  {
    a2[i] &= ~(a1[i]);
  }
}

static inline int boolArray_isEmpty(uint32_t* array, int nItems){
  int flag = 1;
  int nelems = ceil(nItems/32.0);
  for (size_t i = 0; i < nelems; i++)
  {
    flag &= !array[i];
  }
  return flag;
}

// size is an output
static inline int* boolArray_toIndexList(uint32_t* array, int nItems, int* size){
  int count = 0;
  int nelems = ceil(nItems/32.0);
  for (size_t i = 0; i < nelems; i++)
  {
    uint32_t val = array[i];
    if(!val) continue;
    for (size_t j = 0; j < 32; j++)
    {
      if((val) & (1 << (j))) count++;
    }
  }

  *size = count;

  int* res = NULL;

  //early exit
  if(count == 0){
    return res;
  }

  ALLOC_ARRAY(res, count);
  count = 0;
  for (size_t i = 0; i < nelems; i++)
  {
    uint32_t val = array[i];
    if(!val) continue;
    for (size_t j = 0; j < 32; j++)
    {
      if((val) & (1 << (j))) res[count++] = j + i*32;
    }
  }
  
  return res;
}

// Assume Idx always has a match in mp
// Note for future: binary searchh is viable here
static inline int findPermIdx(int* mp, int mp_size, int Idx ){
  int res = binarySearch(mp, 0, mp_size-1, Idx);
  // for (size_t i = 0; i < mp_size; i++)
  // {
  //   if(mp[i] == Idx)
  //     return i;
  // }
  // ABORT("Idx: %d has no counterpart in mp", Idx) // 414 not found in 0
  // DEBUGLOG("Idx: %d has no counterpart in mp", Idx)
  // TODO: Enable line below
  return res;
}

static inline void fillLocal(CSR* loc, int* locp, const CSR big){
  for (size_t i = 0; i < loc->m; i++){
    int loci = loc->I[i];
    int bigi = big.I[locp[i]];
    int ncols = big.I[locp[i]+1] - bigi; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++)
    {
      int Idx = findPermIdx(locp, loc->m, big.J[bigi+ j]);
      if (Idx == -1)
        ABORT("Idx: %d has no counterpart in mp", Idx)
      loc->J[loci + j] = Idx;
      loc->val[loci + j] = big.val[bigi + j];
    }
  }
}

// new function that takes a completely empty loc (except loc.m)
static inline void fillPart(CSR* part,  int* perm_row, int* perm_col, uint32_t* mask_loc, const CSR big){

  CALLOC_ARRAY(part->I, part->m + 1);
  part->nnz = 0;
  
  // dry run to fill loc.I and count nnz
  for (size_t i = 0; i < part->m; i++){
    int loci = part->I[i];

    int bigi = perm_row[i];
    int bigj = big.I[bigi];
    int ncols = big.I[bigi +1] - bigj; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++)
    {
      if(!boolArray_get(mask_loc, big.J[bigj + j])) continue;

      part->nnz++;
      loci++;

      // int Idx = findPermIdx(perm, part->m, big.J[bigj+ j]);
      // if (Idx == -1) ABORT("Idx: %d has no counterpart in mp", Idx)
      // loc->J[loci + j] = Idx;
      // loc->val[loci + j] = big.val[bigj + j];
    }
    part->I[i+1] = loci;
  }

  if(part->I[part->m] != part->nnz)
    ABORT("Number of non-zeroes (%d) in part doesnt equal last element (%d) in compressed index vector", part->nnz, part->I[part->m])

  ALLOC_ARRAY(part->J, part->nnz);
  ALLOC_ARRAY(part->val, part->nnz);

  int count = 0;

  for (size_t i = 0; i < part->m; i++)
  {
    int loci = part->I[i];

    int bigi = perm_row[i];
    int bigj = big.I[bigi];
    int ncols = big.I[bigi+1] - bigj; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++)
    {
      if(!boolArray_get(mask_loc, big.J[bigj + j])) continue;
      int Idx = findPermIdx(perm_col, part->n, big.J[bigj+ j]);
      if (Idx == -1)
        ABORT("Idx: %d has no counterpart in mp", Idx)
    
      part->J[count] = Idx;
      part->val[count] = big.val[bigj + j];
      count++;
    }
  }

  if(count != part->nnz) ABORT("Count (%d) does not match nnz (%d)", count, part->nnz)
}

//Simplifying assumptions:
// - Matrix is split into 2 pieces. (CPU GPU)
// - shared matrix is empty (cutsize is 0)
// - split is row wise
// - partition is same on rows as columns
SPLIT_CSR* cleanSplit(CSR big, int* partvec){

  SPLIT_CSR* res;
  ALLOC_ARRAY(res, 2);
  res[0] = (SPLIT_CSR){};  // GPU 
  res[1] = (SPLIT_CSR){};  // CPU 

  /*
   * Things to do:
   *  - Count how many rows and non-zeros are owned by each CPU-GPU.
   *  - Calculate the "transpose" of partvec.
   */

  for( size_t i=0; i < big.m; i++){
    int proc = partvec[i];
    res[proc].loc.m++;
    // res[proc].loc.n++;
    res[proc].loc.nnz += (big.I[i+1] - big.I[i]);   //  CSR matrix file  
  }
  ALLOC_ARRAY(res[0].locp, res[0].loc.m);  // loc p permutation index 
  ALLOC_ARRAY(res[1].locp, res[1].loc.m);

  ALLOC_ARRAY(res[0].loc.I, res[0].loc.m + 1);
  ALLOC_ARRAY(res[0].loc.J, res[0].loc.nnz);
  ALLOC_ARRAY(res[0].loc.val, res[0].loc.nnz);
  ALLOC_ARRAY(res[1].loc.I, res[1].loc.m + 1);
  ALLOC_ARRAY(res[1].loc.J, res[1].loc.nnz);
  ALLOC_ARRAY(res[1].loc.val, res[1].loc.nnz);

  {
    res[0].loc.I[0] = 0;   // 
    res[1].loc.I[0] = 0;
    for( size_t i=0; i < big.m; i++){
      int proc = partvec[i];
      int* n = &(res[proc].loc.n);
      int* I = res[proc].loc.I;
      res[proc].locp[(*n)++] = i;
      I[*n] =  (big.I[i+1] - big.I[i]) + I[*n -1];
    }
  }

  if (res[0].loc.m != res[0].loc.n)
    ABORT(
      "split sanity check failed: id:%d, loc.m:%d, loc.n:%d",
      0, res[0].loc.m, res[0].loc.n)

  if (res[1].loc.m != res[1].loc.n)
    ABORT(
      "split sanity check failed: id:%d, loc.m:%d, loc.n:%d",
      1, res[1].loc.m, res[1].loc.n)

  fillLocal( &(res[0].loc), res[0].locp, big);
  fillLocal( &(res[1].loc), res[1].locp, big);

  return res;
}

//Simplifying assumptions:
// - big matrix is square
// - Matrix is split into 2 pieces. (CPU GPU)
// - split is row wise
// - partition is same on rows as columns
SPLIT_CSR* sparseSplit(CSR big, int* partvec){

  SPLIT_CSR* res;
  ALLOC_ARRAY(res, 2);
  res[0] = (SPLIT_CSR){};  // GPU 
  res[1] = (SPLIT_CSR){};  // CPU

  uint32_t* mask_loc[2] = {
    boolArray_new(big.n),
    boolArray_new(big.n)
  };

  for( size_t i=0; i < big.m; i++){
    int proc = partvec[i];
    boolArray_set(mask_loc[proc], i, 1);
    res[proc].loc.m++;
  }

  // Potential sanity check: mmask_loc[0] and mask_loc[1] should be opposites

  res[0].locp = boolArray_toIndexList(mask_loc[0], big.n, &(res[0].loc.n));
  res[1].locp = boolArray_toIndexList(mask_loc[1], big.n, &(res[1].loc.n));

  if(res[0].loc.n != res[0].loc.m)
    ABORT(
    "Sanity check failed on matrix split. Number of set booleans (%d) is different than the number of rows (%d) on 0",
    res[0].loc.n, res[0].loc.m
  );
  if(res[1].loc.n != res[1].loc.m)
    ABORT(
    "Sanity check failed on matrix split. Number of set booleans (%d) is different than the number of rows (%d) on 1",
    res[1].loc.n, res[1].loc.m
  );

  uint32_t* mask_shr[2] = {
    boolArray_new(big.n),
    boolArray_new(big.n)
  };

  for (size_t i = 0; i < big.m; i++)
  {
    int proc = partvec[i];
    for (size_t j = big.I[i]; j < big.I[i+1]; j++)
      boolArray_set(mask_shr[proc], big.J[j], 1);
  }
  boolArray_diff(mask_loc[0], mask_shr[0], big.m);
  boolArray_diff(mask_loc[1], mask_shr[1], big.m);

  res[0].shrp = boolArray_toIndexList(mask_shr[0], big.n, &(res[0].shr.n));
  res[0].shr.m = res[0].loc.m;
  res[1].shrp = boolArray_toIndexList(mask_shr[1], big.n, &(res[1].shr.n));
  res[1].shr.m = res[1].loc.m;

  fillPart(&(res[0].loc), res[0].locp, res[0].locp, mask_loc[0], big);
  if(res[0].shr.n != 0)
    fillPart(&(res[0].shr), res[0].locp, res[0].shrp, mask_shr[0], big);

  fillPart(&(res[1].loc), res[1].locp, res[1].locp, mask_loc[1], big);
  if(res[1].shr.n != 0)
    fillPart(&(res[1].shr), res[1].locp, res[1].shrp, mask_shr[1], big);

  FREE_AND_NULL(mask_loc[0]);
  FREE_AND_NULL(mask_loc[1]);

  FREE_AND_NULL(mask_shr[0]);
  FREE_AND_NULL(mask_shr[1]);

  return res;
  
}

// combines loc and shr of a SPLIT_CSR into loc
SPLIT_CSR combineSplit(const SPLIT_CSR in){
  SPLIT_CSR res = {
    .loc = {
      .I = 0,
      .J = 0,
      .m = in.loc.m,
      .n = in.loc.n + in.shr.n,
      .nnz = in.loc.nnz + in.shr.nnz,
      .val = 0
    },
    .locp = 0,
    .shr = {},
    .shrp = 0
  };
  CALLOC_ARRAY(res.loc.I, res.loc.m + 1);
  ALLOC_ARRAY(res.loc.J, res.loc.nnz);
  ALLOC_ARRAY(res.loc.val, res.loc.nnz);
  ALLOC_ARRAY(res.locp, res.loc.n);

  // fill res.locp
  for(int i=0; i<in.loc.n; i++)
    res.locp[i] = in.locp[i];
  for(int i=0; i<in.shr.n; i++)
    res.locp[i + in.loc.n] = in.shrp[i];

  // fill res.loc.I
  {
    int *loci = in.loc.I;
    int *shri = in.shr.I;
    for (int i=0; i < res.loc.m; i++){
      res.loc.I[i+1] = res.loc.I[i] + (loci[i+1] - loci[i]) + (shri[i+1] - shri[i]);
    }
  }

  // fill res.loc.J and res.loc.val
  {
    int shr_off = in.loc.n;
    for (int i=0; i<res.loc.m; i++){
      int start = res.loc.I[i];
      
      int loc_start = in.loc.I[i];
      int loc_cnt = in.loc.I[i+1] - in.loc.I[i];
      for (int j=0; j < loc_cnt; j++){
        res.loc.J[start + j] = in.loc.J[loc_start + j];
        res.loc.val[start + j] = in.loc.val[loc_start + j];
      }

      int shr_start = in.shr.I[i];
      int shr_cnt = in.shr.I[i+1] - in.shr.I[i];
      for (int j=0; j < shr_cnt; j++){
        res.loc.J[start + loc_cnt + j] = shr_off + in.shr.J[shr_start + j];
        res.loc.val[start + loc_cnt + j] = in.shr.val[shr_start + j];
      }

      if ( shr_cnt + loc_cnt != res.loc.I[i+1] - start)
        ABORT(
          "Split merge, loc_cnt (%d) + shr_cnt (%d) is not equal to res_cnt (%d)",
          loc_cnt, shr_cnt, res.loc.I[i+1] - start
        )
    }
  }

  return res;
}