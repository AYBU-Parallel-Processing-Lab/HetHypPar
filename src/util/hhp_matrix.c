#include "bora_spmxv.h"
#include "hhp_cpu.h"
#include "math.h"
#include "mmio.h"
#include "patoh.h"
#include "stdint.h"
#include "stdio.h"
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include "hhp_common.h"
#include "hhp_matrix.h"
#include "hhp_util.h"
#include "hhp_wrap_bora.h"

// improved via claude.ai
static inline CSR CSC_to_CSR(const CSC *in) {
  CSR out = {.I = (int *)malloc(sizeof(int) * (in->m + 1)),
             .J = (int *)malloc(sizeof(int) * in->nnz),
             .val = (double *)malloc(sizeof(double) * in->nnz),
             .nnz = in->nnz,
             .m = in->m,
             .n = in->n};
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

COO CSR_to_COO(const CSR *in) {
  COO out = {.nnz = in->nnz, .m = in->m, .n = in->n};
  ALLOC_ARRAY(out.I, in->nnz);
  ALLOC_ARRAY(out.J, in->nnz);
  ALLOC_ARRAY(out.val, in->nnz);

  for (int i = 0; i < out.m; i++) {
    for (int j = in->I[i]; j < in->I[i + 1]; j++) {
      out.I[j] = i;
      out.J[j] = in->J[j];

      if ((out.J[j] >= out.m) || (out.J[j] > out.n)) {
        DEBUGLOG("out.J[j (%d)] (%d) is bigger than out.m or out.n", j,
                 out.J[j]);
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

CSR buReadSparseMatrix(char *fname) {
  buMatrix bigA;

  readMatrixMarketFromFile(&bigA, fname);

  CSR out = {bigA.ia, bigA.ja, bigA.val, bigA.nnz, bigA.gm, bigA.gn};

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

static void nnzDistribution(int *cweights, int numpart, int nrow, int *partvec,
                            char *resultFName) {

  int nnzCount[numpart];
  for (int i = 0; i < numpart; i++)
    nnzCount[i] = 0;

  for (int i = 0; i < nrow; i++) {

    nnzCount[partvec[i]] += cweights[i];
  }

  //-----------------------Result file ----------------------------------
  FILE *fptr = fopen(resultFName, "a");

  if (fptr == NULL) {
    printf("Result file cannot opened!!");
    exit(-1);
  } else {

    time_t mytime;
    mytime = time(NULL);

    fprintf(fptr, "%s\n", ctime(&mytime));
    fprintf(fptr, "Number of Non-zero Count : \n");
    for (int i = 0; i < numpart; i++) {
      printf("[%d] %d ", i, nnzCount[i]);
      fprintf(fptr, " [%d] : %d ", i, nnzCount[i]);
    }
  }
  fclose(fptr);
  //--------------------- Ending Result file -----------------------------
  printf("\n");
}

int *CalcPartVec(int nparts, const CSC *cscmatrix, const CSR *csrmatrix,
                 char *fName, double final_imbal, int seed, char *resultFName) {

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
  args.final_imbal = final_imbal;
  args.seed = seed;
  // =========================================================================================================

  int *partvec =
      malloc(sizeof(int) * csrmatrix->n); // contains the resulting partition
  int partweights[args._k]; // contains the weights of the partition
  int cut;

  float targetweigths[nparts];

  // Open a file in read mode
  FILE *fptr;
  if ((fptr = fopen(fName, "r")) == NULL) {
    fprintf(stderr, "file can't be opened %s file \n", fName);
    exit(-1);
  }

  int i = 0;
  while (fscanf(fptr, "%f ", targetweigths + i++) == 1)
    ;

  fclose(fptr);

  PaToH_Alloc(&args, csrmatrix->n, csrmatrix->m, 1, cweights, nweights,
              csrmatrix->I, csrmatrix->J);

  PaToH_Part(&args, csrmatrix->n, csrmatrix->m, 1, 0, cweights, nweights,
             csrmatrix->I, csrmatrix->J, targetweigths, partvec, partweights,
             &cut);

  nnzDistribution(cweights, nparts, cscmatrix->m, partvec,
                  resultFName); // print nnz per part

  printf("cut : %d", cut);

  // --------------------------- open result file for write cut size
  // -----------------------
  FILE *file = fopen(resultFName, "a");

  if (file == NULL) {
    printf("Result file cannot opened!!");
    exit(-1);
  } else {
    fprintf(file, "\n\ncut  : %d \n", cut);
  }
  fclose(file);

  // --------------------------- closing file
  // ----------------------------------------------

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

// Taken From G2G
//  A iterative binary search function. It returns location
//  of x in given array arr[l..r] if present, otherwise -1
static int binarySearch(int arr[], int l, int r, int x) {
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

static inline uint32_t *boolArray_new(int nItems) {
  uint32_t *res;
  int nelems = ceil(nItems / 32.0);
  CALLOC_ARRAY(res, nelems);
  return res;
}

static inline int boolArray_get(uint32_t *array, int index) {
  return (array[index / 32]) & (1 << (index % 32));
}

static inline void boolArray_set(uint32_t *array, int index, int val) {
  uint32_t mask = -1;
  mask ^= 1 << (index % 32);
  array[index / 32] &= mask;
  array[index / 32] |= (val) << (index % 32);
}

// find ~a1 & a2
// store result into a2
static inline void boolArray_diff(uint32_t *a1, uint32_t *a2, int nItems) {
  int nelems = ceil(nItems / 32.0);
  for (size_t i = 0; i < nelems; i++) {
    a2[i] &= ~(a1[i]);
  }
}

static inline int boolArray_isEmpty(uint32_t *array, int nItems) {
  int flag = 1;
  int nelems = ceil(nItems / 32.0);
  for (size_t i = 0; i < nelems; i++) {
    flag &= !array[i];
  }
  return flag;
}

// size is an output
static inline int *boolArray_toIndexList(uint32_t *array, int nItems,
                                         int *size) {
  int count = 0;
  int nelems = ceil(nItems / 32.0);
  for (size_t i = 0; i < nelems; i++) {
    uint32_t val = array[i];
    if (!val)
      continue;
    for (size_t j = 0; j < 32; j++) {
      if ((val) & (1 << (j)))
        count++;
    }
  }

  *size = count;

  int *res = NULL;

  // early exit
  if (count == 0) {
    return res;
  }

  ALLOC_ARRAY(res, count);
  count = 0;
  for (size_t i = 0; i < nelems; i++) {
    uint32_t val = array[i];
    if (!val)
      continue;
    for (size_t j = 0; j < 32; j++) {
      if ((val) & (1 << (j)))
        res[count++] = j + i * 32;
    }
  }

  return res;
}

// Assume Idx always has a match in mp
// Note for future: binary searchh is viable here
static inline int findPermIdx(int *mp, int mp_size, int Idx) {
  int res = binarySearch(mp, 0, mp_size - 1, Idx);
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

static inline void fillLocal(CSR *loc, int *locp, const CSR big) {
  for (size_t i = 0; i < loc->m; i++) {
    int loci = loc->I[i];
    int bigi = big.I[locp[i]];
    int ncols = big.I[locp[i] + 1] - bigi; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++) {
      int Idx = findPermIdx(locp, loc->m, big.J[bigi + j]);
      if (Idx == -1)
        ABORT("Idx: %d has no counterpart in mp", Idx)
      loc->J[loci + j] = Idx;
      loc->val[loci + j] = big.val[bigi + j];
    }
  }
}

// new function that takes a completely empty loc (except loc.m)
static inline void fillPart(CSR *part, int *perm_row, int *perm_col,
                            uint32_t *mask_loc, const CSR big) {

  CALLOC_ARRAY(part->I, part->m + 1);
  part->nnz = 0;

  // dry run to fill loc.I and count nnz
  for (size_t i = 0; i < part->m; i++) {
    int loci = part->I[i];

    int bigi = perm_row[i];
    int bigj = big.I[bigi];
    int ncols = big.I[bigi + 1] - bigj; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++) {
      if (!boolArray_get(mask_loc, big.J[bigj + j]))
        continue;

      part->nnz++;
      loci++;

      // int Idx = findPermIdx(perm, part->m, big.J[bigj+ j]);
      // if (Idx == -1) ABORT("Idx: %d has no counterpart in mp", Idx)
      // loc->J[loci + j] = Idx;
      // loc->val[loci + j] = big.val[bigj + j];
    }
    part->I[i + 1] = loci;
  }

  if (part->I[part->m] != part->nnz)
    ABORT("Number of non-zeroes (%d) in part doesnt equal last element (%d) in "
          "compressed index vector",
          part->nnz, part->I[part->m])

  ALLOC_ARRAY(part->J, part->nnz);
  ALLOC_ARRAY(part->val, part->nnz);

  int count = 0;

  for (size_t i = 0; i < part->m; i++) {
    int loci = part->I[i];

    int bigi = perm_row[i];
    int bigj = big.I[bigi];
    int ncols = big.I[bigi + 1] - bigj; // number of non-zeroes in row
    for (size_t j = 0; j < ncols; j++) {
      if (!boolArray_get(mask_loc, big.J[bigj + j]))
        continue;
      int Idx = findPermIdx(perm_col, part->n, big.J[bigj + j]);
      if (Idx == -1)
        ABORT("Idx: %d has no counterpart in mp", Idx)

      part->J[count] = Idx;
      part->val[count] = big.val[bigj + j];
      count++;
    }
  }

  if (count != part->nnz)
    ABORT("Count (%d) does not match nnz (%d)", count, part->nnz)
}

// //Simplifying assumptions:
// // - Matrix is split into 2 pieces. (CPU GPU)
// // - shared matrix is empty (cutsize is 0)
// // - split is row wise
// // - partition is same on rows as columns
// SHARD_CSR* cleanSplit(CSR big, int* partvec){

//   SHARD_CSR* res;
//   ALLOC_ARRAY(res, 2);
//   res[0] = (SHARD_CSR){};  // GPU
//   res[1] = (SHARD_CSR){};  // CPU

//   /*
//    * Things to do:
//    *  - Count how many rows and non-zeros are owned by each CPU-GPU.
//    *  - Calculate the "transpose" of partvec.
//    */

//   for( size_t i=0; i < big.m; i++){
//     int proc = partvec[i];
//     res[proc].loc.m++;
//     // res[proc].loc.n++;
//     res[proc].loc.nnz += (big.I[i+1] - big.I[i]);   //  CSR matrix file
//   }
//   ALLOC_ARRAY(res[0].locp, res[0].loc.m);  // loc p permutation index
//   ALLOC_ARRAY(res[1].locp, res[1].loc.m);

//   ALLOC_ARRAY(res[0].loc.I, res[0].loc.m + 1);
//   ALLOC_ARRAY(res[0].loc.J, res[0].loc.nnz);
//   ALLOC_ARRAY(res[0].loc.val, res[0].loc.nnz);
//   ALLOC_ARRAY(res[1].loc.I, res[1].loc.m + 1);
//   ALLOC_ARRAY(res[1].loc.J, res[1].loc.nnz);
//   ALLOC_ARRAY(res[1].loc.val, res[1].loc.nnz);

//   {
//     res[0].loc.I[0] = 0;   //
//     res[1].loc.I[0] = 0;
//     for( size_t i=0; i < big.m; i++){
//       int proc = partvec[i];
//       int* n = &(res[proc].loc.n);
//       int* I = res[proc].loc.I;
//       res[proc].locp[(*n)++] = i;
//       I[*n] =  (big.I[i+1] - big.I[i]) + I[*n -1];
//     }
//   }

//   if (res[0].loc.m != res[0].loc.n)
//     ABORT(
//       "split sanity check failed: id:%d, loc.m:%d, loc.n:%d",
//       0, res[0].loc.m, res[0].loc.n)

//   if (res[1].loc.m != res[1].loc.n)
//     ABORT(
//       "split sanity check failed: id:%d, loc.m:%d, loc.n:%d",
//       1, res[1].loc.m, res[1].loc.n)

//   fillLocal( &(res[0].loc), res[0].locp, big);
//   fillLocal( &(res[1].loc), res[1].locp, big);

//   return res;
// }

// //Simplifying assumptions:
// // - big matrix is square
// // - Matrix is split into 2 pieces. (CPU GPU)
// // - split is row wise
// // - partition is same on rows as columns
// SHARD_CSR* sparseSplit(CSR big, int* partvec){

//   SHARD_CSR* res;
//   ALLOC_ARRAY(res, 2);
//   res[0] = (SHARD_CSR){};  // GPU
//   res[1] = (SHARD_CSR){};  // CPU

//   uint32_t* mask_loc[2] = {
//     boolArray_new(big.n),
//     boolArray_new(big.n)
//   };

//   for( size_t i=0; i < big.m; i++){
//     int proc = partvec[i];
//     boolArray_set(mask_loc[proc], i, 1);
//     res[proc].loc.m++;
//   }

//   // Potential sanity check: mmask_loc[0] and mask_loc[1] should be opposites

//   res[0].locp = boolArray_toIndexList(mask_loc[0], big.n, &(res[0].loc.n));
//   res[1].locp = boolArray_toIndexList(mask_loc[1], big.n, &(res[1].loc.n));

//   if(res[0].loc.n != res[0].loc.m)
//     ABORT(
//     "Sanity check failed on matrix split. Number of set booleans (%d) is
//     different than the number of rows (%d) on 0", res[0].loc.n, res[0].loc.m
//   );
//   if(res[1].loc.n != res[1].loc.m)
//     ABORT(
//     "Sanity check failed on matrix split. Number of set booleans (%d) is
//     different than the number of rows (%d) on 1", res[1].loc.n, res[1].loc.m
//   );

//   uint32_t* mask_shr[2] = {
//     boolArray_new(big.n),
//     boolArray_new(big.n)
//   };

//   for (size_t i = 0; i < big.m; i++)
//   {
//     int proc = partvec[i];
//     for (size_t j = big.I[i]; j < big.I[i+1]; j++)
//       boolArray_set(mask_shr[proc], big.J[j], 1);
//   }
//   boolArray_diff(mask_loc[0], mask_shr[0], big.m);
//   boolArray_diff(mask_loc[1], mask_shr[1], big.m);

//   res[0].shrp = boolArray_toIndexList(mask_shr[0], big.n, &(res[0].shr.n));
//   res[0].shr.m = res[0].loc.m;
//   res[1].shrp = boolArray_toIndexList(mask_shr[1], big.n, &(res[1].shr.n));
//   res[1].shr.m = res[1].loc.m;

//   fillPart(&(res[0].loc), res[0].locp, res[0].locp, mask_loc[0], big);
//   if(res[0].shr.n != 0)
//     fillPart(&(res[0].shr), res[0].locp, res[0].shrp, mask_shr[0], big);

//   fillPart(&(res[1].loc), res[1].locp, res[1].locp, mask_loc[1], big);
//   if(res[1].shr.n != 0)
//     fillPart(&(res[1].shr), res[1].locp, res[1].shrp, mask_shr[1], big);

//   FREE_AND_NULL(mask_loc[0]);
//   FREE_AND_NULL(mask_loc[1]);

//   FREE_AND_NULL(mask_shr[0]);
//   FREE_AND_NULL(mask_shr[1]);

//   return res;

// }

// // combines loc and shr of a SPLIT_CSR into loc
// SHARD_CSR combineSplit(const SHARD_CSR in){
//   SHARD_CSR res = {
//     .loc = {
//       .I = 0,
//       .J = 0,
//       .m = in.loc.m,
//       .n = in.loc.n + in.shr.n,
//       .nnz = in.loc.nnz + in.shr.nnz,
//       .val = 0
//     },
//     .locp = 0,
//     .shr = {},
//     .shrp = 0
//   };
//   CALLOC_ARRAY(res.loc.I, res.loc.m + 1);
//   ALLOC_ARRAY(res.loc.J, res.loc.nnz);
//   ALLOC_ARRAY(res.loc.val, res.loc.nnz);
//   ALLOC_ARRAY(res.locp, res.loc.n);

//   // fill res.locp
//   for(int i=0; i<in.loc.n; i++)
//     res.locp[i] = in.locp[i];
//   for(int i=0; i<in.shr.n; i++)
//     res.locp[i + in.loc.n] = in.shrp[i];

//   // fill res.loc.I
//   {
//     int *loci = in.loc.I;
//     int *shri = in.shr.I;
//     for (int i=0; i < res.loc.m; i++){
//       res.loc.I[i+1] = res.loc.I[i] + (loci[i+1] - loci[i]) + (shri[i+1] -
//       shri[i]);
//     }
//   }

//   // fill res.loc.J and res.loc.val
//   {
//     int shr_off = in.loc.n;
//     for (int i=0; i<res.loc.m; i++){
//       int start = res.loc.I[i];

//       int loc_start = in.loc.I[i];
//       int loc_cnt = in.loc.I[i+1] - in.loc.I[i];
//       for (int j=0; j < loc_cnt; j++){
//         res.loc.J[start + j] = in.loc.J[loc_start + j];
//         res.loc.val[start + j] = in.loc.val[loc_start + j];
//       }

//       int shr_start = in.shr.I[i];
//       int shr_cnt = in.shr.I[i+1] - in.shr.I[i];
//       for (int j=0; j < shr_cnt; j++){
//         res.loc.J[start + loc_cnt + j] = shr_off + in.shr.J[shr_start + j];
//         res.loc.val[start + loc_cnt + j] = in.shr.val[shr_start + j];
//       }

//       if ( shr_cnt + loc_cnt != res.loc.I[i+1] - start)
//         ABORT(
//           "Split merge, loc_cnt (%d) + shr_cnt (%d) is not equal to res_cnt
//           (%d)", loc_cnt, shr_cnt, res.loc.I[i+1] - start
//         )
//     }
//   }

//   return res;
// }

void print_CSC(FILE* f, CSC m){
  fprintf(f,"==========\nMATRIX PRINT\n========\n");
  
  fprintf(f,"NNZ: %d\n", m.nnz);

  fprintf(f,"I = ");
  for (size_t i = 0; i < m.n+1; i++)
  {
    fprintf(f," %d",m.I[i]);
  }
  fprintf(f,"\n");

  fprintf(f,"J = ");
  for (size_t i = 0; i < m.nnz; i++)
  {
    fprintf(f," %d",m.J[i]);
  }
  fprintf(f,"\n");

  fprintf(f,"val = ");
  for (size_t i = 0; i < m.nnz; i++)
  {
    fprintf(f," %lf",m.val[i]);
  }
  fprintf(f,"\n=========================\n");
}

void print_COMM(FILE* f, COMM comm){
  fprintf(f,"==========\nCOMM PRINT\n========\n");
  
  fprintf(f,"num: %d\n", comm.num);

  fprintf(f,"I = ");
  for (size_t i = 0; i < comm.num+1; i++)
  {
    fprintf(f," %d",comm.I[i]);
  }
  fprintf(f,"\n");

  fprintf(f,"J = ");
  for (size_t i = 0; i < comm.I[comm.num]; i++)
  {
    fprintf(f," %d",comm.J[i]);
  }
  fprintf(f,"\n");

  fprintf(f,"rank = ");
  for (size_t i = 0; i < comm.num; i++)
  {
    fprintf(f," %d",comm.ranks[i]);
  }
  fprintf(f,"\n=========================\n");
}

void print_CSR(FILE* f, CSR m){
  CSC tmat = m;
  tmat.m = m.n;
  tmat.n = m.m;
  print_CSC(f, tmat);
}

void print_ivector(FILE* f, iVector v){
  fprintf(f,"==========\niVECTOR PRINT\n========\n");
  
  fprintf(f,"nval = %d",v.nvals);
  fprintf(f,"val = ");
  for (size_t i = 0; i < v.nvals; i++)
  {
    fprintf(f," %d",v.vals[i]);
  }
  fprintf(f,"\n");

  fprintf(f,"\n=========================\n");
}

void print_CSC_SHARD(FILE* f,SHARD_CSC m){
  
  fprintf(f, "==========\nSHARD PRINT\n========\n");
  
  fprintf(f, "LOC\n");
  
  print_CSC(f, m.loc);
  
  fprintf(f, "\nSHR\n");
  print_CSC(f, m.shr);
  
  fprintf(f, "\nGIND\n");
  print_ivector(f, m.gind);

  fprintf(f, "\nRECV\n");
  print_COMM(f, m.recv);

  fprintf(f, "\nSEND\n");
  print_COMM(f, m.send);

}

// Function to free a CSC matrix
void free_csc(CSC *csc) {
  if (!csc)
    return;

  FREE_AND_NULL(csc->I);
  FREE_AND_NULL(csc->J);
  FREE_AND_NULL(csc->val);
  FREE_AND_NULL(csc);
}

static CSR *internal_CSR_split_row(CSR in, int nshards, iVector partvec) {
  // Note that partvec must be of length in.n
  if (partvec.nvals != in.n) {
    // Error handling
    return NULL;
  }

  // Create and allocate memory for output
  CSR *res;
  ALLOC_ARRAY(res, nshards);

  // Initialize shards (without allocating I yet)
  for (size_t i = 0; i < nshards; i++) {
    res[i] =
        (CSR){.nnz = 0, .m = 0, .n = in.n, .I = NULL, .J = NULL, .val = NULL};
  }

  // First pass: count number of rows and nonzeros in each shard
  for (size_t i = 0; i < in.n; i++) {
    int j = partvec.vals[i];
    if (j < 0 || j >= nshards) {
      ABORT("partvec has an element out of range, line %d -> %d", (int)i, j);
    }
    res[j].m++;
    res[j].nnz += (in.I[i + 1] - in.I[i]);
  }

  // Allocate I, J, val arrays for each shard
  for (size_t i = 0; i < nshards; i++) {
    ALLOC_ARRAY(res[i].I, res[i].m + 1);
    ALLOC_ARRAY(res[i].J, res[i].nnz);
    ALLOC_ARRAY(res[i].val, res[i].nnz);

    // Initialize row pointer array
    res[i].I[0] = 0;
  }

  // Reset counters for second pass
  int *row_counts;
  ALLOC_ARRAY(row_counts, nshards);
  int *nnz_counts;
  ALLOC_ARRAY(nnz_counts, nshards);
  for (size_t i = 0; i < nshards; i++) {
    row_counts[i] = 0;
    nnz_counts[i] = 0;
  }

  // Second pass: build CSR structures
  for (size_t i = 0; i < in.n; i++) {
    int j = partvec.vals[i];
    int size = in.I[i + 1] - in.I[i];

    int current_row = row_counts[j];
    int current_pos = nnz_counts[j];

    // Update row pointer for next row
    res[j].I[current_row + 1] = current_pos + size;

    // Copy data
    int g_start = in.I[i];
    memcpy(res[j].J + current_pos, in.J + g_start, sizeof(int) * size);
    memcpy(res[j].val + current_pos, in.val + g_start,
           sizeof(res[j].val[0]) * size);

    // Update counters
    row_counts[j]++;
    nnz_counts[j] += size;
  }

  // Clean up temporary arrays
  FREE_AND_NULL(row_counts);
  FREE_AND_NULL(nnz_counts);

  return res;
}

static SHARD_CSC internal_CSR_loc_split(CSR in, iVector partvec, int myrank) {
  SHARD_CSC res = {
      .loc = (CSC){.m = in.m, .n = 0, .nnz = 0},
      .shr = (CSC){.m = in.m, .n = 0, .nnz = 0},
  };

  CSC temp_mat = CSR_to_CSC(in);

  // First pass: count nnz and n
  for (size_t i = 0; i < partvec.nvals; i++)
  {
    int owner_id = partvec.vals[i];
    int cnt = temp_mat.I[i+1] - temp_mat.I[i];

    if (owner_id == myrank){
      res.loc.nnz += cnt;
      res.loc.n++;
      continue;
    }

    if (cnt == 0)
      continue;

    res.shr.nnz += cnt;
    res.shr.n++;
  }

  if (res.loc.n != res.loc.m)
    ABORT("loc matrix must be square! (loc.n = %d , loc.m = %d)", res.loc.n, res.loc.m);

  CALLOC_ARRAY(res.loc.I, res.loc.n+1);
  ALLOC_ARRAY(res.loc.J, res.loc.nnz);
  ALLOC_ARRAY(res.loc.val, res.loc.nnz);
  CALLOC_ARRAY(res.shr.I, res.shr.n+1);
  ALLOC_ARRAY(res.shr.J, res.shr.nnz);
  ALLOC_ARRAY(res.shr.val, res.shr.nnz);
  res.gind = ivector_init(res.loc.n + res.shr.n);

  int ncols_loc = 0;
  int ncols_shr = 0;

  // Second pass: Fill loc and shr I J val vectors.
  for (size_t i = 0; i < partvec.nvals; i++)
  {
    int owner_id = partvec.vals[i];
    int cnt = temp_mat.I[i+1] - temp_mat.I[i];

    int isloc = owner_id == myrank;

    
    if (isloc){
      res.gind.vals[ncols_loc] = i;
      ncols_loc++;
    }
    else if (cnt != 0){
      res.gind.vals[res.loc.n + ncols_shr] = i;
      ncols_shr++;
    }else{
      continue;
    }

    CSC mymat = isloc ? res.loc : res.shr;
    int mycol = isloc ? ncols_loc : ncols_shr;

    int myoff = mymat.I[mycol -1];
    mymat.I[mycol] = myoff + cnt;
    
    int hisoff = temp_mat.I[i];

    for (size_t j = 0; j < cnt; j++)
    {
      mymat.J[myoff + j] = temp_mat.J[hisoff + j];
      mymat.val[myoff + j] = temp_mat.val[hisoff + j];
    }
  }

  if (ncols_loc != res.loc.n)
    ABORT("gind for loc was not filled properly, entered %d values when %d was expected", ncols_loc, res.loc.n);
  if (ncols_shr != res.shr.n)
    ABORT("gind for shr was not filled properly, entered %d values when %d was expected", ncols_shr, res.shr.n);


  freeSparseMatrix(&temp_mat);
  
  return res;
}

/**
 * @brief Setup communication patterns for a sharded CSR matrix
 * 
 * This function determines which elements need to be communicated between MPI processes
 * for shared elements and sets up the COMM structures in the SHARD_CSR result.
 * 
 * @param result The SHARD_CSC structure with local and shared matrices already defined
 * @param partvec The partition vector mapping columns to MPI ranks
 * @param mpi_rank The current process rank
 * @param mpi_size The total number of processes
 */
static void internal_setup_communication(SHARD_CSC *result, iVector partvec, int mpi_rank, int mpi_size)
{

  // [x] IS FREED?
  iVector recv_count = ivector_init(mpi_size + 1);
  // Initialize the arrays
  for (int r = 0; r < recv_count.nvals; r++)
    recv_count.vals[r] = 0;

  { // fill out recv
    COMM recv = {};
    
    // First pass: count how many columns each rank owns
    for (int i = 0; i < result->shr.n; i++) {
      int global_col = result->gind.vals[result->loc.n + i];
      int owner_rank = partvec.vals[global_col];
        
      // Skip if it's our own rank (shouldn't happen for shared columns)
      if (owner_rank == mpi_rank)
        MPI_ABORT("Shared matrix has a local index (%d at rank %d)", owner_rank, mpi_rank);
        
      // Just count for now
      recv_count.vals[owner_rank+1]++;
    }
    
    for (size_t i = 1; i < mpi_size+1; i++){
      if (recv_count.vals[i] != 0)
        recv.num++;
    
      // cummulate recv count
      recv_count.vals[i] += recv_count.vals[i-1];
    }

    if (recv_count.vals[mpi_size] != result->shr.n)
      MPI_ABORT("recv_count.vals[mpi_size] (%d) != result->shr.n (%d)", recv_count.vals[mpi_size], result->shr.n);
    
  
    CALLOC_ARRAY(recv.I, recv.num+1);
    CALLOC_ARRAY(recv.ranks, recv.num);
    CALLOC_ARRAY(recv.J, recv_count.vals[mpi_size]);
    CALLOC_ARRAY(recv.val, recv_count.vals[mpi_size]);
    
    int num = 0;
    for (size_t i = 0; i < mpi_size; i++)
    {
      int cnt = recv_count.vals[i+1] - recv_count.vals[i];
      if (cnt == 0)
        continue;
      if (i == mpi_rank)
        MPI_ABORT("This is impossible");

      recv.ranks[num] = i;
      recv.I[num+1] = recv_count.vals[i+1];
      num++;
    }

    if(num != recv.num)
      ABORT("recv.num (%d) doesn't match num (%d)", recv.num, num);

    // [x] IS FREED?
    iVector recv_count2 = ivector_init(mpi_size);
    for (size_t i = 0; i < recv_count2.nvals; i++)
      recv_count2.vals[i] = 0;

    for (size_t i = 0; i < result->shr.n; i++)
    {
      int global_col = result->gind.vals[result->loc.n + i];
      int owner_rank = partvec.vals[global_col];

      recv.J[recv_count.vals[owner_rank] + recv_count2.vals[owner_rank]] = i;

      recv_count2.vals[owner_rank]++;
    }
    
    ivector_destroy(&recv_count2);
    
    result->recv = recv;
  }
  
  if (mpi_rank == 0)
    DEBUGLOG("Recv filled out");


  { // Fill out sends
    // [x] PLACED INTO RESULT?
    COMM send = {};

    MPI_Request reqs[mpi_size - 1];
    MPI_Status stats[mpi_size - 1];

    // [x] IS FREED?
    iVector send_count = ivector_init(mpi_size);
    
    for (size_t i = 0; i < send_count.nvals; i++)
    {
      send_count.vals[i] = 0;
    }
    
    // convert from cummulative to normal
    for (size_t i = 0; i < mpi_size; i++)
      recv_count.vals[i] = recv_count.vals[i+1] - recv_count.vals[i];
    
    
    for (int r = 0; r < mpi_size; r++)
    {
      if (r == mpi_rank){
        continue;
      }

      int off = (r > mpi_rank) ? -1 : 0;

      // Communicate with every other rank to figure out how much you send to each
      // We are not interested in send state, only recv
      MPI_Request temp_req;
      MPI_CHECK( MPI_Isend(recv_count.vals + r, 1, MPI_INT, r, 0, MPI_COMM_WORLD, &temp_req) );
      MPI_CHECK(MPI_Request_free(&temp_req)); // We are not interested in this result

      MPI_CHECK( MPI_Irecv(send_count.vals + r, 1, MPI_INT, r, 0, MPI_COMM_WORLD, reqs + r + off));
    }
  
    MPI_CHECK(MPI_Waitall(mpi_size - 1, reqs, stats));
  
    int stotal = 0;
    for (size_t i = 0; i < mpi_size; i++)
    {
      stotal += send_count.vals[i];
      if (send_count.vals[i] != 0)
        send.num++;

      if ((send_count.vals[i] < 0) || (send_count.vals[i] > partvec.nvals))
        MPI_ABORT("Got invalid value at send_count.vals[%d] (%d)", i, send_count.vals[i]);


      // send_count.vals[i] += send_count.vals[i-1];
    }

    CALLOC_ARRAY(send.I, send.num+1);
    CALLOC_ARRAY(send.ranks, send.num);
    CALLOC_ARRAY(send.J, stotal);
    CALLOC_ARRAY(send.val, stotal);

    for (size_t i = 0; i < send.num+1; i++)
    {
      send.I[i] = 0;
    }

    
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    ivector_destroy(&recv_count);
    
    int num = 0;
    for (size_t r = 0; r < mpi_size; r++)
    {
      int mysize = send_count.vals[r];
      if(mysize == 0)
        continue;
      if(r == mpi_rank)
        MPI_ABORT("send count is non-zero (%d) for self, should be impossible", mysize);

      send.ranks[num] = r;
      send.I[num+1] = mysize + send.I[num];

      num++;
    }



    ivector_destroy(&send_count);

    // [x] IS FREED?
    // recv_gJ maps each shared column to its global index for the
    // Isend/Irecv exchange below. When shr.n == 0 (cutsize 0 on this
    // rank), there are no shared columns so we use an empty sentinel.
    iVector recv_gJ = {.nvals = 0, .vals = NULL};
    if (result->shr.n > 0) {
      recv_gJ = ivector_init(result->shr.n);
      for (size_t i = 0 ; i < result->shr.n; i++)
      {
        int lind = result->recv.J[i];
        int gind = result->gind.vals[result->loc.n + lind];

        recv_gJ.vals[i] = gind;
      }
    }

    // [x] IS FREED?
    // maps every column to their corresponding entry in local x vector, -1 if not owned
    iVector locmap = ivector_init(partvec.nvals);
    for (size_t i = 0; i < locmap.nvals; i++)
      locmap.vals[i] = -1; // initialize
    for (size_t i = 0; i < result->loc.n; i++)
      locmap.vals[result->gind.vals[i]] = i; // map loc entries

    // FIX: Use-after-free race condition in MPI communication.
    //
    // Previously this used MPI_Isend + MPI_Request_free (fire-and-forget),
    // then freed recv_gJ after only waiting on receives. MPI_Request_free
    // detaches the handle but does NOT guarantee the send has completed —
    // if the receiver (slower rank) hasn't consumed the data yet, MPI may
    // still be reading from recv_gJ when it gets freed, causing the receiver
    // to get garbage in send.J and segfault at the locmap lookup below.
    //
    // This was non-deterministic and correlated with structurally asymmetric
    // matrices (lhr14, bayer02, g7jac*, etc.) which create lopsided
    // communication patterns — one rank finishes much faster and frees the
    // buffer before the other rank's receive completes.
    //
    // Fix: track send requests and MPI_Waitall on them before freeing recv_gJ.
    MPI_Request *send_reqs2 = NULL;
    if (result->recv.num > 0)
      ALLOC_ARRAY(send_reqs2, result->recv.num);
    for (size_t i = 0; i < result->recv.num; i++)
    {
      int target = result->recv.ranks[i];
      int size = result->recv.I[i+1] - result->recv.I[i];
      MPI_CHECK( MPI_Isend(recv_gJ.vals + result->recv.I[i], size, MPI_INT, target, 0, MPI_COMM_WORLD, send_reqs2 + i) );
    }

    // receive for send targets
    for (size_t i = 0; i < send.num; i++)
    {
      int target = send.ranks[i];
      int size = send.I[i+1] - send.I[i];
      if (size <= 0)
        MPI_ABORT("Rank %d got invalid size (%d) at i=%d", mpi_rank, size,(int)i);
      // reuse reqs
      MPI_CHECK( MPI_Irecv( send.J + send.I[i], size, MPI_INT, target, 0, MPI_COMM_WORLD, reqs + i));
    }

    MPI_CHECK(MPI_Waitall(send.num, reqs, stats));
    // Wait for sends to complete before freeing the send buffer (recv_gJ)
    if (result->recv.num > 0) {
      MPI_CHECK(MPI_Waitall(result->recv.num, send_reqs2, MPI_STATUSES_IGNORE));
      FREE_AND_NULL(send_reqs2);
    }

    // convert global index in send.J into local index
    for (size_t i = 0; i < send.I[send.num]; i++)
      send.J[i] = locmap.vals[send.J[i]];

    ivector_destroy(&locmap);
    if (recv_gJ.vals)
      ivector_destroy(&recv_gJ);

    result->send = send;
  }
  if (mpi_rank == 0)
    DEBUGLOG("Send filled out");
  
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // WE ARE DONE HERE
  return;
}

SHARD_CSC MPI_CSR_split_row(CSR big, iVector partvec) {
  // CSR big is only defined in rank == 0

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Create result structure
  SHARD_CSC result = {0};

  CSR temp = {0};

  if (mpi_rank == 0) {

    DEBUGLOG("Beginning matrix partitioning");
    
    // Split the matrix into shards
    CSR *Aall = internal_CSR_split_row(big, mpi_size, partvec);

    DEBUGLOG("Row part done");

    // Keep the local shard for rank 0
    temp = Aall[0];

    // Send each shard to its corresponding rank (1 to mpi_size-1)
    for (int dest_rank = 1; dest_rank < mpi_size; dest_rank++) {
      CSR shard = Aall[dest_rank];

      // Send dimensions first - use VLA for small array
      int dims[3] = {shard.m, shard.n, shard.nnz};
      MPI_Send(dims, 3, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);

      // Send the arrays (I, J, val)
      MPI_Send(shard.I, shard.m + 1, MPI_INT, dest_rank, 1, MPI_COMM_WORLD);
      MPI_Send(shard.J, shard.nnz, MPI_INT, dest_rank, 2, MPI_COMM_WORLD);
      MPI_Send(shard.val, shard.nnz, MPI_DOUBLE, dest_rank, 3, MPI_COMM_WORLD);
    }

    // Free the array of shards (but not the shard data, which is now owned by
    // recipients)
    for (size_t i = 1; i < mpi_size; i++)
      freeSparseMatrix(&Aall[i]);
    FREE_AND_NULL(Aall);
  } else {
    // Other ranks receive their shards
    int dims[3]; // Use VLA for small array
    MPI_Recv(dims, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Unpack dimensions
    temp = (CSR){.m = dims[0], .n = dims[1], .nnz = dims[2]};

    ALLOC_ARRAY(temp.I, temp.m + 1);
    ALLOC_ARRAY(temp.J, temp.nnz);
    ALLOC_ARRAY(temp.val, temp.nnz);

    // Receive the arrays
    MPI_Recv(temp.I, temp.m + 1, MPI_INT, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(temp.J, temp.nnz, MPI_INT, 0, 2, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(temp.val, temp.nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  // Synchronize all processes
  MPI_Barrier(MPI_COMM_WORLD);

  if (mpi_rank == 0)
      DEBUGLOG("Distributed the row parted matrixes to each rank");
  // Now apply the local/shared splitting to the temp matrix
  result = internal_CSR_loc_split(temp, partvec, mpi_rank);
  if (mpi_rank == 0)
      DEBUGLOG("Loc split done");
      
  // Setup the communication patterns for shared data
  internal_setup_communication(&result, partvec, mpi_rank, mpi_size);
  if (mpi_rank == 0)
      DEBUGLOG("Comm setup done");
  
  freeSparseMatrix(&temp);

  // // TODO: REMOVE AFTER DEBUGGING
  // char fpath[256] = {};
  // sprintf(fpath, "DEBUG_PRINT_%d.txt",mpi_rank);
  // FILE* f = fopen(fpath, "w");
  // print_CSC_SHARD(f, result);
  // fclose(f);

  return result;
}

void COMM_destroy(COMM *c){
    if(c == NULL) return;
    
    FREE_AND_NULL_IF(c->I)
    FREE_AND_NULL_IF(c->J);
    FREE_AND_NULL_IF(c->val);
    FREE_AND_NULL_IF(c->ranks);
  }
  
  void SHARD_CSC_destroy(SHARD_CSC *A){
  if(A == NULL) return;

  freeSparseMatrix(&(A->loc));
  freeSparseMatrix(&(A->shr));
  ivector_destroy(&(A->gind));
  COMM_destroy(&(A->send));
  COMM_destroy(&(A->recv));
}

// SHARD_CSR MPI_CSR_split_row(CSR big, iVector partvec) {
//   // CSR big is only defined in rank == 0

//   int mpi_rank, mpi_size;
//   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
//   MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

//   // Create result structure
//   SHARD_CSR result = {0};

//   CSR temp = {0};

//   if (mpi_rank == 0) {
//     // Split the matrix into shards
//     CSR *Aall = internal_CSR_split_row(big, mpi_size, partvec);

//     // Keep the local shard for rank 0
//     temp = Aall[0];

//     // Send each shard to its corresponding rank (1 to mpi_size-1)
//     for (int dest_rank = 1; dest_rank < mpi_size; dest_rank++) {
//       CSR shard = Aall[dest_rank];

//       // Send dimensions first
//       int dims[3] = {shard.m, shard.n, shard.nnz};
//       MPI_Send(dims, 3, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);

//       // Send the arrays (I, J, val)
//       MPI_Send(shard.I, shard.m + 1, MPI_INT, dest_rank, 1, MPI_COMM_WORLD);
//       MPI_Send(shard.J, shard.nnz, MPI_INT, dest_rank, 2, MPI_COMM_WORLD);
//       MPI_Send(shard.val, shard.nnz, MPI_DOUBLE, dest_rank, 3, MPI_COMM_WORLD);
//     }

//     // Free the array of shards (but not the shard data, which is now owned by
//     // recipients)
//     for (size_t i = 1; i < mpi_size; i++)
//       freeSparseMatrix(&Aall[i]);
//     FREE_AND_NULL(Aall);
//   } else {
//     // Other ranks receive their shards
//     int dims[3];
//     MPI_Recv(dims, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//     // Unpack dimensions
//     temp = (CSR){.m = dims[0], .n = dims[1], .nnz = dims[2]};

//     ALLOC_ARRAY(temp.I, temp.m + 1);
//     ALLOC_ARRAY(temp.J, temp.nnz);
//     ALLOC_ARRAY(temp.val, temp.nnz);

//     // Receive the arrays
//     MPI_Recv(temp.I, temp.m + 1, MPI_INT, 0, 1, MPI_COMM_WORLD,
//              MPI_STATUS_IGNORE);
//     MPI_Recv(temp.J, temp.nnz, MPI_INT, 0, 2, MPI_COMM_WORLD,
//              MPI_STATUS_IGNORE);
//     MPI_Recv(temp.val, temp.nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD,
//              MPI_STATUS_IGNORE);
//   }

//   // Synchronize all processes
//   MPI_Barrier(MPI_COMM_WORLD);


//   // Now apply the local/shared splitting to the temp matrix
//   result = internal_CSR_loc_split(temp, partvec, mpi_rank);

//   // TODO: Setup the communication patterns for shared data
//   // This requires determining what to send and receive for SpMV operations
//   // Use the shared part of the result.gind vector to determine what ranks to receive. and communicate with other ranks to determine what ranks to send

//   freeSparseMatrix(&temp);
//   return result;
// }


/**
 * Convert a CSR matrix to CSC format
 *
 * @param csr Input matrix in CSR format
 * @return A new matrix in CSC format, or NULL on failure
 */
 CSC CSR_to_CSC(CSR csr) {
  CSC csc = {.I = NULL,
             .J = NULL,
             .val = NULL,
             .nnz = csr.nnz,
             .m = csr.m,
             .n = csr.n};
  // Empty matrix case
  if (csr.nnz == 0){
    ABORT("Tried to convert empty matrix to CSC");
    // CALLOC_ARRAY(csc.I, csc.n + 1);
    // return csc;
  }

  // Allocate memory for CSC structure
  CALLOC_ARRAY(csc.I, csc.n + 1); // Column pointers (n+1 entries)
  ALLOC_ARRAY(csc.J, csc.nnz);    // Row indices
  ALLOC_ARRAY(csc.val, csc.nnz);  // Values

  if (!csc.I || !csc.J || !csc.val)
    ABORT("Failed memory allocatiion");

  // First pass: Count number of elements in each column to determine column
  // pointers
  for (int i = 0; i < csr.nnz; i++) {
    csc.I[csr.J[i] + 1]++; // csr.J contains column indices
  }

  // Cumulative sum to get column pointers
  for (int i = 1; i <= csc.n; i++) {
    csc.I[i] += csc.I[i - 1];
  }

  // Second pass: Fill in row indices and values
  // We need a temporary array to keep track of the current position in each
  // column
  int *col_counts;
  CALLOC_ARRAY(col_counts, csc.n);

  if (!col_counts)
    ABORT("Failed memory allocatiion");

  // Process each row in the CSR matrix
  for (int i = 0; i < csr.m; i++) {
    // Process all elements in this row
    for (int j = csr.I[i]; j < csr.I[i + 1]; j++) {
      int col = csr.J[j];                      // Column index in CSR
      int dest = csc.I[col] + col_counts[col]; // Position in CSC

      csc.J[dest] = i;            // Row index in CSC
      csc.val[dest] = csr.val[j]; // Copy the value
      col_counts[col]++;
    }
  }

  // Clean up temporary array
  FREE_AND_NULL(col_counts);

  return csc;
}
