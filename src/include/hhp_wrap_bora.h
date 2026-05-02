#pragma once

#include "hhp_common.h"
#include "bora_spmxv.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Aout is the partitioned matrix after the distribution.
 * Ain is the CSC matrix read using our functions. Only defined in Master node.
 * partScheme can bee either PART_BY_ROWS or PART_BY_COLUMNS.
 * partArr is the result of PaToH and is only present in master node.
 */
void distributeMatrix(buMatrix *Aout, CSC *Ain, spmxv_const partScheme, int* partArr);

/* Converts CSC matrix to buMatrix struct.
 */
void readMatrixFromCSC(CSC *Ain , buMatrix *Aout);
void readMatrixFromCSR(CSR const*const Ain , buMatrix *Aout);

/* Fills out buMatrix partition map using PaToH output.
 */
void retrieveMatrixParts(buMatrix *A, spmxv_const partScheme, int *inpartarr,
                         int *outpartarr); 

/*
   modified verson of read matrix function to handle matrix market file  
*/
void readMatrixMarket(buMatrix *A, char *fname, int partScheme, char *inpartfname,
                char *outpartfname) ;
void readMatrixMarketFromMemory(buMatrix *Aout, const CSR Ain, int partScheme, int const*const inpart,
                int const*const outpart);

void readMatrixMarketFromFile(buMatrix *A, char *fname);
#ifdef __cplusplus
}
#endif