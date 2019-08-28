#distutils: extra_compile_args = -fopenmp -O1
#distutils: extra_link_args = -fopenmp

#
#  Copyright (c) 2016 Intel Corporation. All Rights Reserved.
#
#  Portions of the source code contained or described herein and all documents related
#  to portions of the source code ("Material") are owned by Intel Corporation or its
#  suppliers or licensors.  Title to the Material remains with Intel
#  Corporation or its suppliers and licensors.  The Material contains trade
#  secrets and proprietary and confidential information of Intel or its
#  suppliers and licensors.  The Material is protected by worldwide copyright
#  and trade secret laws and treaty provisions.  No part of the Material may
#  be used, copied, reproduced, modified, published, uploaded, posted,
#  transmitted, distributed, or disclosed in any way without Intel's prior
#  express written permission.
#
#  No license under any patent, copyright, trade secret or other intellectual
#  property right is granted to or conferred upon you by disclosure or
#  delivery of the Materials, either expressly, by implication, inducement,
#  estoppel or otherwise. Any license under such intellectual property rights
#  must be express and approved by Intel in writing.
#
cimport cython
import numpy as np
cimport openmp
from libc.math cimport log
from cython.parallel cimport prange
from cython.parallel cimport parallel
cimport openmp

THOUSAND = 1024
FACTOR = 100
NUM_TOTAL_ELEMENTS = FACTOR * THOUSAND * THOUSAND
X1 = -1 + 2*np.random.rand(NUM_TOTAL_ELEMENTS)
X2 = -1 + 2*np.random.rand(NUM_TOTAL_ELEMENTS)
Y = np.zeros(X1.shape)

def test_serial():
    serial_loop(X1,X2,Y)


cdef serial_loop(double[:] A, double[:] B, double[:] C):
    cdef:
        int N = A.shape[0]
        int i
    #pragma omp parallel for 
    for i in range(N):
        C[i] = log(A[i]) * log(B[i])

def test_parallel():
    parallel_loop(X1,X2,Y)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef parallel_loop(double[:] A, double[:] B, double[:] C):
    cdef: 
        int N = A.shape[0]
        int i
    with nogil:
        for i in prange(N, schedule='dynamic'): C[i] = log(A[i]) * log(B[i])

