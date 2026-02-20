#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

def lu_factorization(A):
    """Performs LU factorization with partial pivoting to decompose matrix A into P, L, and U."""
    n = A.shape[0]
    P = np.eye(n)
    L = np.eye(n)
    U = A.astype(float).copy()

    for i in range(n):
        # Partial Pivoting: Find the row with the largest absolute value in the current column
        pivot_row = np.argmax(abs(U[i:, i])) + i
        
        # Stop and report if the matrix is singular (no nonzero pivots exist)
        if U[pivot_row, i] == 0:
            raise ValueError("The matrix is singular and cannot be factorized.")

        # Swap rows in U and the permutation matrix P
        U[[i, pivot_row]] = U[[pivot_row, i]]
        P[[i, pivot_row]] = P[[pivot_row, i]]
        
        # Swap rows in L for the elements already computed
        if i > 0:
            L[[i, pivot_row], :i] = L[[pivot_row, i], :i]

        # Elimination step to fill L and update U
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, :] -= factor * U[i, :]
            
    return P, L, U

def forward_backward_solver(L, U, P, b):
    """Solves the linear system Ax = b using forward substitution for L and backward substitution for U."""
    # Apply the permutation matrix to the right-hand side vector b
    b_permuted = np.dot(P, b)
    n = L.shape[0]
    
    # Forward substitution: Solve Ly = Pb
    y = np.zeros_like(b_permuted)
    for i in range(n):
        y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])
        
    # Backward substitution: Solve Ux = y
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
    return x


# In[ ]:




