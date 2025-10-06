import numpy as np
import logging_setup


def plu_decomposition_in_place(A):
    """
    Compute an in-place PA = LU decomposition with row pivoting.

    This follows the derivation in derivation.pdf: PA = LU where P is a row
    permutation matrix. Row exchanges are simulated via a permutation vector P.
    Additionally, this function identifies which columns are pivot columns and
    returns them as Q for compatibility with the solver.

    Parameters
    ----------
    A : numpy.ndarray, shape (m, n)
        Input matrix to be factorized in-place.

    Returns
    -------
    P : numpy.ndarray, shape (m,), dtype=int
        Row permutation vector. Row i of the permuted matrix is row P[i] of the original.

    Q : numpy.ndarray, shape (r,), dtype=int
        Pivot column indices (the columns that contain pivots), where r is the numerical rank.
        This is useful for identifying basic vs free variables.

    A : numpy.ndarray, shape (m, n)
        The matrix A modified in-place with L and U factors stored.
    """
    if A.ndim != 2:
        raise ValueError("A must be two-dimensional")

    if not np.issubdtype(A.dtype, np.number):
        raise TypeError("A's dtype must be a supported numeric type")

    m, n = A.shape

    # Initialize row permutation vector
    P = np.arange(m, dtype=int)

    # Track which columns are pivot columns
    pivot_cols = []

    # Compute tolerance for numerical rank
    eps = np.finfo(A.dtype if np.issubdtype(A.dtype, np.floating) else np.float64).eps
    max_abs = np.max(np.abs(A)) if A.size > 0 else 1.0
    # Use a more conservative tolerance for rank detection
    tol = max(m, n) * eps * max_abs * 1e3

    k_max = min(m, n)

    for k in range(k_max):
        # Find pivot column (column with largest entry in row P[k:])
        pivot_col = -1
        max_pivot = 0.0

        # Search for best pivot in remaining columns
        for j in range(k, n):
            # Find max in column j among rows P[k:]
            col_max_idx = -1
            col_max_val = 0.0

            for i_idx in range(k, m):
                i = P[i_idx]
                val = abs(A[i, j])
                if val > col_max_val:
                    col_max_val = val
                    col_max_idx = i_idx

            if col_max_val > max_pivot:
                max_pivot = col_max_val
                pivot_col = j
                pivot_row_idx = col_max_idx

        # Stop if no suitable pivot found
        if max_pivot <= tol:
            break

        # Record this as a pivot column
        pivot_cols.append(pivot_col)

        # Swap row indices in P (simulated row exchange)
        if pivot_row_idx != k:
            P[k], P[pivot_row_idx] = P[pivot_row_idx], P[k]

        # Perform Gaussian elimination on column pivot_col
        pivot = A[P[k], pivot_col]

        for i_idx in range(k + 1, m):
            i = P[i_idx]
            multiplier = A[i, pivot_col] / pivot
            A[i, pivot_col] = multiplier  # Store L multiplier

            # Update remaining columns
            for j in range(pivot_col + 1, n):
                A[i, j] -= multiplier * A[P[k], j]

    # Convert pivot columns list to array
    Q = np.array(pivot_cols, dtype=int)

    # Note: P must be a vector, not array
    return P, Q, A
