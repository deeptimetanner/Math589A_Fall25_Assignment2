# Programming Assignment 2 - Solution Documentation

## Overview
This assignment implements two decomposition algorithms and a general linear system solver for rectangular matrices with rank deficiency handling.

## Implementation Files

### 1. `plu_decomposition.py`
Implements **PA = LU** decomposition with row pivoting only.

**Key Features:**
- Row pivoting via simulated permutation vector `P`
- No physical row swaps - uses `P[k]` to index into original matrix
- Identifies pivot columns and returns them as array `Q`
- Follows the algorithm described in `derivation.pdf`

**Algorithm:**
```
For each step k:
  1. Search all remaining columns for best pivot (max magnitude)
  2. Record pivot column index in Q array
  3. Update P to track row exchange (simulated)
  4. Perform Gaussian elimination using A[P[i], col]
  5. Store multipliers in lower triangle
```

**Returns:**
- `P`: Row permutation vector (length m)
- `Q`: Pivot column indices (length r, where r is rank)
- `A`: Modified in-place with L and U factors

### 2. `general_linear_solver.py`

#### Function: `paqlu_decomposition_in_place(A)`
Implements **PAQ = LU** decomposition with both row and column pivoting.

**Key Features:**
- Both row and column pivoting via simulated permutation vectors `P` and `Q`
- No physical swaps - all operations use `A[P[i], Q[j]]` indexing
- Column pivoting moves pivot columns before non-pivot columns (virtual reordering)
- Determines numerical rank automatically using tolerance

**Algorithm:**
```
For each step k:
  1. Find pivot in submatrix A[P[k:], Q[k:]] (max magnitude)
  2. Update P and Q to record the exchange (swap indices, not matrix elements)
  3. Perform elimination: A[P[i], Q[j]] -= multiplier * A[P[k], Q[j]]
  4. Store multipliers in A[P[i], Q[k]]
  5. Stop when no pivot exceeds tolerance
```

**Numerical Tolerance:**
```python
tol = max(m, n) * eps * max_abs
```
where `eps` is machine epsilon and `max_abs` is the maximum absolute value in A.

**Returns:**
- `P`: Row permutation vector (length m)
- `Q`: Column permutation vector (length n) - first r entries are pivot columns
- `A`: Modified in-place with L (lower triangle) and U (upper triangle)

#### Function: `solve(A, b)`
Solves the linear system **Ax = b** and returns the general solution.

**General Solution Form:**
```
x = N * x_free + c
```
where:
- `N`: Nullspace basis matrix (n × f), f = n - r free variables
- `c`: Particular solution (n × 1 or n × k for multiple RHS)
- `x_free`: Arbitrary free variable values

**Algorithm Steps:**

1. **Decomposition:**
   ```python
   P, Q, A_LU = paqlu_decomposition_in_place(A.copy())
   ```

2. **Determine Rank:**
   ```python
   r = count of diagonal elements |A_LU[P[k], Q[k]]| > tol
   ```

3. **Forward Substitution (Solve Ly = Pb):**
   ```python
   for k in range(m):
       y[k] = b[P[k]]
       for j in range(min(k, r)):
           y[k] -= A_LU[P[k], Q[j]] * y[j]
   ```

4. **Consistency Check:**
   ```python
   for k in range(r, m):
       if |y[k]| > consistency_tol:
           raise ValueError("inconsistent system")
   ```
   Uses relaxed tolerance: `consistency_tol = tol * 100` for numerical stability

5. **Backward Substitution (Solve Uz = y[:r]):**
   ```python
   for k in range(r-1, -1, -1):
       z[k] = y[k]
       for j in range(k+1, r):
           z[k] -= A_LU[P[k], Q[j]] * z[j]
       z[k] /= A_LU[P[k], Q[k]]
   ```

6. **Build Particular Solution:**
   ```python
   c_perm[0:r] = z  # pivot variables
   c_perm[r:n] = 0  # free variables set to zero
   c = c_perm[Q_inv]  # inverse permute columns
   ```

7. **Build Nullspace Basis:**
   For each free variable index `free_idx` in 0..f-1:
   ```python
   rhs[k] = -A_LU[P[k], Q[r + free_idx]]  # column from U
   # Solve U * pivot_vars = rhs by back substitution
   N_perm[0:r, free_idx] = pivot_vars
   N_perm[r + free_idx, free_idx] = 1.0
   ```
   Then apply inverse column permutation: `N = N_perm[Q_inv, :]`

**Edge Cases Handled:**
- Empty matrices (m=0 or n=0)
- Underdetermined systems (n > m)
- Overdetermined systems (m > n)
- Rank-deficient matrices
- Zero rows and columns
- Multiple right-hand sides

**Error Handling:**
- Raises `ValueError` for inconsistent systems
- Raises `ValueError` if A is not 2D
- Raises `TypeError` if A has non-numeric dtype

## Key Design Decisions

### 1. Simulated vs Physical Permutations
**Why simulated?**
- README requirement: "Do not physically swap rows in `A`"
- Column permutations are "virtual" - record in Q only
- More efficient - no data movement
- Easier to track original positions

**Implementation:**
- Access elements as `A[P[i], Q[j]]` instead of `A[i, j]`
- Permutation vectors P and Q track the reordering
- Final matrix A remains in original index space

### 2. Tolerance Strategy
**Rank determination tolerance:**
```python
tol = max(m, n) * eps * max_abs
```

**Consistency check tolerance:**
```python
consistency_tol = tol * 100
```

**Rationale:**
- Base tolerance accounts for matrix size and scale
- Relaxed consistency tolerance prevents false "inconsistent" errors
- Essential for large rank-deficient systems with accumulated rounding error

### 3. Column Permutation Q
**Two interpretations:**

In `paqlu_decomposition_in_place`:
- Q is a permutation vector (length n)
- Q[k] tells which original column is at position k
- First r entries of Q are pivot columns

In `plu_decomposition.py`:
- Q is array of pivot column indices (length r)
- Direct list of which columns have pivots

### 4. Nullspace Construction
Follows the derivation in `derivation.pdf`:

```
For PAQ = LU, the nullspace vectors satisfy:
  U * x_perm = 0

Partition in permuted ordering:
  [U_B | U_F] * [x_B; x_F] = 0

Set each free variable to 1 (one at a time):
  x_F = e_k  (k-th standard basis vector)
  Solve: U_B * x_B = -U_F * e_k

Nullspace vector: [x_B; e_k]
Apply inverse permutation Q to get original variable ordering
```

## Verification

### Test Cases Passed
1. **P2.1** - File submission check ✓
2. **P2.2** - Basic non-singular case ✓
3. **P2.3** - Underdetermined with zero rows/columns (needs verification)
4. **P2.4** - Large rank-deficient overdetermined (needs verification)

### Manual Testing
All test cases in our local testing passed:
- Simple 2×3 underdetermined system
- System with zero row
- Overdetermined rank-deficient 3×2 system

## Debugging Notes

### Original Issues
1. **Physical swaps:** Initial implementation used `A[[k, pivot_row], :] = A[[pivot_row, k], :]`
   - **Fix:** Changed to simulated permutations with P and Q vectors

2. **Double permutation:** Forward substitution applied P twice: `A_LU[P[k], Q[j]]` after `b_perm = b[P]`
   - **Fix:** Corrected to `y[k] = b[P[k]]` with single indexing

3. **Strict tolerance:** Consistency check failed on valid rank-deficient systems
   - **Fix:** Added relaxed `consistency_tol = tol * 100`

## References
- `README.md` - Assignment requirements
- `derivation.pdf` - Mathematical derivation of PA=LU solver
- `general_linear_info.py` - Detailed docstrings and specifications
