[TOC]



# Linear Algebra

## Vectors and Matrices

* Linear combination: $c v+d w=c\left[\begin{array}{l}
  {1} \\
  {1}
  \end{array}\right]+d\left[\begin{array}{l}
  {2} \\
  {3}
  \end{array}\right]=\left[\begin{array}{c}
  {c+2 d} \\
  {c+3 d}
  \end{array}\right]$
* Dot product: $\boldsymbol{v} \cdot \boldsymbol{w}=v_{1} w_{1}+v_{2} w_{2}$. 
* Dot product = 0 indicates orthogonality: $v \cdot w = 0 \iff v \perp w$
* Length: $||v|| = \sqrt{v \cdot v}$
* Schwarz Inequality: $|\boldsymbol{v} \cdot \boldsymbol{w}| \leq\|\boldsymbol{v}\|\|\boldsymbol{w}\|$
* Triangle Inequality: $\|\boldsymbol{v}+\boldsymbol{w}\| \leq\|\boldsymbol{v}\|+\|\boldsymbol{w}\|$

### Matrix x Vector Multiplication

* Row multiplication: $A x=\left[\begin{array}{ll}
  {(\operatorname{row} I)} & {\cdot x} \\
  {(\operatorname{row} 2)} & {\cdot x} \\
  {(\operatorname{row} 3)} & {x}
  \end{array}\right]$
  * Example: $A \boldsymbol{x}=\left[\begin{array}{rrr}
    {1} & {0} & {0} \\
    {-1} & {1} & {0} \\
    {0} & {-1} & {1}
    \end{array}\right]\left[\begin{array}{l}
    {x_{1}} \\
    {x_{2}} \\
    {x_{3}}
    \end{array}\right]=\left[\begin{array}{c}
    {(1,0,0) \cdot\left(x_{1}, x_{2}, x_{3}\right)} \\
    {(-1,1,0) \cdot\left(x_{1}, x_{2}, x_{3}\right)} \\
    {(0,-1,1) \cdot\left(x_{1}, x_{2}, x_{3}\right)}
    \end{array}\right]$
* Column multiplication: $A x=x(\operatorname{column} 1)+y(\operatorname{column} 2)+z(\operatorname{column} 3)$
  * Example: $Ax = \left[\begin{array}{rrr}
    {2} & {5} \\
    {1} & {3} 
    \end{array}\right]\left[\begin{array}{l}
    {x_{1}} \\
    {x_{2}}
    \end{array}\right]=x_1\left[\begin{array}{c}
    {2} \\
    {1}
    \end{array}\right] + x_2\left[\begin{array}{c}
    {5} \\
    {3}
    \end{array}\right] $

### Matrix Multiplication

* Matrix multiplication is associative and distributive but not commutative: (AB)C = A(BC)
* To multiply AB, if A has n columns, B must have n rows. Left columns = Right rows

1. Dot product each row in A with column in B 
   * The entry in row i and column j of AB is (row i of A) · ( column j of B)
2. Matrix A times every column of B
   * Each column of AB is a combination of the columns of A.
   * $A\left[b_{1} \cdots b_{p}\right]=\left[A b_{1} \cdots A b_{p}\right]$
3. Every row of A times matrix B
   * Each row of AB is a combination of the rows of B.
   * $\left[\begin{array}{cc}
     {\operatorname{row} i} & {\text { of } A}
     \end{array}\right]\left[\begin{array}{ccc}
     {1} & {2} & {3} \\
     {4} & {5} & {6} \\
     {7} & {8} & {9}
     \end{array}\right]=\left[\begin{array}{cc}
     {\operatorname{row} i} & {\text { of } A B}
     \end{array}\right]$
4. Multiply columns 1 ton of A times rows 1 ton of B. Add those matrices.
   * Column 1 of A multiplies row 1 of B. Columns 2 and 3 multiply rows 2 and 3.
   * $\left[\begin{array}{ccc}
     {\operatorname{col} 1} & {\operatorname{col} 2} & {\operatorname{col} 3} \\
     {\cdot} & {\cdot} & {\cdot} \\
     {\cdot} & {\cdot} & {\cdot}
     \end{array}\right]\left[\begin{array}{cc}
     {\operatorname{row} 1} & {\cdots} \\
     {\operatorname{row} 2} & {\cdots} \\
     {\operatorname{row} 3} & {\cdots}
     \end{array}\right]=(\operatorname{col} 1)(\operatorname{row} 1)+(\operatorname{col} 2)(\operatorname{row} 2)+(\operatorname{col} 3)(\operatorname{row} 3)$
   * $A B=\left[\begin{array}{l}
     {a} \\
     {c}
     \end{array}\right]\left[\begin{array}{ll}
     {E} & {F}
     \end{array}\right]+\left[\begin{array}{l}
     {b} \\
     {d}
     \end{array}\right]\left[\begin{array}{ll}
     {G} & {H}
     \end{array}\right] = \left[\begin{array}{cc}
     {\boldsymbol{a} \boldsymbol{E}+b G} & {\boldsymbol{a} \boldsymbol{F}+b H} \\
     {\boldsymbol{c} \boldsymbol{E}+d G} & {\boldsymbol{c} \boldsymbol{F}+d H}
     \end{array}\right]$

* Block multiplication - if blocks of A are the right size to multiply blocks of B, can divide into smaller matrices and multiply
  * $\left[\begin{array}{ll}
    {A_{11}} & {A_{12}} \\
    {A_{21}} & {A_{22}}
    \end{array}\right]\left[\begin{array}{l}
    {B_{11}} \\
    {B_{21}}
    \end{array}\right]=\left[\begin{array}{l}
    {A_{11} B_{11}+A_{12} B_{21}} \\
    {A_{21} B_{11}+A_{22} B_{21}}
    \end{array}\right]$

### Inverses

* $A^{-1}A = I = AA^{-1}$ if the inverse exists r = m = n
* For square matrices, a left inverse = right inverse
* If singular, det = 0, inverse does not exist

#### Left Inverse

* Matrix with full column rank r = n, N(A) = {0}, independent columns, 0 or 1 solutions to Ax = b
* $(A^TA)^{-1}A^TA = I$ so the left inverse is $(A^TA)^{-1}A^T$
* $AA_{left}^{-1}$ is then a projection onto the column space
* Used in least squares

#### Right Inverse

* Matrix with full row rank r = m, $N(A^T) = \{0\}$ independent rows, infinite solutions to Ax = b, n - m free variables
* $AA^T(AA^T)^{-1}= I$ so the right inverse is $A^T(AA^T)^{-1}$
* $A_{right}^{-1}A$ is then a projection onto the row space

#### Pseudoinverse

* $y = A^+(Ay)$
* If x, y both in the row space, then $Ax \neq Ay$ - there is a one to one mapping from x to Ax. 
* Pseudoinverse often used in stats since least squares matrices often are not of full rank
* Finding pseudoinverse
  * SVD: $A = U\Sigma V^T$
  * Here $\Sigma = \left[\begin{array}{ccc}\sigma_1 & 0 & ...\\ ... & \sigma_r & ... \\ 0 & 0 & 0 \end{array}\right]$ has rank r, n x m
  * $\Sigma^{+} = \left[\begin{array}{ccc}1/\sigma_1 & 0 & ...\\ ... & 1/\sigma_r & ... \\ 0 & 0 & 0 \end{array}\right]$ n x m, rank r
  * Then $A^+ = V\Sigma^+ U^T$
* Note that $\Sigma \Sigma^+$ is m x m with diagonal ones and is a projection onto row space, $\Sigma^+ \Sigma$ an n x n matrix projecting onto column space.

### Permutations

* A matrix that executes row exchanges - this may be needed to make a matrix invertible. PA = LU
* P = identity matrix with reordered rows. n! possible matrices / reorderings
* For invertible P, $P^{-1} = P^T$

### Transposes

* Switch columns and rows - $(A^T)_{ij} = A_{ji}$
* For symmetric matrices, $A^T= A$. 
* Note $A^TA$ is always symmetric, square.

### Echelon Form

* Staircase from top corner separating values from 0’s, ie. U or L
* R = reduced row echelon form. $R = \left[\begin{array}{rrr}
  {I} & {F} \\
  {0} & {0} 
  \end{array}\right]$
  * Take matrix to echelon form, then make 0’s above and below all pivots 
  * Normalize each row to make the pivots equal to 1. 
  * Matrix forms I in the pivot rows and columns. The R form above has r pivot columns that make up I and n-r free columns that make up F

## Elimination and Factorization

### Manual Elimination

* Pick pivot in the first row - pivot must not be zero. If we have a zero pivot, can try to exchange rows to produce a non-zero pivot. 
* Reduce the numbers below pivot to zero using linear combinations of rows
* For well-behaved matrix, will reduce to U, an upper triangular matrix.
* Back Substitution
  * Perform elimination on augmented matrix, eg. $\left[\begin{array}{rrr}
    {2} & {5} \\
    {1} & {3} 
    \end{array}\Big|\begin{array}{rrr}
    {1} \\ {2} 
    \end{array}\right]$
  * Use upper triangular equations to solve for variables in reverse order $x_1, ...,x_n$. At each step have one additional unknown in each equation

### Elimination in Matrix Form

* All elimination steps could be combined into a single matrix E, that transforms A into U, ie. EA = U

### Factorization A= LU

* From elimination we have EA = U for some unknown E. For $L = E^{-1}$, then get A = LU
* L adds back to U what E removed from A. If there are no row exchanges, L is just made of the column multipliers

## Vector Spaces

* All vector spaces contain the origin. 
* Vectors form a subspace if all linear combinations of those vectors are also in the subspace. Vector space must be closed under linear combinations.
* All subspaces of $\R^3$: $\R^3$, plane through the origin, line through the origin, zero vector only
* Rank of A = # of pivots from elimination

### Solving Ax = 0

* Elimination does not change solutions, so null space is unchanged by elimination
* Can solve Ux = 0 instead then after elimination. Left with pivot columns and free columns, the columns without pivots in which any value can be assigned to the x’s corresponding to that number column
* Pivot variables can be found through back substitution, free columns we choose values freely. Set values or free columns to 1 and 0 - this forms our special solution
* The null space contains all the combinations of the special solutions. There is one special solution per free variable, and the number of free variables is n - r

### Solving Ax = b

* Typical approach - augmented matrix -> elimination
* Solvability condition - Ax = b is solvable only when b is in C(A)
* Finding complete solution
  1. Set all free variables to 0, solve Ax= b for the pivot variables. This gives us a particular solution
  2. Find solutions in the null space
  3. Take linear combination of particular solution and null space solutions. $X_{total} = X_p + X_n$. $Ax_p = b$ + $Ax_n =0 $ = $A(x_n+x_p) = b$. With a particular solution, can add anything in the null space and still get b
  4. $X_n$ is combination of the special solutions - $X_{total} = X_p + c_1X_{SS1} + c_2X_{SS2}...$

### Rank

* \# of pivot columns
* Dimension of C(A) column space

#### Full Column Rank

* r = n < m
* No free variables, N(A) = {0}
* Solution to Ax = b - unique if a solution exists -> 0 or 1 solutions
* $R = \left[\begin{array}{c} I \\ 0 \end{array}\right]$

#### Full Row Rank

* r = m < n
* Solution exists for Ax = b for all b due to free variables
* Have n - r = n - m free variables
* $R = \left[\begin{array}{cc} I & F \end{array}\right]$

#### Full Row + Column Rank

* r = m = n -> square matrix, always invertible (defines invertibility)
* N(A) = {0}
* Ax = b has 1 solution - see this by combing rules for full row and full col rank
* $$R = \left[\begin{array}{c} I  \end{array}\right]$$

#### Singular Matrix r < m, n

* Ax = b has 0 or $\infty$ solutions
* $R = \left[\begin{array}{c} I & F \\ 0  & 0\end{array}\right]$

### Independence, Span, Basis

* Vectors $x_1, ...,x_n$ are independent if no combination gives the zero vector (except the zero combination with scalars = 0): $c_1x_1 + ... + c_nx_n \neq 0$

* Columns are independent if N(A) = {0} $\iff$ rank = n

* Columns are dependent if some vector is in the null space $\iff$ rank < n . Think of 3 vectors in a plane - these must be dependent

* Span - vectors $v_1, ..., v_n$ span a space means a space consists of all combinations of these vectors

* Basis - for a vector space, a basis is a sequence of vectors which

  1. are Independent
  2. span the space

  * A basis is not unique, but all bases for a space have the same number of vectors. In $\R^n$ need n vectors to form basis - this is the dimension D of the space

* Dimension D = number of vectors needed to form a basis

### The Four Subspaces

#### Column Space

* $C(A) \in \R^m$
* C(A) = all linear combinations of the columns
* We can solve Ax = b when b is in the column space
* dim(C(A)) = # of pivot columns = rank(A) = r
* Basis = pivot columns

#### Null Space

* $N(A) \in \R^n$
* All solutions x to equation Ax = 0
* Zero vector is always in the null space
* The null space contains all the combinations of the special solutions. There is one special solution per free variable, and the number of free variables is n - r
* Null space matrix N, RN = 0. $N = \left[\begin{array}{c} -F \\ I \end{array}\right]$
* dim(N(A)) = # of free variables = n - r
* Basis = special solutions

#### Row Space

* All linear combination of the rows or the column space of $A^T$, $C(A^T)$
* $C(A^T) \in \R^n$
* $dim(C(A^T)) = r$

#### Null Transpose Space

* Nullspace of $A^T$, $N(A^T)$, the left nullspace of A
  * For $A^Ty=0, \, y \in N(A^T) \implies y^TA = 0$
* $N(A^T) \in \R^m$
* $dim(N(A^T)) = m - r$, the number of free columns in $A^T$

### Matrix Spaces

* S = symmetric, U = upper triangular
* $S \cap U = $ symmetric + upper triangular = diagonal. S+U is the linear combinations of the matrices in the two spaces
* $dim(S) + dim(U) = dim(S \cap U) + dim(S+U)$

## Orthogonality

### Vector Orthogonality

* For two vectors $x \cdot y = x^Ty =0 \iff x \perp y$
* $x \perp y \implies ||x||^2 + ||y||^2 = ||x+y||^2$
* Zero vector orthogonal to all other vectors

### Subspace Orthogonality

* Subspaces $S \perp T \implies$ all vectors in S perp to all vectors in T
* $C(A^T) \perp N(A)$: Row Space orthogonal to Null Space
  * Ax = 0 for x in N(A)
  * Then by defn x is perpendicular to each row in A using matrix multiplication
  * $(c_1row_1 + c_2row_2...)^Tx = 0$
* $C(A) \perp N(A^T)$: Column space orthogonal to left nullspace
* Orthogonal complements: a complement contains all vectors perp to the other space
  * Null space and row space are orthogonal complements in $\R^n$

### Projections

* $A^TA$ invertible $\iff$ A has independent columns. Share a rank and null space
* p = projection of b onto a. Since p lies on a, it is a scalar multiple of a: $p = xa$. Left to find x
* $x = \frac{a^Tb}{a^Ta}$
  * Define e = b - p = orthogonal vector from b to a vector. Therefore $a \perp b- p$
  * Then $a^T(b - xa) = 0 \implies xa^Ta = a^Tb \implies x = \frac{a^Tb}{a^Ta}$
* $p =xa= a\frac{a^Tb}{a^Ta} = \frac{aa^T}{a^Ta}b = Pb$ for 
* Projection Matrix $P = \frac{aa^T}{a^Ta}$
  * $P^T = P$ - symmetric
  * $P^2 = P$ - projection of projection same as single projection
* Point of projection - Ax = b may have no solutions, ie. b not in C(A). Can instead solve $A\hat{x} = p$, where p is the projection of b onto the column space and $\hat{x}$ is the solution to this altered problem.
* Higher dimensions - $p = A\hat{x}$
  * Key is $e = b - A\hat{x} \perp plane$
  * **$A^TA\hat{x} = A^Tb$** 
  * **$\hat{x} = (A^TA)^{-1}A^Tb$**
  * projection matrix $P = A(A^TA)^{-1}A^T$. The inverse cannot be distributed bc A not necessarily square - $A^TA$ is square however
* Some implications:
  * If b in C(A), $Pb = b$. Derived from $b = Ax \implies A(A^TA)^{-1}A^TAx = Ax = b$
  * If $b \perp C(A), \, Pb =0$. Derived from $A(A^TA)^{-1}A^Tb = A(A^TA)^{-1}(0) = 0$

### Least Squares

* Given number of non-collinear points, have some line $Ax = b$ with errors $||e||^2 = e^2_1 + e_2^2 + ...$
* We take points on the line $p_1, p_2, p_3, ...$ instead of original points $b_1, b_2, b_3,...$
* Use $A^TA\hat{x} = A^Tb$ to derive normal equations. The partial derivatives of $||Ax- b||^2$ are zero when $A^TA\hat{x} = A^Tb$
* If A has independent columns, then $A^TA$ is invertible is crucial to making this work. 
  * If $A^TAx = 0$, then x can only be the zero vector

### Orthogonal Matrices

* Orthonormal vectors: $q^T_iq_j = \begin{cases} 0 & if \; i = j \\ 1 & if \; i \neq j\end{cases}$
* $Q = \left[\begin{array}{c} q_1 q_2...q_n\end{array}\right]$
* $Q^TQ = I$. For square Q, this implies $Q^T = Q^{-1}$
* Q has orthonormal columns, to project onto its column space $P = Q(Q^TQ)^{-1}Q^T = QQ^T$
  * If P is square, then projection is onto whole space $QQ^T = I$

### Gram-Schmidt

* Process to find orthonormal projection
* For independent vectors a, b, c, let A, B, C be orthogonal, then $q_1 = \frac{A}{||A||}, \, q_2 = \frac{B}{||B||}, q_3 = \frac{C}{||C||}$
* Let a = A, then need to change b to be orthogonal to a. Requires B = e, the error vector
* $B = b - \frac{A^Tb}{A^TA}A$, then $A^TB = 0 \implies A\perp B$
* To make C, need a third vector orthogonal to both A, B by subtracting off the components in the a, b directions
* $C = c - \frac{A^Tc}{A^TA}A - \frac{B^Tc}{B^TB}B$
* A = QR
  * Basic expression of G-S
  * $A = \left[\begin{array}{c} a_1 a_2\end{array}\right] =  \left[\begin{array}{c} q_1 q_2\end{array}\right]R$ for R upper triangular

## Determinants

* A number associated with every **square** matrix. Test for invertibility when $det(A) \neq 0$

### Properties

1. $det\; I = 1$

2. Row exchanges: for each row exchange, reverse the sign of the determinant

   * Det of permutations either +- 1 depending on if we do even or odd number of exchanges

3. Scalar factoring and linear function

   a. Scalar factor can be pulled out of a row: $\left[\begin{array}{c} ta & tb \\ c & d\end{array}\right] = t \left[\begin{array}{c} a & b \\ c & d\end{array}\right]$

   b. Determinant is a linear function of rows (within row, not globally $det(A+B) \neq det(A) + det(B)$): $\left[\begin{array}{c} a + a' & b + b' \\ c & d\end{array}\right] = \left[\begin{array}{c} a & b \\ c & d\end{array}\right] + \left[\begin{array}{c} a' & b' \\ c & d\end{array}\right]$

4. 2 equal rows $\implies det = 0$

   * Proof: exchange the equal rows, the determinant should be the same since matrix is unchanged but violates property 2. Therefore must be 0

5. Subtraction scaled row l from row k $\implies$ determinant does not change, ie. elimination does not change determinant

   * $\left[\begin{array}{c} a & b \\ c - la & d - lb\end{array}\right] = \left[\begin{array}{c} a & b \\ c & d\end{array}\right] + \left[\begin{array}{c} a & b \\ -la & -lb\end{array}\right] = \left[\begin{array}{c} a & b \\ c & d\end{array}\right] + -l\left[\begin{array}{c} a & b \\ a & b\end{array}\right] = \left[\begin{array}{c} a & b \\ c & d\end{array}\right]$
   * Proved using properties 3b, 3a, and 4

6. Row of 0’s $\implies det = 0$

   * ie. elimination gives zero row - singular and non-invertible
   * Say t = 0, then det = 0 by 3b

7. Product of diagonals for triagular matrix: $det \; U = d_1 \times d_2 \times ... \times d_n$

   * Product of pivots after elimination (with sign determined by row exchanges too)
   * Using properties 5, 3a, 1 could make diagonal and factor out d’s: $d_1...d_n(I)$

8. $det \: A =0 \iff$ A is singular

   * Another way of seeing A is invertible only when have pivots of full rank

9. $det\;AB = (det \; A)(det \; B)$

10. $det\; A^T = det\;A$

### Cofactors

* $det \; A = \sum_{n! \;terms} \pm a_{1\alpha}a_{2\beta}a_{3\gamma}...a_{n\omega}$
* For 3x3: $\left|\begin{array}{ccc} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{array}\right| = a_{11}a_{22}a_{33} + (-1)a_{11}a_{23}a_{32} + a_{12}a_{21}a_{33} + (-1) a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} + (-1)a_{13}a_{22}a_{31}$
* Basic approach: determinant of smaller sub matrix: $\left|\begin{array}{ccc} a_{11} &  &  \\  & a_{22} & a_{23} \\  & a_{32} & a_{33}\end{array}\right|$
* Cofactor of $a_{ij} = C_{ij}$, given + if i + j is even, - if i+j odd: $\left|\begin{array}{ccc} + & - & + \\ - & + & - \\ + & - & +\end{array}\right|$
* Cofactor formula: $det \; A = a_{11}C_{11} + ... + a_{1n}C_{1n}$ along a single row (here row 1)

### Cramer’s Rule



## Eigenvalues and Eigenvectors

* $Ax = \lambda x$ - for a square matrix A
  * eigenvector - a vector that fits the property Ax || x
  * eigenvalue - some scalar value that allows eigenvector property to hold
* If A is singular, 0 will be an eigenvalue
* Projection matrix - Px = x with $\lambda = 1$ for vectors in the plane, $\lambda = 0$ for x fitting Px = 0
* Permutation matrix - $\lambda = 1, -1$
* Trace  = sum down diagonal of A = $a_{11} + a_{22} + ... + a_{nn}$

### Tricks

* With symmetric matrices, eigenvalues will always be real
* Eigenvalues are always complex conjugates
* Triangular matrix - eigenvalues are just the diagonal values

### General Procedure

* Write $Ax = \lambda x$
* Rewrite as $(A - \lambda I)x = 0$, giving us a singular matrix $(A - \lambda I)$ with det = 0
* Get characteristic equation from $det(A - \lambda I) = 0$
* Solve for $\lambda$ 
* An n x n matrix will have n eigenvalues, though could be repeated
* Once lambdas found, find x with elimination, finding the null space of the singular matrix with lambda values plugged in 
* If repeated eigenvalue, will have fewer eigenvectors
* **Eigenvalues sum to trace, multiply to determinant**

### Diagonalization

* Suppose n linearly independent eigenvectors of A - form columns of matrix S
* $AS = A\left[\begin{array}{c}x_1x_2...x_n\end{array}\right] = \left[\begin{array}{c}\lambda_1x_1...\lambda_nx_n\end{array}\right] = \left[\begin{array}{c}x_1x_2...x_n\end{array}\right]\left[\begin{array}{c}\lambda_1 & 0 & ... & 0\\ 0& \lambda_2 & ... & 0 \\ 0& 0 & ... & \lambda_n\end{array}\right] = S\Lambda$
* Key: $ A = S\Lambda S^{-1}$
* Powers of A: $A^K = S \Lambda^KS^{-1}$
* Theorem: $A^K \rightarrow 0 \text{ as } k \rightarrow \infty$ if all $|\lambda_i| < 1$
  * dependent on assumption of n independent eigenvectors, otherwise cannot diagonalize
* A is sure to be diagonalizable if all eigenvalues are different. If eigenvalues repeated, may or may not have n independent eigenvectors
* Can use to solve recursive formulae: $u_{k+1} = Au_k$ then can solve using $u_{k+1} = A^{k+1}u_0$

### Symmetric Matrices

* Eigenvalues are always real and eigenvectors can be chosen to be orthogonal
* While usually have $A = S\Lambda S^{-1}$, now have $A = Q\Lambda Q^{-1}= Q\Lambda Q^{T}$ (can make the eigenvectors orthonormal to fit Q defn)
* Notation for complex conjugates: x is conjugate of $\bar{x}$
* Defn symmetric $A = A^T$ for real, otherwise $A = \bar{A}^T$. $A = Q\Lambda Q^T = \left[\begin{array}{c}q_1...q_n\end{array}\right]\left[\begin{array}{c}\lambda_1 & 0 & ... & 0\\ 0& \lambda_2 & ... & 0 \\ 0& 0 & ... & \lambda_n\end{array}\right] \left[\begin{array}{c}q_1\\...\\q_n\end{array}\right] = \lambda_1q_1q_1^T + ... \lambda_nq_nq_n^T$
* Each $qq^T$ is a projection matrix - every symmetric matrix is a combination of perpendicular projection matrices
* For symmetric matrices, signs of the pivots are the same as the signs of the eigenvalues: # pos pivots = # pos eigenvalues

### Positive Definite Matrices

* Symmetric matrices with only positive eigenvalues
* All pivots are positive, all subdeterminants are positive
* Tests
  1. Eigenvalues: $\lambda > 0 \;\forall \lambda$
  2. Determinant: $a > 0, \; ac - b^2 > 0$
  3. Pivots: $a > 0,\; \frac{ac-b^2}{a} > 0$
  4. Key Test: $x^TAx > 0$
* $x^TAx$ produces a quadratic form: $A = \left[\begin{array}{c} 2 & 6 \\ 6 & 20\end{array}\right] \implies x^TAx = \left[\begin{array}{c}x_1&x_2\end{array}\right]\left[\begin{array}{c}2x_1 + 6x_2 \\6x_1 + 20x_2\end{array}\right] = 2x_1^2 + 12x_1x_2 + 20x^2_2$
* det = 4, trace = 22 - then both eigenvalues must be positive. $x^TAx > 0$ except at x = 0
* Intuition is we need the squares to overwhelm the combined term. Notice $a_{11},\;a_{22}$ are the cofficients for the squares
* Essentially minimizing $f(x,y) = 2x^2 + 12xy + 20y^2 = 2(x+3y)^2 + 2y^2$ - factoring using complete the square shows the sum of two squares -> always positive
* If A,B pos def, then A + B also pos def. Notice for least squares, $A^TA$ is square and symmetric, so $x^TA^TAx = (Ax)^T(Ax) = ||Ax||^2 \geq 0$. Only 0 when the vector is 0, so for any matrix of rank n can say least squares will be positive definite

### Complex Matrices

* For complex $z = \left[\begin{array}{c}z_1\\...\\z_n\end{array}\right] \in \C^n$, cannot use $z^Tz$ for length squared since it is negative!
* Instead use $\bar{z}^Tz = ||z||^2$, where z-bar is the complex conjugate (flipped sign on complex part). Eg. $z = \left[\begin{array}{c}1\\i\end{array}\right], \; \bar{z} = \left[\begin{array}{c}1\\-i\end{array}\right]$
* Hermetian: $z^Hz = \bar{z}^Tz$ - use this for the inner product when dealing with complex vectors
* Hermetian matrices: $A^H = A$ - real eigenvalues, orthogonal eigenvectors. Just like real symmetric matrices

### Similar Matrices

* For 2 square matrices, A and B are similar if they have the same eigenvalues (easily checked with trace / determinant)
* For some matrix, can factor B: $B = M^{-1}AM$
* Have already seen a similar matrix in $\Lambda = S^{-1}AS \implies$ A is similar to $\Lambda$
* All matrices with same eigenvalues of A are similar to A and can be transformed via some M - form families 
* If $\lambda_1 = \lambda_2$, then might not be diagonalizable depending if there is one eigenvector or two - this will split into different families. 

#### Jordan Form

* Take the most diagonalizable family of similar matrices. Not always easy to do in practice since we need exactly the same eigenvalues
* Create jordon blocks that contain a single eigvector: $J_i = \left[\begin{array}{c}\lambda_i & 0 & ... & 0\\ 0& \lambda_i & ... & 0 \\ 0& 0 & ... & \lambda_i\end{array}\right]$
* Every square matrix A is similar to a jordan matrix made of these jordon blocks.
* \# blocks = # of eigenvectors

## Singular Value Decomposition

* $A = U\Sigma V^T$ for $\Sigma$ diagonal, U,V orthogonal
* Special case of positive definite: $A = Q\Lambda Q^T$
* Basic Idea: for V a basis in the row space, U a basis in the col space, and for sigma scaling factors:
  * $AV = U\Sigma \implies A\left[\begin{array}{c}v_1...v_r\end{array}\right]= \left[\begin{array}{c}u_1...u_r\end{array}\right]\left[\begin{array}{c}\sigma_1 & &\\ & \sigma_2 & \\& &...\end{array}\right]$
  * $A = U\Sigma V^{-1} = U\Sigma V^T$

### Procedure

* Using $A = \left[\begin{array}{cc} 4 & 4  \\ -3 & 3\end{array}\right]$

1. $A^TA = \left[\begin{array}{cc} 4 & -3  \\ 4 & 3\end{array}\right]\left[\begin{array}{cc} 4 & 4  \\ -3 & 3\end{array}\right] = \left[\begin{array}{cc} 25 & 7  \\ 7 & 25\end{array}\right]$
2. Find eigens of $A^TA$: $x_1 = \left[\begin{array}{cc} 1 \\ 1\end{array}\right], \; x_2 = \left[\begin{array}{cc} 1 \\ -1\end{array}\right]$ and $\lambda_1 = 32, \; \lambda_2 = 18$. Normalize eigenvectors: divide by length (here $\sqrt{2}$)
3. Set up: $A = \left[\begin{array}{cc} 4 & 4  \\ -3 & 3\end{array}\right] = A = \left[\begin{array}{cc}  &   \\  & \end{array}\right]\left[\begin{array}{cc} \sqrt{32} & 0  \\ 0 & \sqrt{18}\end{array}\right]\left[\begin{array}{cc} 1/\sqrt{2} & 1/\sqrt{2}  \\ 1/\sqrt{2} & -1/\sqrt{2}\end{array}\right] = U\Sigma V^T$
4. Find U’s. $AA^T$ is a positive definite symmetric matrix. $AA^T = U\Sigma V^T V \Sigma^T U^T = U\Sigma\Sigma^T U^T$
   * Calc $AA^T$: $AA^T = \left[\begin{array}{cc} 4 & 4  \\ -3 & 3\end{array}\right]\left[\begin{array}{cc} 4 & -3  \\ 4 & 3\end{array}\right] = \left[\begin{array}{cc} 32 & 0  \\ 0 & 18\end{array}\right]$
   * $\lambda’s$ of $AA^T$ are the same as for $A^TA$ - (32, 18). $x_1 = \left[\begin{array}{cc} 1 \\ 0\end{array}\right], \; x_2 = \left[\begin{array}{cc} 0 \\ 1\end{array}\right]$
   * Then $U = x_1 = \left[\begin{array}{cc} x_1 & x_2\end{array}\right] = \left[\begin{array}{cc} 1 & 0  \\ 0 & 1\end{array}\right]$
5. $A = U\Sigma V^T = \left[\begin{array}{cc} 1 &  0 \\ 0 & 1\end{array}\right]\left[\begin{array}{cc} \sqrt{32} & 0  \\ 0 & \sqrt{18}\end{array}\right]\left[\begin{array}{cc} 1/\sqrt{2} & 1/\sqrt{2}  \\ 1/\sqrt{2} & -1/\sqrt{2}\end{array}\right] $

## Linear Transformations

* Follow two rules
  1. $T(v+w) = T(v) + T(w)$
  2. $T(cV) = cT(v)$
* If we want to use a matrix, we need to use coordinates that the transformation is relative to: $T(v) = Av$
* Coordinates come from a basis - $v = c_1v_1 + ... + c_nv_n$. We typically assume the standard basis but not necessary

### Constructing a Transformation Matrix

* $T: \R^n \rightarrow \R^m$
* Choose basis $v_1,...,v_n$ for inputs and $w_1,...,w_m$ for outputs
* For projection onto a line, choose v1 = line itself, v2 = vector perpendicular to line. Then for $v = c_1v_1 + c_2v_2$, $T(v)= c_1v_1 \text{ taking } (c_1, c_2) \rightarrow (c_1, 0)$. Then $A = \left[\begin{array}{cc}1 & 0 \\ 0& 0\end{array}\right]$
* Easiest choice for a transformation matrix is the eigenvector basis, since this leads to transformation $\Lambda$
* To find A given a basis, let the first column of A equal $T(v_1)=a_{11}w_1 + ... + a_{m1}w_m$, the second column equal $T(v_2)=a_{12}w_1 + ... + a_{m2}w_m$, etc...

### Change of Basis

* W matrix of new basis vectors as columns
* To go to vector x in new basis from c in old basis, $x = Wc$
* Transforming between bases is equivalent to similar matrices. M is a change of basis matrix in $B = M^{-1}AM$

## Applications

### Markov Matrices

* Example: $A = \left[\begin{array}{ccc}.1 & .01 & .3 \\ .2 & .99 & .3 \\ .7 & 0 & .4 \end{array}\right]$
* All entries $\geq 0$ and all columns add to 1
* Property 2 guarantees 1 is an eigenvalue, all other eigenvals must be less than 1
* Taking $A - 1\lambda$ creates singular matrix, cols add to 0. 

### Fast Fourier Transform

### Differential Equations

### Matrix Exponentials

