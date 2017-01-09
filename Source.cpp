/* 
Compiles with msvc and nvcc compiler in 64-bits mode

self-contained except for standard headers available in a vs2013 (with CUDA 7.0 and up) installation &
paul.h
*/

/*
Implements SOP(D)(Compiled) framwork purely in CPP.
The function f and its derivatives as well as sigma are supplied as a template parameter.
The variable 'names' X can be any type, making this nearly as convenient to use as the Mathematica interface,
c.f. SOPD-family of functions.
*/

/*
Programming conventions:

Violating constraints leads to undefined behaviour,
as often as possible this results in assertion failures, but might also give garbage out which will propagate,
or create infinite loops.

Impure functions are annotated as such (they are avoided where possible).

_Out_ parameters are used to work around the limitation of returning just one value (and the awkwardness of using pair<> to overcome this).
Since they are often passed by-reference, this might have effects on other parts of the system.
However, the intention is that these output-parameters are allocated by the caller just before calling.

Use assert with reasonable error message/details wherever possible.
The expression in assert(...) may never have side-effects. TODO maybe develop 'executeAndAssert' macro

Where conveniently possible, appropriate types are used to enforce compile-time checking of constraints.

Prefer const == x -style and f() == x tests over x == const to enforce compile-time error on the typing error 'x = cons' and 'x = f()'. Prefer 'const == f()' over 'f() == const' (f could be a reference!)

(Array) sizes and (lists of) indices as well as loop counters are unsigned int (instead of size_t) (to save storage).
    Explicit casting is needed when transitioning from the .size()-world of std::vector.
    Use restrictSize() to do the conversion including basic check

Compile self defined code using highest warning level, use sal annotations, use run code analysis on solution, justify
    things it doesn't recognize.
    Goal is no warnings, but don't sacrifice productivity.

Plan architecture and interfaces first.

Test.
*/


// paul.h
#include <paul.h>










#ifndef _WIN64
#error Must be compiled in 64 bits. struct cs is assumed to be a multiple of 8 bytes in size (important for CUDA memory accesses).
#endif




























































// windows.h also has an implementation of this
// #define DBG_UNREFERENCED_LOCAL_VARIABLE(x) ((void)(x));













































































unsigned int restrictSize(const size_t s) {
    assert(s <= 0xffffffffu);
    return (unsigned int)s;
}












template<typename T>
FUNCTION(T, MIN, (T x, T y), "min(x,y)") {
    return x < y ? x : y;
}












_Must_inspect_result_ bool approximatelyEqual(float x, float y) {
    return abs(x - y) < 1e-5f;
}






















// Memory management

// memoryAllocate/Free manage 'universal'/'portable' dynamic memory, allocatable on the CPU and available to the GPU too
#ifdef __CUDACC__

#define CUDA_CHECK_ERRORS() {auto e = cudaGetLastError(); printf("cudaGetLastError %d %s %s\n", e, cudaGetErrorName(e), cudaGetErrorString(e));}
#ifdef __CUDACC__
#include <time.h>
#define CUDAKERNEL_LAUNCH(name, griddim, blockdim, ...) {auto t0 = clock(); name<<<griddim, blockdim>>>(__VA_ARGS__); cudaDeviceSynchronize();printf("%s finished in %f s\n",#name,(double)(clock()-t0)/CLOCKS_PER_SEC);CUDA_CHECK_ERRORS();}
#else
#define CUDAKERNEL_LAUNCH(name, griddim, blockdim, ...) 
#endif

#define memoryAllocate(ptr, sizeInBytes) {cudaDeviceSynchronize();cudaMallocManaged(&ptr, (sizeInBytes));cudaDeviceSynchronize();assert(ptr);CUDA_CHECK_ERRORS();}
#define memoryFree(ptr) {cudaDeviceSynchronize();cudaFree(ptr);cudaDeviceSynchronize();CUDA_CHECK_ERRORS();}

#else
#define memoryAllocate(ptrtype, ptr, sizeInBytes) {ptr = (ptrtype)/*decltype not working*/malloc(sizeInBytes); assert(ptr); }
#define memoryFree(ptr) {::free(ptr);}
#endif

// Universal thread identifier
#ifdef __CUDACC__
__host__ __device__
#endif
inline int linear_global_threadId() {
#ifdef __CUDA_ARCH__
    return blockDim.x * blockIdx.x + threadIdx.x;
#else
    return 0; // TODO support CPU multithreading (multi-blocking - one thread is really one block, there is no cross cooperation ideally to ensure lock-freeness and best performance)
#endif
}

#ifdef __CUDACC__
#define INLINEFUNCTION inline __host__
#else
#define INLINEFUNCTION inline
#endif

// Convenience wrappers for memoryAllocate

template<typename T>
INLINEFUNCTION T* tmalloc(const size_t n) {
    assert(n);
    T* out;
    memoryAllocate(T*, out, sizeof(T) * n);
    return out;
}

template<typename T>
INLINEFUNCTION T* tmalloczeroed(const size_t n) {
    assert(n);
    T* out;
    memoryAllocate(T*, out, sizeof(T) * n);
    memset(out, 0, sizeof(T) * n);
    return out;
}

template<typename T>
INLINEFUNCTION T* mallocmemcpy(T const * const x, const size_t n) {
    assert(n);
    auto out = tmalloc<T>(n);
    memcpy(out, x, sizeof(T) * n);
    return out;
}

template<typename T>
void freemalloctmemcpy(T** dest, const T* const src, int n)  {
    assert(n);
    if (*dest) memoryFree(*dest);

    auto sz = sizeof(T) * n;

    memoryAllocate(T*, *dest, sz);
    memcpy(*dest, src, sz);
}






























// --- SOPCompiled framework ---


/*
This program solves least-squares problems with energies of the form

\sum_{P \in Q} \sum_{p \in P} ||f(select_p(x))||_2^2

Q gives a partitioning of the domain. In the simplest case, there is only one partition.

The solution to this may or may not be close to the solution to

\sum_{p \in \Cup Q} ||f(select_p(x))||_2^2

*/




// todo put elsewhere, use CONSTANT() to be able to read them from outside -- this is currently not possible because interplay with the // preprocessor is not implemented in the WSTP wrapper code
/**/



/*
CSPARSE
A Concise Sparse Matrix Package in C

http://people.sc.fsu.edu/~jburkardt/c_src/csparse/csparse.html

CSparse Version 1.2.0 Copyright (c) Timothy A. Davis, 2006

reduced to only the things needed for sparse conjugate gradient method

by Paul Frischknecht, August 2016

and for running on CUDA, with a user-supplied memory-pool

modified & used without permission
*/


/* --- primary CSparse routines and data structures ------------------------- */
struct cs    /* matrix in compressed-column or triplet form . must be aligned on 8 bytes */
{
    unsigned int nzmax;	/* maximum number of entries allocated for triplet. Actual number of entries for compressed col. > 0 */
    unsigned int m;	    /* number of rows > 0 */

    unsigned int n;	    /* number of columns  > 0 */
    int nz;	    /* # of used entries (of x) in triplet matrix, NZ_COMPRESSED_COLUMN_INDICATOR for compressed-col, >= 0 otherwise --> must be signed*/

    // Note: this order preserves 8-byte pointer (64 bit) alignment, DO NOT CHANGE
    // all pointers are always valid
    unsigned int *p;	    /* column pointers (size n+1) or col indices (size nzmax) */

    unsigned int *i;	    /* row indices, size nzmax */

    float *x;	/* numerical values, size nzmax. if cs_is_compressed_col, nzmax entries are used, if cs_is_triplet, nz many are used (use cs_x_used_entries()) */

};

FUNCTION(bool, cs_is_triplet, (const cs *A), "whether A is a triplet matrix") {
    assert(A);
    return A->nz >= 0;
}

const int NZ_COMPRESSED_COLUMN_INDICATOR = -1;

FUNCTION(bool, cs_is_compressed_col, (const cs *A), "whether A is a crompressed-column form matrix") {
    assert(A);
    assert(A->m >= 1 && A->n >= 1);
    return A->nz == NZ_COMPRESSED_COLUMN_INDICATOR;
}

unsigned int cs_x_used_entries(const cs *A) {
    assert(cs_is_triplet(A) != /*xor*/ cs_is_compressed_col(A));
    return cs_is_triplet(A) ? A->nz : A->nzmax;
}


// hacky arbitrary memory-management by passing
// reduces memory_size and increases memoryPool on use
#define MEMPOOL char*& memoryPool, /*using int to detect over-deallocation */ int& memory_size // the function taking these modifies memoryPool to point to the remaining free memory
#define MEMPOOLPARAM memoryPool, memory_size 


FUNCTION(char*, cs_malloc_, (MEMPOOL, unsigned int sz /* use unsigned int?*/),
    "allocate new stuff. can only allocate multiples of 8 bytes to preserve alignment of pointers in cs. Use nextEven to round up when allocating 4 byte stuff (e.g. int)"){
    // assert(sz < INT_MAX) // TODO to be 100% correct, we'd have to check that memory_size + sz doesn't overflow
    assert((size_t)memory_size >= sz);
    assert(aligned(memoryPool, 8));
    assert(divisible(sz, 8));
    auto out = memoryPool;
    memoryPool += sz;
    memory_size -= (int)sz; // TODO possible loss of data...
    return out;
}

#define cs_malloc(varname, sz) {(varname) = (decltype(varname))cs_malloc_(MEMPOOLPARAM, (sz));}

FUNCTION(void, cs_free_, (char*& memoryPool, int& memory_size, unsigned int sz), "free the last allocated thing of given size"){
    // assert(sz < INT_MAX) // TODO to be 100% correct, we'd have to check that memory_size + sz doesn't overflow
    assert(divisible(sz, 8));
    assert(aligned(memoryPool, 8));
    memoryPool -= sz;
    memory_size += (int)sz;
    assert(memory_size >= 0);
}

#define cs_free(sz) {cs_free_(MEMPOOLPARAM, (sz));}


FUNCTION(unsigned int, cs_spalloc_size, (const unsigned int m, const unsigned int n, const unsigned int nzmax, bool triplet),
    "amount of bytes a sparse matrix with the given characteristics will occupy"){
    DBG_UNREFERENCED_PARAMETER(m); // independent of these
    DBG_UNREFERENCED_PARAMETER(n);
    return sizeof(cs) + nextEven(triplet ? nzmax : n + 1) * sizeof(int) + nextEven(nzmax) *  (sizeof(int) + sizeof(float));
}


FUNCTION(cs *, cs_spalloc, (const unsigned int m, const unsigned int n, const unsigned int nzmax, bool triplet, MEMPOOL),
    "allocates a sparse matrix using memory starting at memoryPool,"
    "uses exactly"
    "sizeof(cs) + cs_spalloc_size(m, n, nzmax, triplet) BYTES"
    "of the pool"
    )
{
    assert(nzmax <= m * n); // cannot have more than a full matrix
    char* initial_memoryPool = memoryPool;
    assert(nzmax > 0);

    cs* A; cs_malloc(A, sizeof(cs));    /* allocate the cs struct */

    A->m = m;				    /* define dimensions and nzmax */
    A->n = n;
    A->nzmax = nzmax;
    A->nz = triplet ? 0 : NZ_COMPRESSED_COLUMN_INDICATOR;		    /* allocate triplet or comp.col */

    // Allocate too much to preserve alignment
    cs_malloc(A->p, nextEven(triplet ? nzmax : n + 1) * sizeof(int));
    cs_malloc(A->i, nextEven(nzmax) * sizeof(int));
    cs_malloc(A->x, nextEven(nzmax) * sizeof(float));

    assert(memoryPool == initial_memoryPool + cs_spalloc_size(m, n, nzmax, triplet));
    return A;
}


FUNCTION(unsigned int, cs_cumsum, (_Inout_updates_all_(n + 1) unsigned int *p, _Inout_updates_all_(n) unsigned int *c, const unsigned int n),
    "p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c "
    )
{
    assert(p && c); /* check inputs */
    unsigned int i, nz = 0;
    for (i = 0; i < n; i++)
    {
        p[i] = nz;
        nz += c[i];
        c[i] = p[i];
    }
    p[n] = nz;
    return (nz);		    /* return sum (c [0..n-1]) */
}

FUNCTION(unsigned int*, allocZeroedIntegers, (const int n, MEMPOOL), "Allocate n integers set to 0. Implements calloc(n, sizeof(int)). n must be even") {
    assert(divisible(n, 2));
    unsigned int* w;
    cs_malloc(w, n * sizeof(unsigned int));
    memset(w, 0, n*sizeof(unsigned int)); // w = (int*)cs_calloc(n, sizeof(int)); /* get workspace */
    return w;
}

// alloc/free a list of integers w, initialized to 0
#define allocTemporaryW(count) unsigned int wsz = nextEven((count)); unsigned int* w = allocZeroedIntegers(wsz, MEMPOOLPARAM); 
#define freeTemporaryW() cs_free(wsz * sizeof(unsigned int)); 


FUNCTION(cs *, cs_transpose, (const cs * const A, MEMPOOL),
    "C = A'"
    ""
    "memoryPool must be big enough to contain the following:"
    "cs_spalloc_size(n, m, Ap[n], 0) --location of output"
    "nextEven(m)*sizeof(int) --temporary")
{
    assert(A && cs_is_compressed_col(A));

    const unsigned int m = A->m;
    const unsigned int n = A->n;
    unsigned int const * const Ai = A->i;
    unsigned int const * const Ap = A->p;
    float const * const Ax = A->x;

    cs *C; C = cs_spalloc(n, m, Ap[n], 0, MEMPOOLPARAM); /* allocate result */

    allocTemporaryW(m); /* get workspace */

    unsigned int* const Cp = C->p; unsigned int* const Ci = C->i; float* const Cx = C->x;
    assert(Cp && Ci && Cx);

    for (unsigned int p = 0; p < Ap[n]; p++) w[Ai[p]]++;	   /* row counts */
    cs_cumsum(Cp, w, m);				   /* row pointers */
    for (unsigned int j = 0; j < n; j++)
    {
        for (unsigned int p = Ap[j]; p < Ap[j + 1]; p++)
        {
            int q;
            Ci[q = w[Ai[p]]++] = j;	/* place A(i,j) as entry C(j,i) */
            Cx[q] = Ax[p];
        }
    }

    freeTemporaryW();

    return C;	/* success; free w and return C */
}

FUNCTION(cs *, cs_triplet, (const cs * const T, MEMPOOL),
    "C = compressed-column form of a triplet matrix T"
    ""
    "memoryPool must be big enough to contain the following"
    "cs_spalloc_size(m, n, nz, 0) --location of output"
    "nextEven(n)* sizeof(int) --temporary")
{
    assert(T && cs_is_triplet(T));/* check inputs */

    const int m = T->m;
    const int n = T->n;
    unsigned int const * const Ti = T->i;
    unsigned int const * const Tj = T->p;
    float const * const Tx = T->x;
    const auto nz = T->nz;

    assert(m > 0 && n > 0);
    cs *C; C = cs_spalloc(m, n, nz, 0, memoryPool, memory_size);		/* allocate result */

    allocTemporaryW(n); /* get workspace */

    unsigned int* const Cp = C->p; unsigned int* const Ci = C->i; float* const Cx = C->x;
    assert(Cp && Ci && Cx);

    for (int k = 0; k < nz; k++) w[Tj[k]]++;		/* column counts */
    cs_cumsum(Cp, w, n);				/* column pointers */
    for (int k = 0; k < nz; k++)
    {
        int p;
        Ci[p = w[Tj[k]]++] = Ti[k];    /* A(i,j) is the pth entry in C */
        Cx[p] = Tx[k];
    }

    freeTemporaryW();

    return C;	    /* success; free w and return C */
}

FUNCTION(int, cs_entry, (cs * const T, const unsigned int i, const unsigned int j, const float x),
    "add an entry to a triplet matrix; return 1 if ok, assertion failure otherwise ")
{
    assert(cs_is_triplet(T));
    assert(i < T->m && j <= T->n); // cannot enlarge matrix
    assert(T->nz < (int)T->nzmax); // cannot enlarge matrix
    assert(T->x);
    assertFinite(x);

    T->x[T->nz] = x;
    T->i[T->nz] = i;
    T->p[T->nz++] = j;
    return (1);
}


FUNCTION(int, cs_print, (const cs * const A, int brief = 0), "print a sparse matrix")
{
    assert(A);
    unsigned int p, j, m, n, nzmax;
    unsigned int *Ap, *Ai;
    float *Ax;

    m = A->m; n = A->n; Ap = A->p; Ai = A->i; Ax = A->x;
    nzmax = A->nzmax; 

    printf("CSparse %s\n",
#ifdef __CUDA_ARCH__
        "on CUDA"
#else
        "on CPU"
#endif
        );
    assert(m > 0 && n > 0);
    if (cs_is_compressed_col(A))
    {
        printf("%d-by-%d, nzmax: %d nnz: %d\n", m, n, nzmax,
            Ap[n]);
        for (j = 0; j < n; j++)
        {
            printf("    col %d : locations %d to %d\n", j, Ap[j], Ap[j + 1] - 1);
            for (p = Ap[j]; p < Ap[j + 1]; p++)
            {
                assert(Ai[p] < m);
                printf("      %d : %g\n", Ai[p], Ax ? Ax[p] : 1);
                if (brief && p > 20) { printf("  ...\n"); return (1); }
            }
        }
    }
    else
    {
        auto nz = A->nz;
        printf("triplet: %d-by-%d, nzmax: %d nnz: %d\n", m, n, nzmax, nz);
        assert(nz <= (int)nzmax);
        for (p = 0; p < (unsigned int)nz; p++)
        {
            printf("    %d %d : %f\n", Ai[p], Ap[p], Ax ? Ax[p] : 1);
            assert(Ap[p] >= 0);
            if (brief && p > 20) { printf("  ...\n"); return (1); }
        }
    }
    return (1);
}


FUNCTION(int, cs_mv, (float * y, float alpha, const cs *  A, const float * x, float beta),
    "y = alpha A x + beta y"
    "the memory for y and x cannot overlap"
    "TODO implement a version that can transpose A implicitly")
{
    assert(A && x && y);	    /* check inputs */
    assert(cs_is_compressed_col(A));
    assertFinite(beta);
    assertFinite(alpha);

    unsigned int p, j, n; unsigned int *Ap, *Ai;
    float *Ax;
    n = A->n;
    Ap = A->p; Ai = A->i; Ax = A->x;

    // the height of A is the height of y. Premultiply y with beta, then proceed as before, including the alpha factor when needed 
    // TODO (can we do better?)
    // Common special cases
    if (beta == 0)
        memset(y, 0, sizeof(float) * A->m);
    else
        for (unsigned int i = 0; i < A->m; i++) y[i] *= beta;

    if (alpha == 1)
        for (j = 0; j < n; j++) for (p = Ap[j]; p < Ap[j + 1]; p++) y[Ai[p]] += Ax[p] * x[j];
    else if (alpha != 0) // TODO instead of deciding this at runtime, let the developer choose the right function xD
        for (j = 0; j < n; j++) for (p = Ap[j]; p < Ap[j + 1]; p++) y[Ai[p]] += alpha * Ax[p] * x[j];
    // if alpha = 0, we are done

    return (1);
}
// ---


// logging/debugging

GLOBAL(
    int,
    dprintEnabled,
    true,
    "if true, dprintf writes to stdout, otherwise dprintf does nothing"
    "It would be more efficient to compile with dprintf defined to nothing of course"
    "Default: true"
    );

#ifdef __CUDA_ARCH__
#define dprintf(formatstr, ...) {if (dprintEnabled) printf("CUDA " formatstr, __VA_ARGS__);}
#else
#define dprintf(formatstr, ...) {if (dprintEnabled) printf(formatstr, __VA_ARGS__);}
#endif

FUNCTION(void,
    print,
    (_In_z_ const char* const x),
    "prints a string to stdout"){
    printf("print: %s\n", x);
}


FUNCTION(
    void,
    printd,
    (_In_reads_(n) const int* v, size_t n),
    "dprints a vector of integers, space separated and newline terminated"
    ) {
    while (n--) dprintf("%d ", *v++); dprintf("\n");
}

FUNCTION(
    void,
    printd,
    (_In_reads_(n) const unsigned int* v, size_t n),
    "dprints a vector of integers, space separated and newline terminated"
    ) {
    while (n--) dprintf("%u ", *v++); dprintf("\n");
}

FUNCTION(
    void,
    printJ,
    (cs* J),
    "prints a sparse matrix"
    ){
    if (dprintEnabled) cs_print(J);
}


// for conjgrad/sparse leastsquares:

// infrastructure like CusparseSolver

/*
Implementation note:
size_t n is not const because the implementation modifies it

changes to n are not visible outside anyways:
-> the last const spec is always an implementation detail, not a promise to the caller, but can indicate conceptual thinking
*/
FUNCTION(
    void,
    printv,
    (_In_reads_(n) const float* v, size_t n),
    "dprints a vector of doubles, space separated and newline terminated"
    ) {
    while (n--) dprintf("%f ", *v++); dprintf("\n");
}

struct fvector {
    float* x;
    unsigned int n;

    MEMBERFUNCTION(void, print, (), "print this fvector") {
        printv(x, n);
    }
};

FUNCTION(void, assertFinite, (_In_reads_(n) const float* const x, const unsigned int n), "assert that each element in v is finite") {
    FOREACH(y, x, n)
        assertFinite(y);
}

FUNCTION(void, assertFinite, (const fvector& v), "assert that each element in v is finite") {
    assertFinite(v.x, v.n);
}

FUNCTION(fvector, vector_wrapper, (float* x, int n), "create a fvector object pointing to existing memory for convenient accessing") {
    fvector v;
    v.n = n;
    v.x = x;
    assertFinite(v);
    return v;
}

FUNCTION(fvector, vector_allocate, (int n, MEMPOOL), "Create a new fvector. uninitialized: must be written before it is read!") {
    fvector v;
    v.n = n;
    cs_malloc(v.x, sizeof(float) * nextEven(v.n));
    return v;
}

FUNCTION(fvector, vector_copy, (const fvector& other, MEMPOOL), "create a copy of other") {
    fvector v;
    v.n = other.n;
    cs_malloc(v.x, sizeof(float) * nextEven(v.n));
    memcpy(v.x, other.x, sizeof(float) * v.n);
    assertFinite(v);
    return v;
}

struct matrix {
    const cs* const mat; // in compressed column form (transpose does not work with triplets)

    __declspec(property(get = getRows)) int rows;
    MEMBERFUNCTION(int, getRows, (), "m") const {
        return mat->m;
    }
    __declspec(property(get = getCols)) int cols;
    MEMBERFUNCTION(int, getCols, (), "n") const {
        return mat->n;
    }


    MEMBERFUNCTION(, matrix, (const cs* const mat), "construct a matrix wrapper") : mat(mat) {
        assert(!cs_is_triplet(mat));
        assert(mat->m && mat->n);
        assertFinite(mat->x, cs_x_used_entries(mat));
    }

    MEMBERFUNCTION(void, print, (), "print this matrix"){
        cs_print(mat, 0);
    }

    void operator=(matrix); // undefined
};


FUNCTION(float, dot, (const fvector& x, const fvector& y), "result = <x, y>, aka x.y or x^T y (the dot-product of x and y)"){
    assert(y.n == x.n);
    float r = 0;
    DO(i, x.n) r += x.x[i] * y.x[i];
    return r;
}

FUNCTION(void, axpy, (fvector& y, const float alpha, const fvector& x), "y = alpha * x + y") {
    assert(y.n == x.n);
    DO(i, x.n) y.x[i] += alpha * x.x[i];
}

FUNCTION(void, axpy, (fvector& y, const fvector& x), "y = x + y") {
    axpy(y, 1, x);
}

FUNCTION(void, scal, (fvector& x, const float alpha), "x *= alpha"){
    DO(i, x.n) x.x[i] *= alpha;
}

FUNCTION(void, mv, (fvector& y, const float alpha, const matrix& A, const fvector& x, const float beta),
    "y = alpha A x + beta y"){
    assert(A.mat->m && A.mat->n);
    assert(y.n == A.mat->m);
    assert(x.n == A.mat->n);
    cs_mv(y.x, alpha, A.mat, x.x, beta);
}

FUNCTION(void, mv, (fvector& y, const matrix& A, const fvector& x), "y = A x"){
    mv(y, 1, A, x, 0);
}

FUNCTION(matrix, transpose, (const matrix& A, MEMPOOL), "A^T") {
    return matrix(cs_transpose(A.mat, MEMPOOLPARAM));
}

#define memoryPush() const auto old_memoryPool = memoryPool; const auto old_memory_size = memory_size; //savepoint: anything allocated after this can be freed again
#define memoryPop() {memoryPool = old_memoryPool; memory_size = old_memory_size;} // free anything allocated since memory push

// core algorithm, adapted from CusparseSolver, originally copied from wikipedia
/* required operations:
- new fvector of given size
- copy/assign fvector
- mv_T, mv (matrix (transpose) times fvector) -- because I did not come up with a transposing-multiply operation, I just compute AT once instead of using mv_T
- scal (scaling)
- axpy // y = alpha * x + y
*/
//function [x] = conjgrad_normal(A,b,x)
/*The conjugate gradient method can be applied to an arbitrary n-by-m matrix by applying it to normal equations ATA and right-hand side fvector ATb, since ATA is a symmetric positive-semidefinite matrix for any A. The result is conjugate gradient on the normal equations (CGNR).
ATAx = ATb
As an iterative method, it is not necessary to form ATA explicitly in memory but only to perform the matrix-fvector and transpose matrix-fvector multiplications.

x is an n-fvector in this case still

x is used as the initial guess -- it may be 0 but must in any case contain valid numbers
*/
FUNCTION(void, conjgrad_normal, (
    const matrix& A,
    const fvector& b,
    fvector& x,
    MEMPOOL),
    "x = A\b"
    )
{
    memoryPush(); //savepoint: anything allocated after this can be freed again

    int m = A.rows, n = A.cols;

    matrix AT = transpose(A, MEMPOOLPARAM); // TODO implement an mv that does transposing in-place

    fvector t = vector_allocate(m, MEMPOOLPARAM);

    fvector r = vector_allocate(n, MEMPOOLPARAM); mv(r, AT, b); mv(t, A, x); mv(r, -1, AT, t, 1);//r=A^T*b; t = A*x; r = -A^T*t + r;//r=A^T*b-A^T*A*x;

    fvector p = vector_copy(r, MEMPOOLPARAM);//p=r;

    float rsold = dot(r, r);//rsold=r'*r;

    fvector Ap;
    if (sqrt(rsold) < 1e-5) goto end; // low residual: solution found

    Ap = vector_allocate(A.cols, MEMPOOLPARAM);

    for (unsigned int i = 1; i <= b.n; i++) {
        mv(t, A, p); mv(Ap, AT, t);//t = A*p;Ap=A^T*t;//Ap=A^T*A*p;

        if (abs(dot(p, Ap)) < 1e-9) { printf("conjgrad_normal emergency exit\n"); break; }// avoid almost division by 0
        float alpha = rsold / (dot(p, Ap));//alpha=rsold/(p'*Ap);

        axpy(x, alpha, p);//x = alpha p + x;//x=x+alpha*p;
        axpy(r, -alpha, Ap);//r = -alpha*Ap + r;//r=r-alpha*Ap;
        float rsnew = dot(r, r);//rsnew=r'*r;
        if (sqrt(rsnew) < 1e-5) break; // error tolerance, might also limit amount of iterations or check change in rsnew to rsold...
        float beta = (rsnew / rsold);
        scal(p, beta); axpy(p, r);//p*=(rsnew/rsold); p = r + p;//p=r+(rsnew/rsold)*p;
        rsold = rsnew;//rsold=rsnew;
    }

end:
    memoryPop(); // free anything allocated since memory push
}

// solving least-squares problems
FUNCTION(int, cs_cg, (const cs * const A, _In_reads_(A->m) const float * const b, _Inout_updates_all_(A->n) float *x, MEMPOOL),
    "x=A\b"
    "current value of x is used as initial guess"
    "Uses memory pool to allocate transposed copy of A and four vectors with size m or n")
{
    assert(A && b && x && memoryPool && memory_size > 0);

    auto xv = vector_wrapper(x, A->n);
    conjgrad_normal(matrix(A), vector_wrapper((float*)b, A->m), xv, MEMPOOLPARAM);

    return 1;
}

/*

CSPARSE library end

*/



// pure functions, utilities

FUNCTION(
    void,
    assertEachInRange,
    (
    _In_reads_(len) const int* v,
    size_t len,
    const int min,
    const int max
    ),
    "computes the same as BoolEval[min <= v <= max]"
    ) {
    assert(v);
    while (len--) { // Note: len reduced once more, gets gigantic if len was already 0
        assert(min <= *v && *v <= max);
        v++;
    }
}


FUNCTION(
    void,
    assertEachInRange,
    (
    _In_reads_(len) const unsigned int* v,
    size_t len,
    const unsigned int min,
    const unsigned int max
    ),
    "computes the same as Assert@And@@BoolEval[min <= v <= max]"
    ) {
    assert(v);
    while (len--) { // Note: len reduced once more, gets gigantic if len was already 0
        assert(min <= *v && *v <= max);
        v++;
    }
}

FUNCTION(
    void,
    assertLessEqual,
    (
    _In_reads_(len) const unsigned int* v,
    _In_ size_t len, // not const for implementation
    const unsigned int max
    ),
    "BoolEval[v <= max]"
    ) {
    assert(v);
    while (len--) { // Note: len reduced once more, gets gigantic if len was already 0
        assert(*v <= max);
        v++;
    }
}

FUNCTION(
    void,
    assertLess,
    (
    _In_reads_(len) const unsigned int* v,
    size_t len,
    const unsigned int max
    ),
    "BoolEval[v <= max]"
    ) {
    assert(v);
    while (len--) { // Note: len reduced once more, gets gigantic if len was already 0
        assert(*v < max, "%d %d", *v, max);
        v++;
    }
}

CPU_FUNCTION(
    void,
    assertLess,
    (
    _In_ const vector<unsigned int>& v,
    const unsigned int max
    ),
    "BoolEval[v <= max]"
    ) {
    assertLess(v.data(), v.size(), max);
}

FUNCTION(
    void,

    axpyWithReindexing,

    (
    _Inout_updates_(targetLength) float* const targetBase,
    const unsigned int targetLength,
    float const a,

    _In_reads_(targetIndicesAndAddedValuesLength) const float* const addedValues,
    _In_reads_(targetIndicesAndAddedValuesLength) const unsigned int* const targetIndices,
    const unsigned int targetIndicesAndAddedValuesLength
    ),
    "targetBase[[targetIndices]] += a * addedValues. Repeated indices are not supported, so addedValues cannot be longer than the target."
    "Note that not necessarily all of target is updated (_Inout_updates_, not _Inout_updates_all_)"

    ) {
    assert(targetLength); // targetLength - 1 overflows otherwise
    assertFinite(a);
    assert(targetIndicesAndAddedValuesLength <= targetLength);
    dprintf("axpyWithReindexing %f %d %d\n", a, targetLength, targetIndicesAndAddedValuesLength);

    dprintf("target before:\n"); printv(targetBase, targetLength);
    dprintf("targetIndices:\n"); printd(targetIndices, targetIndicesAndAddedValuesLength);
    dprintf("addedValues:\n"); printv(addedValues, targetIndicesAndAddedValuesLength);

    assertEachInRange(targetIndices, targetIndicesAndAddedValuesLength, 0, targetLength - 1);

    DO(j, targetIndicesAndAddedValuesLength)
        assertFinite(targetBase[targetIndices[j]] += addedValues[j] * a);

    dprintf("target after:\n"); printv(targetBase, targetLength);
}

FUNCTION(void, extract, (
    _Out_writes_all_(sourceIndicesAndTargetLength) float* const target,

    _In_reads_(sourceLength) const float* const source,
    const unsigned int sourceLength,

    _In_reads_(sourceIndicesAndTargetLength) const unsigned int* const sourceIndices,
    const unsigned int sourceIndicesAndTargetLength
    ),
    "target = source[[sourceIndices]]. Note that all of target is initialized (_Out_writes_all_)"
    ) {
    assertEachInRange(sourceIndices, sourceIndicesAndTargetLength, 0, sourceLength - 1);

    DO(i, sourceIndicesAndTargetLength)
        target[i] = source[sourceIndices[i]];
}


















// SparseOptimizationProblem library

// --- Memory pool passed to the csparse library ---
// used in buildFxAndJFxAndSolve

// this is ideally some __shared__ memory in CUDA: In CUDA (I think) 
// C-style "stack" memory is first register based but then spills to main memory
// (is shared memory also used for the registers? Just another way to access the register file?)
// this memory does not need to be manually freed

// DEBUG TODO moved memory to global space for debugging -- move to __shared__ again.
// down the stack, no two functions should be calling SOMEMEM at the same time!

//__managed__ char memory[40000/*"Maximum Shared Memory Per Block" -> 49152*/ * 1000]; // TODO could allocate 8 byte sized type, should be aligned then (?)
//__managed__ bool claimedMemory = false; // makes sure that SOMEMEM is only called by one function on the stack

// "A default heap of eight megabytes is allocated if any program uses malloc() without explicitly specifying the heap size." -- want more 

#ifdef __CUDACC__
// TODO run this to increase the tiny heap size
void preWsMain() { // using a constructor to do this seems not to work
    int const mb = 400;
    printf("setting cuda malloc heap size to %d mb\n", mb);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, mb * 1000 * 1000); // basically the only memory we will use, so have some! // TODO changes when this is run from InfiniTAM
    CUDA_CHECK_ERRORS();
}
// TODO easily exceeded with lots of partitions on big scenes - small partitions don't need that much memory
#endif

#define SOMEMEM() \
    const size_t memory_size = 8 * 1000  * 1000;\
    char* const memory = (char*)malloc(memory_size);/*use global memory afterall*/\
    char* mem = (char*)(((unsigned long long)memory+7) & (~ 0x7ull)); /* align on 8 byte boundary */\
    assert(aligned(mem, 8) && after(mem, memory));\
    int memsz = memory_size - 8;/*be safe*/ \
    assert(memsz>0);\
    bool claimedMemory = true;\
    printf("allocated %d bytes at %#p using malloc\n", memory_size, memory);\
    assert(memory); /*attempting to access a null pointer just gives a kernel launch failure on GPU most of the time - at least when debugger cannot be attached */

#define FREESOMEMEM() {assert(claimedMemory); claimedMemory = false; ::free(memory); mem = 0;}


#define SOMEMEMP mem, memsz

// --- end of memory pool stuff ---

template<typename f> class SOPDProblem;

// f as in SOPDProblem
// one separate SOP (for one P in Q), shares only "x" with the global problem
// has custom y, p and values derived from that
// pointers are all to __managed__ memory, instances of this shall live in managed memory
// F() is another function for each partition P. It is defined as (f(s_p(x)))_{p in P}
template<typename f>
struct SOPPartition {
    typedef f fType;

    float* minusFx; unsigned int lengthFx; // "-F(x)"
    float* h; unsigned int lengthY; // "h, the update to y (subset of x, the parameters currently optimized over)"

    /*
    "amount of 'points' at which the function f is evaluated."
    "lengthP * lengthz is the length of xIndices, "
    "and sparseDerivativeZtoYIndices contains lengthP sequences of the form (k [k many z indices] [k many y indices]) "
    */
    unsigned int lengthP;

    // integer matrix of dimensions lengthz x lengthP, indexing into x to find the values to pass to f
    unsigned int* xIndices;

    // Used to construct J, c.f. SOPJF
    unsigned int* sparseDerivativeZtoYIndices; // serialized form of this ragged array

    /*
    "the indices into x that indicate where the y are"
    "needed to write out the final update h to the parameters"
    */
    unsigned int* yIndices; /* lengthY */

    // parent (stores shared x and lengthx)
    SOPDProblem<f> const * /*const*/ sopd;
};

// TODO give sop as parameter to these to make it look like functions of sop
#define _lengthz(sop) std::remove_reference<decltype(*(sop))>::type::fType::lengthz
#define _lengthfz(sop) std::remove_reference<decltype(*(sop))>::type::fType::lengthfz
#define _lengthx(sop) ((sop)->sopd->lengthx)
#define _x(sop) ((sop)->sopd->x)


template<typename f>
FUNCTION(void, buildFxandJFx, (SOPPartition<f>* const sop, cs* const J, bool buildFx), "");
template<typename f>
FUNCTION(void, solve, (SOPPartition<f>* const sop, cs const * const J, MEMPOOL), "");

/*
The type f provides the following static members:
* const unsigned int lengthz, > 0
* const unsigned int lengthfz, > 0
* void f(_In_reads_(lengthz) const float* const input,  _Out_writes_all_(lengthfz) float* const out)
* void df(_In_range_(0, lengthz-1) int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out)
df(i) must be the derivative of f by the i-th argument

Note: Functions f and df should be CPU and GPU compilable.

Instances of this must live in GPU memory, because their pointer to x is dereferenced
*/
template<typename f>
class SOPDProblem {
private:
    void operator=(SOPDProblem<f>); // undefined

public:
    typedef f fType;

    SOPDProblem(
        _In_ const vector<float>& x,
        _In_ const vector<vector<unsigned int>>& xIndicesPerPartition,
        _In_ const vector<vector<unsigned int>>& yIndicesPerPartition,
        _In_ const vector<vector<unsigned int>>& sparseDerivativeZtoYIndicesPerPartition) : partitions(restrictSize(xIndicesPerPartition.size())), lengthx(restrictSize(x.size())), partitionTable(0) {
        assert(partitions >= 0);
        allocatePartitions();
        assert(partitionTable);

        // TODO repeat and or externalize parameter checks (occur at quite a few places now but being safe is free)

        receiveSharedOptimizationData(x.data());
        DO(partition, partitions) {

#define r restrictSize
#if _DEBUG
            receiveAndPrintOptimizationData(f::lengthz, f::lengthfz,
                x.data(), r(x.size()),
                sparseDerivativeZtoYIndicesPerPartition[partition].data(), r(sparseDerivativeZtoYIndicesPerPartition[partition].size()),
                xIndicesPerPartition[partition].data(), r(xIndicesPerPartition[partition].size()),
                yIndicesPerPartition[partition].data(), r(yIndicesPerPartition[partition].size())
                );
#endif
            receiveOptimizationData(partition,
                sparseDerivativeZtoYIndicesPerPartition[partition].data(), r(sparseDerivativeZtoYIndicesPerPartition[partition].size()),
                xIndicesPerPartition[partition].data(), r(xIndicesPerPartition[partition].size()),
                yIndicesPerPartition[partition].data(), r(yIndicesPerPartition[partition].size())
                );
        }
    }

    void solve(_In_ const unsigned int iterations = 1) {
        DO(i, partitions) buildFxAndJFxAndSolveRepeatedly(i, iterations); // TODO parallelize partitions
    }

    vector<float> getX() {
        auto xv = vector<float>(x, x+lengthx);
        return xv;
    }

    ~SOPDProblem() {
        // free old stuff 
        FOREACH(sop, partitionTable, partitions) {
            memoryFree(sop.sparseDerivativeZtoYIndices);
            memoryFree(sop.xIndices);
            memoryFree(sop.yIndices);
            memoryFree(sop.minusFx);
            memoryFree(sop.h);
        }

        memoryFree(partitionTable);
        memoryFree(x);
    }

    // accessed externally:

    // "stores the current data vector 'x' which is updated to reduce the energy ||F(x)||^2", of length lengthx
    float* /*const*/ x; // to __managed__ memory
    const unsigned int lengthx; // > 0 

private:
    SOPPartition<f>* /*const*/ partitionTable; // to __managed__ memory
    const unsigned int partitions; // > 0



    //  "set the amount of partitions"
    void allocatePartitions() {
        assert(partitions >= 0);
        // allocate
        partitionTable = tmalloczeroed<SOPPartition<f>>(partitions); // pointers not yet initialized
    }

    // "Receives x"
    void
        receiveSharedOptimizationData
        (
        _In_reads_(lengthx) const float* const xI
        ) {
        x = mallocmemcpy(xI, lengthx);
    }



    // macro for indexing into partitionTable, sop = partitionTable[partition]
#define extractSop(partition) assert(partition >= 0 && partition < partitions); SOPPartition<f>* const sop = &partitionTable[partition];

    CPU_FUNCTION(
        void,
        receiveOptimizationData,
        (
        const unsigned int partition,
        _In_reads_(sparseDerivativeZtoYIndicesLength) const unsigned int* const sparseDerivativeZtoYIndicesI, const unsigned int sparseDerivativeZtoYIndicesLength,
        _In_reads_(xIndicesLength) const unsigned int* const xIndicesI, const unsigned int xIndicesLength,
        _In_reads_(yIndicesLength) const unsigned int* const yIndicesI, const unsigned int yIndicesLength
        ),
        "Receives sparseDerivativeZtoYIndices, xIndices and yIndices"
        "Appropriately sized vectors for receiving these data items are newly allocated in __managed__ memory, hence this is a CPU only function"
        ) {
        extractSop(partition);

        sop->sparseDerivativeZtoYIndices = mallocmemcpy(sparseDerivativeZtoYIndicesI, sparseDerivativeZtoYIndicesLength);
        sop->xIndices = mallocmemcpy(xIndicesI, xIndicesLength);
        sop->yIndices = mallocmemcpy(yIndicesI, yIndicesLength);

        static_assert(f::lengthz > 0, "");
        assert(divisible(xIndicesLength, f::lengthz));
        static_assert(f::lengthfz > 0, "");

        sop->lengthP = xIndicesLength / f::lengthz;
        sop->lengthY = yIndicesLength;
        sop->lengthFx = f::lengthfz * sop->lengthP;

        sop->minusFx = tmalloc<float>(sop->lengthFx);

        sop->h = tmalloc<float>(sop->lengthY);

        sop->sopd = this;
    }

    // trivially parallel over the partitions

    // note that these have to be members for accessing partitions

    // this could be a non-member if I find out how to access lengthz
    FUNCTION(
        void,
        buildFxAndJFxAndSolve,
        (SOPPartition<f> * const sop, bool buildFx),
        "using current data, builds JFx (and Fx) and solves the least squares problem"
        "optionally does not compute Fx, assuming it is current with the x data (true after every solve)"
        ""
        "Note that we must do the solving right here, because this function handles the memory needed by J"
        "the solution is then accessible in h for further processing (updating x at yIndices)"
        ""
        "sop is passed here, not partition. Use buildFxAndJFxAndSolveRepeatedly as the external interface"
        )
    {
        // Build F and JF

        const unsigned int maxNNZ = MIN((
            std::remove_reference<decltype(*sop)>::type::fType::lengthfz // TODO does this work in CUDA? if not, just make lengthfz a data member of sop or sop->sopd
            *
            std::remove_reference<decltype(*sop)>::type::fType::lengthz
            ) * sop->lengthP // very pessimistic estimate/overestimation: assume every derivative figures for every P -- usually not all of them will be needed

            // LIMIT by matrix size (usually much bigger except in very small cases)
            ,
                sop->lengthFx * sop->lengthY
                );

        // ^^ e.g. in vsfs the 3 color channels are all not optimized over, neither doriginal

        // consider using dynamic allocation in SOMEMEM!

        SOMEMEM();
        dprintf("allocating sparse matrix for %d entries\n", maxNNZ);
        cs* J = cs_spalloc(sop->lengthFx, sop->lengthY, maxNNZ, 1, SOMEMEMP); // might run out of memory here

        dprintf("buildFxandJFx\n");
        buildFxandJFx(sop, J, buildFx);

        dprintf("used %d of %d allocated spaces in J\n", J->nz, J->nzmax);
        assert(J->nz > 0); // there must be at least one (nonzero) entry in the jacobian, otherwise we have taken the derivative only over variables no ocurring (or no variables at all!)

        J = cs_triplet(J, SOMEMEMP); // "optimizes storage of J, after which it may no longer be modified" 
        // TODO recycle memory

        // State
        dprintf("-F(x):\n");
        printv(sop->minusFx, sop->lengthFx);
        dprintf("JF(x):\n");
        printJ(J);

        // Solve
        dprintf("solve:\n");
        ::solve(sop, J, SOMEMEMP); // TODO allocates even more memory

        FREESOMEMEM();
    }
    MEMBERFUNCTION(
        void,
        buildFxAndJFxAndSolveRepeatedly,
        (const unsigned int partition, const unsigned int iterations),
        "using current data, builds JFx (and Fx) and solves the least squares problem"
        "then does a gradient descent step"
        "reapeats this whole process as often as desired"
        )
    {
        extractSop(partition);

        // TODO we might want to do this externally
        printf("\n=== buildFxAndJFxAndSolveRepeatedly %d times in partition %d of %d ===\n", iterations, partition, partitions);
        assert(iterations > 0); // TODO iterations should be size_t

        DO(i, iterations) {
            bool buildFx = i == 0; // Fx is always up-to date after first iteration

            buildFxAndJFxAndSolve(sop, buildFx);
            const float delta = addContinuouslySmallerMultiplesOfHtoXUntilNorm2FxIsSmallerThanBefore(sop);
            if (delta > -0.001) {
                dprintf("delta was only %f, stopping optimization\n", delta);
                return;
            }
        }
    }

    /*
    FUNCTION(
    void,
    buildFxAndJFxAndSolveRepeatedlyThreadIdPartition,
    (const int iterations),
    "buildFxAndJFxAndSolveRepeatedly on the partition given by linear_global_threadId."
    "does nothing when linear_global_threadId is >= partitions"
    ""
    "TODO this should be the block id, threads in the same block should cooperate in the same partition"
    )
    {
    if (linear_global_threadId() >= partitions) {
    dprintf("\n--- thread id %d has nothing to do  - there are only %d partitions\n", linear_global_threadId(), partitions);
    return;
    }

    printf("\n=== Starting work on partition %d in the thread of the same id ===\n", linear_global_threadId());
    buildFxAndJFxAndSolveRepeatedly(linear_global_threadId(), iterations);
    }
    */
};

FUNCTION(void, writeJFx, (cs* const J, const unsigned int i, const unsigned int j, const float x),
    "set J(i, j) = x"
    ) {
    assert(J);
    assert(cs_is_triplet(J));
    assert(i < J->m && j < J->n);
    assert(J->nz + 1 <= (int)J->nzmax); // matrix should not become overful
    assertFinite(x);

    cs_entry(J, i, j, x);
}

template<typename f>
FUNCTION(void, writeFx, (SOPPartition<f>* const sop, const unsigned int i, const float val), "F(x)_i = val") {
    assert(i < sop->lengthFx);
    assert(sop->minusFx);
    assertFinite(val);

    sop->minusFx[i] = -val;
}

// -----------------------
/*
Given access to :

int lengthP
int lengthY
const int lengthz 
const int lengthfz
f(fz_out, z)
df(i, fz_out, z)

float* x <-- global

int* xIndices (a list of indices into x, lengthfz * n many)
int* sparseDerivativeZtoYIndices (a list of n lists of integers of the structure {k   (deriveby - k integers from 0 to argcount(f)-1) (store at - k integers from 0 to y_length-1)

This creates the fvector
Fx
and the sparse matrix
JFx

By calling

void writeFx(int i, float val)
void writeJFx(int i, int j, float val)

using only elementary C constructs
*/
// TODO move these functions to SOPPartition instead of passing the pointer all the time
template<typename f>
FUNCTION(void, readZ, (
    SOPPartition<f> const * const sop,

    _Out_writes_all_(lengthz) float* z,
    const size_t rowz
    ), "z = x[[xIndices[[rowz;;rowz+lengthz-1]]]]"){
    assert(divisible(rowz, _lengthz(sop)));

    extract(z, sop->sopd->x, sop->sopd->lengthx, sop->xIndices + rowz, _lengthz(sop)); // z = x[[xIndices]] // only place where x & lengthz is accessed
}


template<typename f>
FUNCTION(void, readZandSetFxRow, (
    SOPPartition<f>* const sop,
    _Out_writes_all_(lengthz) float* z,
    const unsigned int rowz,
    const unsigned int rowfz
    ), "compute and store Fx[[rowfz;;rowfz+lengthfz-1]] = f(z) and return the z = x[[xIndices[[rowz;;rowz+lengthz-1]]]] required for that"){
    assert(divisible(rowz, _lengthz(sop)));
    assert(divisible(rowfz, _lengthfz(sop)));

    readZ(sop, z, rowz); // z = x[[xIndices]]

    float fz[_lengthfz(sop)];
    f::f(z, fz); // fz = f(z) // the only place f is called

    DO(i, _lengthfz(sop)) writeFx(sop, rowfz + i, fz[i]); // Fx[[rowfz;;rowfz+lengthfz-1]] = fz
}

template<typename f>
FUNCTION(void, setFxRow, (
    SOPPartition<f>* const sop,
    const unsigned int rowz,
    const unsigned int rowfz
    ), "compute and store Fx[[rowfz;;rowfz+lengthfz-1]]"){
    float z[_lengthz(sop)];
    readZandSetFxRow(sop, z, rowz, rowfz);
}

template<typename f>
FUNCTION(void, buildFx, (SOPPartition<f>* const sop), "from the current x, computes just F(x)"){
    unsigned int rowz = 0;
    unsigned int rowfz = 0;

    FOR(unsigned int, i, 0, sop->lengthP, (rowz += _lengthz(sop), rowfz += _lengthfz(sop), 1)) MAKE_CONST(rowz) MAKE_CONST(rowfz) {
        DBG_UNREFERENCED_LOCAL_VARIABLE(i);
        setFxRow(sop, rowz, rowfz);
    }
}

template<typename f>
FUNCTION(void, buildFxandJFx, (SOPPartition<f>* const sop, cs* const J, bool buildFx),
    "from the current x, computes F(x) [if buildFx == true] and JF(x)"
    "Note that J is stored into the matrix pointed to"
    "this J must by in triplet form and have allocated enough space to fill in the computed df"
    ) {
    assert(cs_is_triplet(J));
    auto* currentSparseDerivativeZtoYIndices = sop->sparseDerivativeZtoYIndices;
    unsigned int rowz = 0;
    unsigned int rowfz = 0;

    FOR(unsigned int, i, 0, sop->lengthP, (rowz += _lengthz(sop), rowfz += _lengthfz(sop), 1)) MAKE_CONST(rowz) MAKE_CONST(rowfz) {
        DBG_UNREFERENCED_LOCAL_VARIABLE(i);
        float z[_lengthz(sop)];
        if (buildFx)
            readZandSetFxRow(sop, z, rowz, rowfz);
        else
            readZ(sop, z, rowz);

        // deserialize sparseDerivativeZtoYIndices, c.f. flattenSparseDerivativeZtoYIndices
        // convert back to two lists of integers of the same length (K)
        const unsigned int K = *currentSparseDerivativeZtoYIndices++;
        assert(K <= _lengthz(sop));
        const unsigned int* const zIndices = currentSparseDerivativeZtoYIndices; currentSparseDerivativeZtoYIndices += K;
        const unsigned int* const yIndices = currentSparseDerivativeZtoYIndices; currentSparseDerivativeZtoYIndices += K;

        // construct & insert localJ columnwise
        DO(k, K) {
            const unsigned int zIndex = zIndices[k];
            const unsigned int yIndex = yIndices[k];

            assert(zIndex < _lengthz(sop));
            assert(yIndex < sop->lengthY);

            float localJColumn[_lengthfz(sop)];
            f::df(zIndex, z, localJColumn);// the only place df is called

            // put in the right place (starting at rowfz, column yIndex)
            DO(j, _lengthfz(sop)) {
                writeJFx(J, rowfz + j, yIndex, localJColumn[j]);
            }
        }
    }
}
// -----------------------

// Core algorithms

template<typename f>
FUNCTION(void,
    solve,
    (SOPPartition<f>* const sop, cs const * const J, MEMPOOL),
    "assumes x, -Fx and J have been built"
    "computes the adjustment fvector h, which is the least-squares solution to the system"
    "Jh = -Fx"
    ) {
    assert(J && sop && _lengthx(sop) && _x(sop) && sop->minusFx && sop->h);
    assert(cs_is_compressed_col(J));

    printf("sparse leastSquares (cg) %d x %d... (this might take a while)\n",
        J->m, J->n);

    assert(sop->lengthY > 0);

    // h must be initialized -- initial guess -- use 0
    memset(sop->h, 0, sizeof(float) * sop->lengthY); // not lengthFx! -- in page writing error -- use struct fvector to keep fvector always with its length (existing solutions?)

    cs_cg(J, sop->minusFx, sop->h, MEMPOOLPARAM);

    dprintf("h:\n"); printv(sop->h, sop->lengthY);
    assertFinite(sop->h, sop->lengthY);
}

template<typename f>
FUNCTION(
    float,
    norm2Fx,
    (SOPPartition<f> const * const sop), "Assuming F(x) is computed, returns ||F(x)||_2^2"
    ) {
    assert(sop->minusFx);
    float x = 0;
    DO(i, sop->lengthFx) x += sop->minusFx[i] * sop->minusFx[i];
    return assertFinite(x);
}

template<typename f>
FUNCTION(
    float,
    addContinuouslySmallerMultiplesOfHtoXUntilNorm2FxIsSmallerThanBefore,
    (SOPPartition<f> * const sop),
    "scales h such that F(x0 + h) < F(x) in the 2-norm and updates x = x0 + h"
    "returns total energy delta achieved which should be negative but might not be when the iteration count is exceeded"
    ) {
    assert(sop);
    assert(sop->yIndices);
    assert(sop->minusFx);
    assert(_x(sop));

    // determine old norm
    const float norm2Fatx0 = norm2Fx(sop);
    dprintf("||F[x0]||_2^2 = %f\n", norm2Fatx0);

    // add full h
    float lambda = 1.;
    dprintf("x = "); printv(_x(sop), _lengthx(sop));
    axpyWithReindexing(_x(sop), _lengthx(sop), lambda, sop->h, sop->yIndices, sop->lengthY); // xv = x0 + h
    dprintf("x = x0 + h = "); printv(_x(sop), _lengthx(sop));

    buildFx(sop);
    float norm2Faty0 = norm2Fx(sop);
    dprintf("||F[x0 + h]||_2^2 = %f\n", norm2Faty0);


    // Reduce step-size if chosen step does not lead to reduction by subtracting lambda * h
    size_t n = 0; // safety net, limit iterations
    while (norm2Faty0 > norm2Fatx0 && n++ < 20) {
        lambda /= 2.;
        axpyWithReindexing(_x(sop), _lengthx(sop), -lambda, sop->h, sop->yIndices, sop->lengthY); // xv -= lambda * h // note the -!

        dprintf("x = "); printv(_x(sop), _lengthx(sop));

        buildFx(sop); // rebuild Fx after this change to x
        norm2Faty0 = norm2Fx(sop); // reevaluate norm
        dprintf("reduced stepsize, lambda =  %f, ||F[y0]||_2^2 = %f\n", lambda, norm2Faty0);
    }
    dprintf("optimization finishes, total energy change: %f\n", norm2Faty0 - norm2Fatx0);
    /*assert(norm2Faty0 - norm2Fatx0 <= 0.);*/ // might not be true if early out was used
    return norm2Faty0 - norm2Fatx0;
}

// Interface





// Prototyping functions

FUNCTION(void,
    receiveAndPrintOptimizationData,
    (
    const unsigned int lengthz,
    const unsigned int lengthfz,

    _In_reads_(xLength) const float* const x, const unsigned int xLength,
    _In_reads_(sparseDerivativeZtoYIndicesLength) const unsigned int* const sparseDerivativeZtoYIndices, const unsigned int sparseDerivativeZtoYIndicesLength,
    _In_reads_(xIndicesLength) const unsigned int* const xIndices, const unsigned int xIndicesLength,
    _In_reads_(yIndicesLength) const unsigned int* const yIndices, const unsigned int yIndicesLength
    ),
    "Receives x, sparseDerivativeZtoYIndices, xIndices and yIndices, checks and prints them,"
    "emulating arbitrary lengthz, lengthfz"
    "Note: lengthz, lengthfz are fixed at compile-time for other functions"
    "This is a prototyping function that does not allocate or copy anything"
    "use for testing"
    ) {

    const unsigned int lengthP = xIndicesLength / lengthz;
    const unsigned int lengthY = yIndicesLength;
    const unsigned int lengthFx = lengthfz * lengthP;
    const unsigned int maxNNZ = (lengthfz*lengthz) * lengthP; // could go down from lengthz to maximum k in sparseDerivativeZtoYIndices
    // or just the actual sum of all such k

    dprintf("lengthz: %d\n", lengthz);
    dprintf("lengthfz: %d\n", lengthfz);
    dprintf("lengthP: %d\n", lengthP);
    dprintf("lengthY: %d\n", lengthY);
    dprintf("lengthFx: %d\n", lengthFx);
    dprintf("maxNNZ: %d\n", maxNNZ);

    assert(lengthz > 0);
    assert(lengthfz > 0);
    assert(lengthY > 0);

    dprintf("x:\n");
    printv(x, xLength);

    dprintf("sparseDerivativeZtoYIndices:\n");
    const unsigned int* p = sparseDerivativeZtoYIndices;

    // TODO develop REPEAT(n) macro loop
    DO(i, lengthP) {
        DBG_UNREFERENCED_LOCAL_VARIABLE(i);

        unsigned int k = *p++;
        assert(k <= lengthz);
        dprintf("---\n");
        printd(p, k); p += k;
        dprintf("-->\n");
        printd(p, k); p += k;
        dprintf("---\n");
    }
    assert(p == sparseDerivativeZtoYIndices + sparseDerivativeZtoYIndicesLength);

    dprintf("xIndices:\n");
    p = xIndices;
    DO(i, lengthP) {
        DBG_UNREFERENCED_LOCAL_VARIABLE(i);
        printd(p, lengthz);
        p += lengthz;
    }
    assert(p == xIndices + xIndicesLength);
    assertEachInRange(xIndices, xIndicesLength, 0, xLength - 1);

    dprintf("yIndices:\n");
    printd(yIndices, yIndicesLength);
    assertEachInRange(yIndices, yIndicesLength, 0, xLength - 1);
}



FUNCTION(
    void,
    makeAndPrintSparseMatrix,
    (
    const unsigned int m,
    const unsigned int n,
    _In_reads_(xlen) float* x,
    unsigned int xlen,
    _In_reads_(ijlen) int* ij,
    const unsigned int ijlen
    ),
    "Creates a sparse matrix from a list of values and a list of pairs of (i, j) indices specifying where to put the corresponding values (triplet form)"
    "Note: This is a prototyping function without any further purpose"
    ) {
    assert(2 * xlen == ijlen);
    assert(xlen <= m*n); // don't allow repeated entries

    SOMEMEM();
    cs* const A = cs_spalloc(m, n, xlen, 1, SOMEMEMP);

    while (xlen--) {
        int i = *ij++;
        int j = *ij++;
        cs_entry(A, i, j, *x++);
    }

    cs_print(A);

    printf("compress and print again:\n");
    const cs* const B = cs_triplet(A, SOMEMEMP);
    cs_print(B);
    printf("done--\n");


    FREESOMEMEM();
}

TEST(makeAndPrintSparseMatrix1) {
    unsigned int const count = 1;
    float x[] = {1.};
    int ij[] = {0, 0};
    makeAndPrintSparseMatrix(1, 1, x, count, ij, 2 * count);
}








// Misc

// "collection of some tests"
TEST(testMain) {
    float x[] = {1, 2};
    printv(x, 2);
    float y[] = {1};
    printv(y, 1);

    unsigned int to[] = {1};
    axpyWithReindexing(x, 2, 1., y, to, 1); // expect 1.000000 3.000000
    printv(x, 2);
    assert(1.f == x[0]);
    assert(3.f == x[1]);

    float z[] = {0, 0};
    unsigned int from[] = {1, 0};
    extract(z, x, 2, from, 2); // expect 3.000000 1.000000
    printv(z, 2);
    assert(3.f == z[0]);
    assert(1.f == z[1]);

    // expect i: 0-9
    FOR(int, i, 0, 10, 1) {
        dprintf("i: %d\n", i);
        //i = 0; // i is const!
    }

    int i = 0;
    REPEAT(10) 
        dprintf("rep i: %d\n", i++);
}


// exercise SOMEMEM and cs_
TEST(mainc) {

    int cij[] = {0, 0};
    int xlen = 1;
    float xc[] = {0.1f};
    float* x = xc;
    int m = 1, n = 1;
    int* ij = cij;

    SOMEMEM();
    cs* A = cs_spalloc(m, n, xlen, 1, SOMEMEMP);

    while (xlen--) {
        int i = *ij++;
        int j = *ij++;
        cs_entry(A, i, j, *x++);
    }

    cs_print(A);
    assert(cs_is_triplet(A));
    assert(!cs_is_compressed_col(A));

    printf("compress and print again:\n");
    A = cs_triplet(A, SOMEMEMP);
    assert(!cs_is_triplet(A));
    assert(cs_is_compressed_col(A));
    cs_print(A);

    FREESOMEMEM();
}



// f like f for SOPDProblem
// assumes basic sanity checks to xIndicesPerPartition etc have been done
template<typename f>
vector<float> SOPDProblemMakeSolveGetX(
    _In_ const vector<float>& x,
    _In_ const vector<vector<unsigned int>>& xIndicesPerPartition,
    _In_ const vector<vector<unsigned int>>& yIndicesPerPartition,
    _In_ const vector<vector<unsigned int>>& sparseDerivativeZtoYIndicesPerPartition,
    _In_ const unsigned int iterations = 1) {
    SOPDProblem<f> p(x, xIndicesPerPartition, yIndicesPerPartition, sparseDerivativeZtoYIndicesPerPartition);
    p.solve(iterations); // TODO supply iteration count
    return p.getX();
}


// --- end of SOPCompiled framework ---




/*
false if f is not defined for in.
out is undefined in that case

true and out is f[in] otherwise
*/
template<typename In, typename Out>
bool definedQ(_In_ const unordered_map<In, Out>& f, _In_ const In& in, _Out_opt_ Out& out) {
    auto outi = f.find(in);
    if (f.end() == outi) return false;
    out = outi->second;
    return true;
}












// --- SOPCompiled interface for C++ ---





























/*
Given nonempty v, build locator such that for all a
    either locator(a) is undefined or
    v[locator(a)] == a

locator should initially be empty
*/
template<typename A>
void build_locator(_In_ const vector<A>& v, _Out_ unordered_map<A, unsigned int>& locator) {
    assert(v.size() > 0);
    assert(locator.size() == 0); // locator's constructor will have been called, so 6001 doesn't apply here

    locator.reserve(v.size());

    unsigned int i = 0;
    for (const auto& a : v) {
        locator[a] = i;
        // now:
        // assert(v[locator[a]] == a);
        i++;
    }
    assert(i == v.size());
    assert(v.size() == locator.size());
}

TEST(build_locator1) {
    typedef int In;
    vector<In> v = {42, 0};
    unordered_map<In, unsigned int> locator;
    build_locator(v, locator);

    // check constraints
    assert(v.size() == locator.size());
    assert(0 == locator[42]);
    assert(1 == locator[0]);
}






























/* 
Given the FiniteMapping f, return a vector v of the same size as f and an injective function g such that

    f(a) == v[g(a)] for all a for which f is defined

g crashes when supplied an argument for which f was not defined.
v & g should be empty initially.
*/
template<typename A, typename B>
void linearize(_In_ const unordered_map<A, B>& f, _Out_ vector<B>& v, _Out_ function<unsigned int(A)>& g) {
    assert(f.size() > 0);
    assert(v.size() == 0);
    assert(!g);

    v.resize(f.size());
    unordered_map<A, unsigned int> g_map;

    unsigned int i = 0;
    for (const auto& fa : f) {
        v[i] = fa.second;
        g_map[fa.first] = i;
        // Now the following holds:
        // assert(v[g_map[fa.first]] == fa.second);

        i++;
    }
    assert(i == f.size());

    // build the closure g
    g = [=](A a) -> unsigned int {
        auto ga = g_map.find(a);
        assert(ga != g_map.end());
        return ga->second;
    };

}

TEST(linearize1) {
    // Construct some f
    typedef int In;
    typedef float Out;

    unordered_map<In, Out> f;
    f[0] = 1.;
    f[1] = 2.;

    // Call linearize
    vector<Out> v; function<unsigned int(In)> g;
    linearize(f, v, g);

    // check constraints
    assert(v.size() == f.size());
    for (auto& fa : f) {
        assert(fa.second == v[g(fa.first)]);
    }
}












/*
Change the FiniteMapping f, such that

    f(a) == v[g(a)]

whenever f(a) was already defined.
g must be injective and defined for every a that f is defined for and return a valid index into v for these a.
v must have the same size as f.
g and v returned by linearize satisfy these constraints relative to the input f.

! "Impure"/reference-modifying function for performance. Other parts of the system will be influenced indirectly (side-effect) if f is referenced from elsewhere.
*/
template<typename A, typename B>
void update_from_linearization(_Inout_ unordered_map<A, B>& f, _In_ const vector<B>& v, _In_ const function<unsigned int(A)> g) {
    assert(f.size() > 0);
    assert(v.size() == f.size());
    assert(g);

    for (auto& fa : f) {
        f[fa.first] = v[g(fa.first)];
    }
}

TEST(update_from_linearization1) {
    // Construct some f
    typedef int In;
    typedef float Out;
    unordered_map<In, Out> f;
    f[0] = 1.;
    f[1] = 2.;

    vector<Out> v; function<unsigned int(In)> g;
    linearize(f, v, g);

    // Modify v
    for (auto& x : v) x += 2.;

    // Propagate updates to f

    assert(f.size() == 2);
    assert(f[0] == 1.);
    assert(f[1] == 2.);

    update_from_linearization(f, v, g);

    assert(f.size() == 2);
    assert(f[0] == 3.);
    assert(f[1] == 4.);
}










template<typename X, typename P> struct SigmaFunction
{
    typedef std::function<vector<X>(P)> T;
};

/*
Evaluate sigma at each p in P to obtain the lengthz many names (list of v \in X) of the variables that should be supplied to f at p.
Convert this list to a set of indices into the variable-name |-> value list x.
Concatenate all of these index-lists.

locateInX should give a valid index into the (implied) variable-value list x or crash when supplied a variable name that does not occur there
    It is an injective function.

xIndices should be empty initially
*/
template<typename X, typename PT>
void SOPxIndices(
    _In_ const function<unsigned int(X)> locateInX, 
    _In_ const vector<PT>& P,
    _In_ const function<vector<X>(PT)>& sigma,
    _Out_ vector<unsigned int>& xIndices
    ) {
    assert(P.size() > 0);
    assert(locateInX);
    assert(sigma);
    assert(xIndices.size() == 0);

    const unsigned int lengthz = restrictSize(sigma(P[0]).size());
    assert(lengthz > 0);

    xIndices.resize(lengthz * P.size());

    unsigned int i = 0;
    for (const auto& p : P) {
        const auto zatp = sigma(p);

        for (const auto& z : zatp) {
            xIndices[i++] = locateInX(z);
        }
    }
    assert(i == xIndices.size());
}

TEST(SOPxIndices1) {
    // X === int, {0, -1} in this order
    // P === short, {0, 1}
    // sigma: at point p, supply variables {-p}
    auto locateInX = [](int i) -> unsigned int {return -i;};
    vector<short> P = {0,1};
    auto sigma = [](short p) -> vector<int> {return{-p}; };

    const unsigned int lengthz = restrictSize(sigma(P[0]).size());
    assert(1 == lengthz);

    vector<unsigned int> xIndices;
    SOPxIndices<int, short>(locateInX, P, sigma, xIndices);
    //assert(xIndices == {0,1});
    assert(xIndices.size() == 2);
    assert(0 == xIndices[0]);
    assert(1 == xIndices[1]);

    // Retry with different P:
    P = {1, 0};
    xIndices.clear();
    SOPxIndices<int, short>(locateInX, P, sigma, xIndices);
    //assert(xIndices == {1,0});
    assert(xIndices.size() == 2);
    assert(1 == xIndices[0]);
    assert(0 == xIndices[1]);
}

/*
locateInX /@ Y

where Y contains only elements for which locateInX is defined.

locateInX should give a valid index into the variable-value list x or crash when supplied a variable name that does not occur there
locateInX should be injective.

yIndices should be void initially
*/
template<typename X>
void SOPyIndices(
    _In_ const function<unsigned int(X)> locateInX, const _In_ vector<X>& Y,
    _Out_ vector<unsigned int>& yIndices
    ) {
    assert(Y.size() > 0);
    assert(locateInX);
    assert(yIndices.size() == 0);

    yIndices.resize(Y.size());

    // --
    transform(Y.begin(), Y.end(),
        yIndices.begin(), locateInX);
    // or
    /*
    unsigned int i = 0;
    for (const auto& y : Y) {
        yIndices[i++] = locateInX(y);
    }
    assert(i == xIndices.size());
    */
    // --
}

TEST(SOPyIndices1) {
    // X === int, {0, -1} in this order
    // Y = {-1,0}
    auto locateInX = [](int i) -> unsigned int { return -i; };
    vector<int> Y = {-1,0};

    vector<unsigned int> yIndices;
    SOPyIndices<int>(locateInX, Y, yIndices);

    //assert(yIndices == {1,0});
    assert(2 == yIndices.size());
    assert(1 == yIndices[0]);
    assert(0 == yIndices[1]);

    // Retry with another Y
    Y = {0, -1};
    yIndices.clear();
    SOPyIndices<int>(locateInX, Y, yIndices);
    assert(2 == yIndices.size());
    assert(0 == yIndices[0]);
    assert(1 == yIndices[1]);
}


/*
for each p in P
    * figure out the variables that will be supplied to f (sigma(p))
    * create empty lists zIndices and yIndices
    * find the indices of these variables, listed as z \in 0:lengthz-1 in Y using locateInY
        if locateInY(variable-z) is undefined, continue;
        push_back(zIndices, z) push_back(yIndices, locateInY(variable-v))
    * let K = size(zIndices) == size(yIndices)
    * append the list {K} <> zIndices <> yIndices to sparseDerivativeZtoYIndices

This list of indices will allow us to figure out which derivatives of f we need and where these vectors are to be placed in the 
jacobian matrix d F / d y, where F(x) = (f(sigma_p(x))_{p in P}

locateInX should crash for undefined x
locateInY is unordered_map because we need to be able to judge whether something is an y or not

The linearization format in sparseDerivativeZtoYIndices is detailed in SOPCompiledFramework.
{{zindices}, {yindices}} /; both of length k

{k, zindices, yindices}

TODO maybe this could be done more efficiently using the SOPxIndices-result and a function that
    translates from x-indices to y-indices.

sparseDerivativeZtoYIndices should be initially empty
*/
template<typename X, typename PT>
void SOPsparseDerivativeZtoYIndices(
    _In_ const function<unsigned int(X)> locateInX, 
    _In_ const unordered_map<X, unsigned int>& locateInY, 
    _In_ const vector<PT>& P,
    _In_ const function<vector<X>(PT)>& sigma,

    _Out_ vector<unsigned int>& sparseDerivativeZtoYIndices
    ) {
    assert(P.size() > 0);
    assert(locateInX);
    assert(locateInY.size() > 0);
    assert(sigma);
    assert(sparseDerivativeZtoYIndices.size() == 0);

    // for checking sigma-constraint
    const unsigned int lengthz = restrictSize(sigma(P[0]).size());
    assert(lengthz > 0);

    for (const auto& p : P) {
        const auto zatp = sigma(p);
        assert(zatp.size() == lengthz); // sigma-constraint

        vector<unsigned int> yIndices, zIndices;
        
        // try to locate each z in y
        unsigned int zIndex = 0;
        for (const auto& z : zatp) {

            unsigned int positionInY;
            if (!definedQ<X, unsigned int>(locateInY, z, positionInY)) goto nextz;
            
            zIndices.push_back(zIndex);
            yIndices.push_back(positionInY);

        nextz:
            zIndex++;
        }

        assert(zIndex == lengthz);
        const unsigned int K = restrictSize(zIndices.size());
        assert(K <= lengthz);
        assert(K == yIndices.size());
        assertLess(zIndices, lengthz);
        //assertLess(yIndices, Y.size())

        sparseDerivativeZtoYIndices.push_back(K);
        sparseDerivativeZtoYIndices.insert(sparseDerivativeZtoYIndices.end(), zIndices.begin(), zIndices.end());
        sparseDerivativeZtoYIndices.insert(sparseDerivativeZtoYIndices.end(), yIndices.begin(), yIndices.end());
    }

    // Post
    assert(sparseDerivativeZtoYIndices.size() >= P.size()); // at least {0, no z-indices, no y-indices} per point
}

/*
Given the data vector x (implicitly), the variables optimized for (y), the points where f is evaluated (p) and the per-point data selection function (sigma),
prepare the index arrays needed by a single SOPCompiled partition.

* sigma(p) must have the same nonzero length for all p in P. It gives the names of variables in x that should be passed to f for the point p.
* xIndices, sparseDerivativeZtoYIndices and yIndices are defined as for the SOPCompiled framework.
    The outputs should initially be empty. They will be initialized for you.

x is given as locator function (locateInX) for efficiency: It can be reused for all partitions.
    Use linearize to compute it.

Note that this function is not specific to any function f. However, the result should be used with a SOPCompiledFramework-instance created for a function f
that matches sigma (i.e. same amount of parameters and same meaning & order).
*/
template<typename X, typename PT>
void prepareSOPCompiledInputForOnePartition(
    _In_ const function<unsigned int(X)> locateInX, 
    _In_ const vector<X>& Y, 
    _In_ const vector<PT>& P,
    _In_ const function<vector<X>(PT)>& sigma,

    _Out_ vector<unsigned int>& xIndices, _Out_ vector<unsigned int>& sparseDerivativeZtoYIndices, _Out_ vector<unsigned int>& yIndices
    ) {
    assert(locateInX);
    assert(Y.size() > 0);
    //assert(y.size() <= x.size()); // at most as many variables optimized over as there are x // cannot be verified because x is given implicitly
    // y should be a set (no repeated elements)
    assert(P.size() > 0);
    assert(sigma);

    assert(xIndices.size() == 0);
    assert(sparseDerivativeZtoYIndices.size() == 0);
    assert(yIndices.size() == 0);

    // Precompute location functions
    unordered_map<X, unsigned int> locateInY;
    build_locator(Y, locateInY);
    assert(locateInY.size() > 0);

    // Compute index sets
    SOPxIndices(locateInX, P, sigma, 
        xIndices);
    SOPyIndices(locateInX, Y, 
        yIndices);
    SOPsparseDerivativeZtoYIndices(locateInX, locateInY, P, sigma,
        sparseDerivativeZtoYIndices);

    // Post
    assert(xIndices.size() >= P.size()); // at least one parameter per point
    assert(sparseDerivativeZtoYIndices.size() >= P.size()); // at least {0, no z-indices, no y-indices} per point
    assert(yIndices.size() == Y.size());
    // assert(allLessThan(yIndices, X.size()))
    // assert(allLessThan(sparseDerivativeZtoYIndices, X.size())) // assuming X is larger than lengthz -- wait it has to be!
}

/*
The type f provides the following static members:
    * unsigned int lengthz, > 0
    * unsigned int lengthfz, > 0
    * void f(_In_reads_(lengthz) const float* const input,  _Out_writes_all_(lengthfz) float* const out)
    * void df(_In_range_(0, lengthz-1) int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out)
        df(i) must be the derivative of f by the i-th argument

    Note: Functions should be CPU and GPU compilable.

Returns the solution vector x of the same size as x.

TODO could modify x in-place for (memory) efficiency - but the stuff is modified on the GPU anyways.
*/
template<typename f>
vector<float> SOPDCompiledSolve(
    _In_ const vector<float>& x,
    _In_ const vector<vector<unsigned int>>& xIndicesPerPartition,
    _In_ const vector<vector<unsigned int>>& yIndicesPerPartition,
    _In_ const vector<vector<unsigned int>>& sparseDerivativeZtoYIndicesPerPartition,
    _In_ const unsigned int iterations = 1) {
    assert(x.size() > 0);
    // forall(i) assert(xIndicesPerPartition[i].size() >= ps[i].size()); // there are at least as many indices into x as there are points, since f has at least 1 argument
    assert(xIndicesPerPartition.size() > 0);
    assert(yIndicesPerPartition.size() == xIndicesPerPartition.size());
    assert(yIndicesPerPartition.size() == sparseDerivativeZtoYIndicesPerPartition.size());
    static_assert(f::lengthz > 0, "");
    static_assert(f::lengthfz > 0, "");

    // TODO implement passing these parameters to the functions from the SOPCompiledFramework
    // make the SOP compiled framework take f as a template parameter as above (but still work with compiled and implicit f -- or copy the code here and adjust)
    // or extract the parts we can share

    auto xSolution = SOPDProblemMakeSolveGetX<f>(x, xIndicesPerPartition, yIndicesPerPartition, sparseDerivativeZtoYIndicesPerPartition, iterations);

    // xSolution = ... (compose from y? and y indices <- more efficient. easier -> query all?)

    // Post
    assert(xSolution.size() == x.size());

    return xSolution;
}

/*
The type fSigma provides the following static members:
* const unsigned int lengthz, > 0
* const unsigned int lengthfz, > 0
* void f(_In_reads_(lengthz) const float* const input,  _Out_writes_all_(lengthfz) float* const out)
* void df(_In_range_(0, lengthz-1) int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out)
df(i) must be the derivative of f by the i-th argument
* vector<X> sigma(P p) gives names of variables supplied to f at p

Note: Functions f and df should be CPU and GPU compilable.

! Modifies x in-place to contain the solution for efficiency. Other parts of system might be influenced by this *side-effect*!
*/
template<typename X, typename PT, typename fSigma>
void SOPDSolve(
    _Inout_ unordered_map<X, float>& x, 
    _In_ const vector<vector<PT>>& ps,
    _In_ const vector<vector<X>>& ys,
    _In_ const unsigned int iterations = 1) {
    assert(iterations > 0);
    assert(ps.size() > 0);
    assert(ps.size() == ys.size()); // amount of partitions
    assert(x.size() > 0);
    assert(ys[0].size() > 0);
    assert(ps[0].size() > 0);

    static_assert(fSigma::lengthz > 0, "lengthz must be positive");
    static_assert(fSigma::lengthfz > 0, "lengthfz must be positive");
    assert(&fSigma::sigma);
    assert(fSigma::sigma(ps[0][0]).size() > 0);
    assert(&fSigma::f);
    assert(&fSigma::df);

    const unsigned int partitions = restrictSize(ps.size());
    assert(partitions > 0);

    // prepare x
    function<unsigned int(X)> locateInX;
    vector<float> xVector;
    linearize(x, xVector, locateInX);

    // prepare indices
    vector<vector<unsigned int>> xIndicesPerPartition(partitions);
    vector<vector<unsigned int>> yIndicesPerPartition(partitions);
    vector<vector<unsigned int>> sparseDerivativeZtoYIndicesPerPartition(partitions);
    DO(i, partitions) {
        prepareSOPCompiledInputForOnePartition<X, PT>(
            /* in */
            locateInX, ys[i], ps[i], &fSigma::sigma,
            /* out */
            xIndicesPerPartition[i], sparseDerivativeZtoYIndicesPerPartition[i], yIndicesPerPartition[i]
            );

        assert(xIndicesPerPartition[i].size() > 0);
        assert(sparseDerivativeZtoYIndicesPerPartition[i].size() > 0); // todo there should be at least one row {k, y-ind,z-ind} in this array with k!=0
        assert(yIndicesPerPartition[i].size() > 0);
    }

    // solve compiled
    auto x1Vector = SOPDCompiledSolve<fSigma>(xVector, xIndicesPerPartition, yIndicesPerPartition, sparseDerivativeZtoYIndicesPerPartition, iterations);
    assert(x1Vector.size() == xVector.size());

    // output
    update_from_linearization(x, x1Vector, locateInX);

    // 
}

































/* example 

PTest[

Block[{select, sop, x},

select[i_] := {IdentityRule[x]};
sop = SparseOptimizationProblemDecomposedMake[{1-x},
select, {{0}}, {x -> 2.`}, {{x}}];

SOPDDataAsRules[SOPDSolve[sop, Method -> "SOPCompiled"]]

] (*end of Block*)

,
{x -> 1.`}
]

*/
enum Example1X { e1x };
enum Example1P { e1p0 };

struct Example1fSigma {
    static
    CPU_FUNCTION(vector<Example1X>, sigma, (Example1P p), "the per-point data selector function") {
        DBG_UNREFERENCED_PARAMETER(p);
        return{e1x};
    }

    static const unsigned int lengthz = 1, lengthfz = 1;

    static
    MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
    "per point energy"){
        out[0] = 1.f-input[0];
    }

    static
    MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f") {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
        out[0] = -1.f;
    }
};

TEST(SOPDSolve1) {
    unordered_map<Example1X, float> x = {{e1x, 2.f}};
    assert(x[e1x] == 2.f);

    
    SOPDSolve<Example1X, Example1P, Example1fSigma>(x, {{e1p0}}, {{e1x}});

    assert(1 == x.size());
    assert(approximatelyEqual(x[e1x], 1.f), "%f", x[e1x]);
}

/* example 2, nonlinear: root of 2.

PTest[
Block[{select, sop, x}, select[i_] := {IdentityRule[x]};
sop = SparseOptimizationProblemDecomposedMake[{2 - x*x},
select, {{0}}, {x -> 2.`}, {{x}}];
SOPDGetX0[SOPDSolve[sop, MaxIterations -> 16]]] (*end of Block*)
, Sqrt@2., {SameTest -> ApproximatelyEqual}]

*/

struct Example2fSigma {
    static
        CPU_FUNCTION(vector<Example1X>, sigma, (Example1P p), "the per-point data selector function") {
        DBG_UNREFERENCED_PARAMETER(p);
        return{e1x};
    }

    static const unsigned int lengthz = 1, lengthfz = 1;

    static
        MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
        "per point energy"){
        out[0] = 2.f - input[0] * input[0];
    }

    static
        MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f") {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
        out[0] = -2.f* input[0];
    }
};

TEST(SOPDSolve2) {
    unordered_map<Example1X, float> x = {{e1x, 2.f}};
    assert(x[e1x] == 2.f);


    SOPDSolve<Example1X, Example1P, Example2fSigma>(x, {{e1p0}}, {{e1x}}, 16);

    assert(1 == x.size());
    assert(approximatelyEqual(x[e1x], sqrtf(2.f)), "%f", x[e1x]);
}

/* example 3, nonlinear & multi-variable, not optimized over everything: root of 2. and 3.
data:

-1 -> 2.
1 -> x

-2 -> 3.
2 -> y

sigma(p): P = {1,2}

a -> p
b -> -p

f(a,b) = b - a*a

Y = {1,2}
*/
typedef int Example3X;
typedef int Example3P;

struct Example3fSigma {
    static
        CPU_FUNCTION(vector<Example3X>, sigma, (Example3P p), "the per-point data selector function") {
        DBG_UNREFERENCED_PARAMETER(p);
        return {p, -p};
    }

    static const unsigned int lengthz = 2, lengthfz = 1;

    static
        MEMBERFUNCTION(void, f, (_In_reads_(lengthz) const float* const input, _Out_writes_all_(lengthfz) float* const out),
        "per point energy"){
        out[0] = input[1] - input[0] * input[0];
    }

    static
        MEMBERFUNCTION(void, df, (_In_range_(0, lengthz - 1) unsigned int const i, _In_reads_(lengthz) float const * const input, _Out_writes_all_(lengthfz) float * const out), "the partial derivatives of f") {
        DBG_UNREFERENCED_PARAMETER(input);
        assert(i < lengthz);
        switch (i) {
        case 0: out[0] = - 2.f * input[0]; return;
        case 1: out[1] = 1.f; 
            fatalError("this derivative should not be needed in the given objective");
            return; 
        }
    }
};

TEST(SOPDSolve3) {
    unordered_map<Example3X, float> x = {{-1, 2.f}, {1, 2.f}, {-2, 3.f}, {2, 3.f}};
    SOPDSolve<Example3X, Example3P, Example3fSigma>(x, /*p*/{{1,2}}, /*Y*/{{1,2}}, /*iterations*/16);

    assert(4 == x.size());
    assert(approximatelyEqual(x[1], sqrtf(2.f)), "%f", x[1]);
    assert(approximatelyEqual(x[2], sqrtf(3.f)), "%f", x[2]);
    // rest perfectly unchanged
    assert(x[-1] == 2.f);
    assert(x[-2] == 3.f);
}


/* example 4: same as example 3, but with the Y in two separate partitions 
the result should be exactly the same, but the computation is parallelizable now*/

TEST(SOPDSolve4) {
    unordered_map<Example3X, float> x = {{-1, 2.f}, {1, 2.f}, {-2, 3.f}, {2, 3.f}};
    SOPDSolve<Example3X, Example3P, Example3fSigma>(x, /*p*/{{1}, {2}}, /*Y*/{{1}, {2}}, /*iterations*/16);

    assert(4 == x.size());
    assert(approximatelyEqual(x[1], sqrtf(2.f)), "%f", x[1]);
    assert(approximatelyEqual(x[2], sqrtf(3.f)), "%f", x[2]);
    // rest perfectly unchanged
    assert(x[-1] == 2.f);
    assert(x[-2] == 3.f);
}

int main() {
    runTests();
}