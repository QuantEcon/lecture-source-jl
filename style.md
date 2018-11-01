# Style Guide

The following guidelines are used for code in the lecture notes (i.e., not to the `QuantEcon.jl` package).  See [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) for the baseline guidelines, which this supplements.

## Basic Principles

Keep in mind that these lectures are targeted at students with (at most!) some self-taught Matlab or Python experience.  Consequently, we want to ensure that the code is clear and simple to focus on the Economics/Math and only expect them to write code using a simplified subset of Julia features.  Some guiding principles:

1. Assume this may be the **first programming language** students learn
2. Use **compact, script-style** code, organized into functions only when it is natural.  Best practices for writing packages and expository/exploratory/interactive code can be different.
3. Keep things as **close to the whiteboard math** as possible, including in the code structure and notation
4. **Don't be clever**, be clear.  Terse is not a virtue unless the terseness makes the code clearer, or closer to the whiteboard math
4. Maintain this **correspondence between math and code** even if the code is less efficient.  Only optimize if it is really necessary.
5. Ensure that all **code can be copied and pasted without modification** into functions for performance and modification without changes to scoping (e.g. no `local` or `global` ever required)
6. **Avoid type annotations** unless they are required for dispatching
7. **Avoid creating custom types** unless it is absolutely necessary
8. Beyond the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/), avoid unnecessary whitespace lines and redundant comments
9. Don't use fancy features and control flow from Julia - unless it makes the code look closer to the math
10. Avoid both micro-optimizations and coding patterns that pessimize (i.e. poor performance with no benefit in code clarity)

We want users to be able to say _"the code is clearer than Matlab, and even closer to the math"_.

## Naming Conventions, Comments, etc.

- **Use unicode for math, ascii for control flow** where possible in names so that symbols match the math in the document
- **Use ascii for control flow** That is,
    - Use `in` instead of `∈`, `!=` instead of `≠`, and `<=` instead of `≤` when writing code.
    - Use `∈` and `∉` when implementing math for sets
- **Be careful** about unicode glyphs and symbols which may not be available in the default REPL, Jupyter, etc. for all platforms.  **TODO** 
- **Do not** use extra whitespace, use comment headers, or redundant comments.  For example, **do not**
```julia
# BAD!
foo(a) #Calls the foo function

# == Parameters == #

bar = 2.0 

# GOOD!
foo(a)

# parameters
bar = 2.0
```
- **Do not** align the `=` sign for construction of variables (though acceptable for matrices).  i.e.
```julia
# BAD!
var1       =  1.0
variable2  =  2.0

# GOOD!
var1 = 1.0
variable2 = 2.0

# ACCEPTABLE BUT OFTEN UNNECESSARY
A = [1 2;
     3 4]
```
- **Do not** use docstrings in any lecture code - except when explaining packages and the `?` environment.
- **Feel free** to use the `⋅` unicode symbol, i.e. `\cdot<TAB>` instead of `dot( , )`
- **Avoid the use of LaTeX** as it does not work well with most graphics backends
  - But if you do, *use `LaTeXStrings.jl`** for all latex literals, i.e. `L"\hat{\alpha}"` instead of `""\$\\hat{\\alpha}\$""`
- **Prefer** `in` to `∈` 
- Comment spacing
  - Comments on their own lines, which are generally prefered, and without capitalization unless intending emphasis
```julia
x = 1

# comment1
x = 2
```
  - Comments on the same line of code
```julia
x = 1 # comment2
```
- **Add comment for equation to code correspondence whenever possible**.  That is, if there was a formula in the document at some point, say `b = a x^2 (14)` where the `14` is the equation number when rendering, then the code which implements it should be
```julia
# GOOD!
b = a * x^2 # (14)
```
  - The exception to this rule is if it is immediately clear that a line of code represents a formula due to immediate proximity in the document.  Always err on the side of extra equation numbers, and add it to the `rst` as required.

## Type Annotations, Parameters, and Generic Programming

- **Do not** use type annotations in function arguments unless required (e.g. for multiple-dispatch).  Let the compiler figure it out with duck-typing
- **Do not** use `struct` purely for collecting a set of parameters (but with no multiple dispatch).  Instead, use named tuples.  To enable construction of different parameters with defaults, use the `@with_kw` from `Parameters.jl`.
```julia
# BAD!
struct MyParams{TF <: Real,
                TAV <: AbstractVector{TF}}
    a::TF
    b::TAV    
end
MyParams(a::TF = 10.0, b::TAV = [1.0, 2.0, 3.0]) where {TF, TAV} = MyParams{TF, TAV}(a,b)
params = MyParams(2.0)

# GOOD!
params = (a = 2.0, b = [1.0 2.0 3.0])

# BETTER!
myparams = @with_kw (a = 10.0, b = [1 2 3]) # generates new named tuples with defaults
params = myparams(a = 2.0) # -> (a=2.0, b=[1.0 2.0 3.0])
myparamsdefaults = myparams() # -> (a = 10.0, b = [1.0 2.0 3.0])
```
- **Use `@unpack`** instead of manually unpacking variables from structures and/or named tuples.
```julia
param = (a=2, b=1.0, c = [1, 2, 3])

# BAD!
function f2(p)
    a, b, c = p.a, p.b, p.c
    return a + b
end
f2(param)

# GOOD!
function f(p)
    @unpack a, b, c = p
    return a + b
end
f(param)
```

## General Control Structures and Code Organization

- **Return Named Tuples** from functions with multiple return values.  That is,

```julia 
# BAD
function foo(x, y, z)
    return (x, y, z) # Julia does this anyway 
end 

~, ~, z = foo(x, y, z) # when we want z

# GOOD 
function foo(x, y, z)
    return (x = x, y = y, z = z)
end

# when we want z 
@unpack z = foo(x, y, z)
z = foo(x, y, z).z
``` 

- **Avoid inplace functions if possible** unless the library requires it, or the vectors are enormous.  That is,
```julia
# BAD! (unless out is a preallocated and very large vector)
function f!(out, x)
    out .= 2 * x
end

# GOOD
function f(x)
   return 2 * x
end

# BEST
f(x) = 2 * x
```
  - The main problem is that the semantics of variable bindings are subtle in julia.  They are likely to accidentally go `out = 2 * x` and it would silently fail because it renames the `out` variable, and doesn't rewrite the contents.
  - The two other reasons are performance:  the compiler can often compile and inline better with out-of-place, and it becomes possible to use `StaticArrays` and other packages which use immutable data-structures.

- **Avoid declaring variable scope** such as `local` and `global` in order to allow copy/paste in different contexts
    - A key requirement is that the source-code we write can be copied directly _inside_ a function and have it work.  Variable scoping breaks this.
    - This comes up especially when trying to use `for` loops at the top level.  For these, note that with the new "soft global scope" in `IJulia.jl`, Jupyter automatically works without worrying.  The REPL may eventually, but at this point copying to the REPL with for loops isn't possible.

- **Avoid:** the short-circuiting pattern unless it is very clear.  This is especially true for assertions which shouldn't be in the normal control flow
```julia
# BAD!
a < 0 && error("`a` parameter invalid")

# GOOD!
@assert a < 0 
```
- **Use** the adjoint notation, `A'` instead of calling `transpose(A)` directly when working with real matrices (where they are identical).  With complex matrices, use whichever is appropriate.
- **Use** the notation for stacking instead of the functions.  That is,
```julia
a = [1 2]
b = [3 4]

# BAD!
hcat(a, b)
vcat(a, b)

# GOOD!
[a b]
[a; b] # or,
[a;
 b]
```
- **Use `I`**, the `UniformScaling` type, instead of constructing concrete matrices.
```julia
using LinearAlgebra
A = [1 2; 3 4] 

# BAD!
Ival = [1 0; 0 1]
A + Ival
A + 2 * Ival

# GOOD!
A + I
A + 2*I
```
- **Slice with copy for clarity**, and if necessary use `@views`
```julia

# GOOD! (usually)
A = [1 2; 3 4]
A[:, 1]

#GOOD when views necessary 
A = [1 2; 3 4]
@views A[:, 1]

# BAD!
A = [1 2; 3 4]
view(A, :, 1)
```
- **Preallocate with `similar`** whenever possible, and avoid type annotations unless necessary.  The goal is to maintain code independent of types, which will aid later in generic programming.  Where you cannot, just allocate zeros, etc. instead 
```julia
N = 5
x = [1.0, 2.0, 3.0, 4.0, 5.0]

# BAD!
y = Array{Float64}(undef, N)
A = Array{Float64}(undef, N, N)

# BETTER!
y = zeros(N) # if we want the default, floats
A = zeros(N,N)

# BEST! (if a candidate `x` exists)
y = similar(x, N) # keeps things generic.  The `N` here is not required if the same size
A = similar(x, N, N) # same type but NxN size
```
- **Create vector literals with `,`** rather than `;` when possible
```julia
# BAD!
y = [1; 2; 3]

# GOOD!
y = [1, 2, 3]
```
- **Leave matrix/vector types as returned types as long as possible**.  That is, avoid `Matrix(...)` just for conversion, leaving multiple-dispatch to do its job.
```julia
x = [1 0; 0 1]

# note, typeof(Q) ==  LinearAlgebra.QRCompactWYQ{Float64,Array{Float64,2}}
Q, R = qr(x)


# BAD!
Q = Matrix(Q)
val = Q * Q' # some calculation using Q and other things

# GOOD!
val = Q * Q' # directly, without any conversion
```
- **Don't use  `push!` when clearer alternatives exist** as it is harder for introductory reasoning and the size is preallocated.  But try to use broadcasting, comprehensions, etc. if clearer
```julia
# BAD!
N = 5
x = [] # really bad since it is an Any vector!
for i in 1:N
    push!(x, 2.0 * i^2)
end
x

# Better!
x = zeros(N)
for i = 1:N
    x[i] = 2.0 * i^2
end
x

# Best!
x = [2.0 * i^2 for i in 1:N]

# or
f(i) = 2.0 * i^2
x = f.(1:5) # use broadcasting
```
- Prefer `eachindex` to accessing the sizes of vectors.
```julia
x = [1.0, 2.0, 5.0, 2.1]
y = similar(x)

# BAD!
for i in 1:length(x)
    y[i] = x[i] + 2
end
y

# BETTER!
for i in eachindex(x)
    y[i] = x[i] + 2
end
y

# GOOD! No way to preallocate y (although easier ways to write)
x = rand(10)
y = similar(x, 0) # empty of same type as x
for val in x
    if val < 0.5
        push!(y, val)
    end
end
y
```
- **Use iterators directly** rather than accessing by index.
```julia
x = [1, 2, 3]

# BAD!
n = 0
for i in eachindex(x)
  n += x[i]^2
end
n

# BETTER!
n = 0
for val in x
  n += val^2
end
n

# BETTER!
sum(xval -> xval^2, x) # i.e. transform each x and then reduce

#BEST! keep it simple with broadcasting
sum(x.^2)
```

- **Use `eachindex`** to iterate through matrices and arrays of dimension > 2 as long as you don't need the actual index.  Otherwise,   For example,
```julia
A = [1 2 3; 4 5 6]
# BAD!
...for loops TODO

# BETTER!
for i in eachindex(A)
    B[i] = A[i]^2
end
```
TODO: when you need the `i` and `j` what do you do?  Just loop over both... 

- **Avoid `range` when possible** and use the `1.0:0.1:1.0` style notation, etc.
```julia
 # BAD!
 range(1, stop=5)

# GOOD!
1:5 
```    
  - Furthermore, if you don't really care if if hits the `zstop` exactly and are willing to give a stepsize then, the following is the clearest
```julia
zmin = 0.0
zstop = 1.0
step = 0.1

# GOOD! But..
r = zmin:step:zstop

# CAREFUL!
r = 0.0:0.22:1.0 # note the end isn't a multiple of the step...
@assert r == 0.0:0.22:0.88
@assert maximum(r) == 0.88 # use to get the maxium of the range, perhaps != 
```
- **Use the new `range` from Compat.jl**. This provides code compatible with Julia 1.1
```julia
# BAD! but the only pure Julia 1.0 version
range(0.0, stop=1.0, length = 10)

# GOOD! but requires Julia 1.1 or Compat
using Compat
range(0.0, 1.0, length=10)
```

- **Minimize use of the ternary operator**.  It is confusing for new users, so use it judiciously, and never purely to make code more terse.

- **Square directly instead of abs2**. There are times where `abs2(x)` is faster than `x^2`, but they are rare, and it is harder to read.
- **Do not use compehensions or generators simply to avoid a temporary**. That is
```julia
x = 1:3
f(x) = x^2

# BAD (even if sometimes faster)
sum(f(xval) for xval in x)
mean(f(xval) for xval in x)

# GOOD!
sum(f.(x))
mean(f.(x))
```
- **Only use comprehension syntax when it makes code clearer**.  Code using single comprehensions can help code clarity and the connection to the mathematica, or it can obfuscate things.  repeated use of `for` in the same comprehension is the hardest to understand.  A few examples
```julia
# BAD! Tough to mentally parse that this is a nested loop creating a single vector vs. a matrix
[i + j for i in 1:3 for j in 1:3]

# GOOD! easier to read that this creates an addition table
[i + j for i in 1:3, j in 1:3]

# OK, and easy enough to read
X = 1:3
[x + 1 for x in X]

# BAD! Tough to read since multiple comprehensiosn
X = 1:3
Y = [1 2 3]

[x^2 + sum(x * y + 1 for y in Y) for x in X]

# BETTER! Put everything into a function
f(x) = x^2 + sum(x * y + 1 for y in Y)
[f(x) for x in X]

#... but at that point?
# BEST! remove the generator/comprehension and just use broadcasting
f(x) = x^2 + sum(x * Y .+ 1)
f.(X)
```

- **Careful with function programming patterns**.  Sometimes they can be clear, but be careful.  In particular, be wary of uses of `reduce`, `mapreduce` and excessive use of `map`
```
# BAD! 
x = 1:3
mapreduce(xval -> xval^2, +, x)

# GOOD! Direct for this case
sum(x.^2)

# QUESTIONABLE! Explain carefully with a comment if using
X = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
reduce(hcat, X)

# GOOD, if verbose 
Y = ones(3,3)
for i in 1:3
    Y[:,i] = X[i]
end
```
The exception, of course, is when dealing with parallel programming where functional patterns are essential.

## Error Handling
- **Call the return from optimizers/etc. `results` when possible**
- **Can Add `converged`, etc. with `using` into the namespace to make the code easier to read**
  - And can safely ignore the conflicting method errors, until smarter method merging becomes possible.
- **Can use the || idiom for error handling**.
  - Eventually people will need to get used to it.  But minimize its use outside of that case
- **Don't ignore errors from fixed-point or solvers**.  In the case below, we can just raise an error so it isn't ignored
```julia
using NLsolve
f(x) = x
x_iv = [1.0]
#f(x) = 1.1 * x # fixed-point blows

# BAD!
xstar = fixedpoint(f, x_iv, inplace=false).zero # assumes convergence
xsol = nlsolve(f, x_iv, inplace=false).zero # assumes convergence

# GOOD!
result = fixedpoint(f, x_iv, inplace=false)
converged(result) || error("Failed to converge in $(result.iterations) iterations")
xstar = result.zero

result = nlsolve(f, x_iv, inplace=false)
converged(result) || error("Failed to converge in $(result.iterations) iterations")
xsol = result.zero
```
- **Handle errors by returning nothing**.  But if you won't handle, prefer to throw errors.
```julia
function g(a)
    f(x) = a * x # won't succeed if a > 1
    result = fixedpoint(f, [1.0], inplace = false)
    converged(result) || return nothing
    xstar = result.zero
    
    #do calculations...
    return xstar
end
val = g(0.8)
@show val == nothing
val = g(1.1)
@show val == nothing;
```
- **Use similar patterns with the Optim and other libraries**
```
# GOOD: make it easier to use, even if there are a few method merge warnings
using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations

# BAD
xmin = optimize(x-> x^2, -2.0, 1.0).minimum

#GOOD
result = optimize(x-> x^2, -2.0, 1.0)
converged(result) || error("Failed to converge in $(iterations(result)) iterations")
xmin = minimizer(result)
minimum(result)

using Optim: converged, maximum, maximizer, minimizer, iterations
f(x) = -x^2
result = maximize(f, -2.0, 1.0)
converged(result) || error("Failed to converge in $(iterations(result)) iterations")
xmin = maximizer(result)
fmax = maximum(result)

```



## Dependencies

- **Use external packages** whenever possible, and never rewrite code that is available in a well-maintained external package (even if it is imperfect)
- The following packages can be used as a dependency without any concerns: `QuantEcon, Parameters, Optim, Roots, Expectations, NLsolve, DataFrames, Plots, Compat`
- **Do** use `using` where possible (i.e. not `import`), and include the whole package as opposed to selecting only particular functions or types.
- **Prefer** to keep packages used throughout the lecture at the top of the first block (e.g. `using LinearAlgebra, Parameters, Compat`)  but packages used only in a single may have the `using` local to that use to ensure students know which package it comes from
    - If `Plots` is only used lower down in the lecture, then try to have it local to that section to ensure faster loading time.
- **Always seed random numbers** in order for automated testing to function using `seed!(...)`

## Work in Progress Discussions
1. How best to stack arrays and unpack them for use with solvers/etc.?  `vec` was mentioned?
