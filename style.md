# Style Guide
The following guidelines are used for code in the lecture notes (i.e., not to the `QuantEcon.jl` package).  See [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) for the baseline guidelines, which this supplements.

## Basic Principles
Keep in mind that these lectures are targeted at students with (at most) a limited amount of Matlab or Python experience.  Consequently, we want to ensure that the code is clear and simple (even if it performs magic in the background).  Some guiding principles
1. Assume this may be the first programming language students learn
2. Keep things as **close to the whiteboard math** as possible, including in the code structure and notation
3. **Avoid type annotations** unless they are required for dispatching
4. Beyond the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/), avoid unnecessary whitespace and redundant comments
5. Don't use fancy features and control flow from Julia - unless it makes the code look closer to the math
6. Avoid both micro-optimizations and coding patterns that pessimize (i.e. poor performance with no benefit in code clarity)

We want users to be able to say _"the code is clearer than Matlab, and even closer to the math"_.

## Naming Conventions, Comments, etc.
- **Use unicode** where possible in names so that symbols match the math in the document
- **Be careful** about unicode glyphs and symbols which may not be available in the default REPL, Jupyter, etc. for all platforms.  **TODO** 
- **Do not** use extra whitespace, use comment headers, or redundant comments.  For example, **do not**
```julia
# BAD!
foo(a) #Calls the foo function

# == Parameters == #

bar = 2.0 

# GOOD!
foo(a)

# Parameters
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


## Type Annotations, Parameters, and Generic Programming
- **Do not** use type annotations in function arguments unless required (e.g. for multiple-dispatch).  Let the compiler figure it out with duck-typing
- **Do not** use `struct` purely for collecting a set of parameters (but with no multiple dispatch).  Instead, use named tuples.  To enable construction of different parameters with defaults, use the `@with_kw` from `Parameters.jl`.
```julia
# BAD!
struct MyParams{T}

end
params = MyParams(2.0)

# GOOD!
params = (a = 2.0, b = [1 2 3])

# BETTER!
myparams = @with_kw (a = 10.0, b = [1 2 3])
params = myparams(a = 2.0) # -> (a=2.0, b=[1 2 3])
```
- 

## Control Structures and Flow
- **Avoid:** the short-circuiting pattern unless it is very clear.  This is especially true for assertions which shouldn't be in the normal control flow
```julia
# BAD!
a < 0 && error("`a` parameter invalid")

# GOOD!
@assert a < 0 
```

## Dependencies
- **Use external packages** whenever possible, and never rewrite code that is available in a well-maintained external package (even if it is imperfect)
- The following packages can be used as a dependency without any concerns: `QuantEcon, Parameters, Optim, Roots, Expectations, NLsolve, DataFrames, Plots, ....`
- **Do:** use `using` where possible (i.e. not `import`), and include the whole package as opposed to selecting only particular functions or types.
