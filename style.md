# Style Guide

The following guidelines are used for code in the lecture notes (i.e., not to the `QuantEcon.jl` package).  See [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) for the baseline guidelines, which this supplements.

## Basic Principles

Keep in mind that these lectures are targeted at students with (at most!) some self-taught Matlab or Python experience.  Consequently, we want to ensure that the code is clear and simple to focus on the Economics/Math and only expect them to write code using a simplified subset of Julia features.  Some guiding principles:

1. Assume this may be the **first programming language** students learn
2. Use **compact, script-style** code, organized into functions only when it is natural.  Best practices for writing packages and expository/exploratory/interactive code can be different.
3. Keep things as **close to the whiteboard math** as possible, including in the code structure and notation
4. Ensure that all **code can be copied and pasted without modification** into functions for performance and modification without changes to scoping (e.g. no `local` or `global` ever required)
5. **Avoid type annotations** unless they are required for dispatching
6. **Avoid creating custom types** unless it is absolutely necessary
7. Beyond the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/), avoid unnecessary whitespace lines and redundant comments
8. Don't use fancy features and control flow from Julia - unless it makes the code look closer to the math
9. Avoid both micro-optimizations and coding patterns that pessimize (i.e. poor performance with no benefit in code clarity)

We want users to be able to say _"the code is clearer than Matlab, and even closer to the math"_.  If they want to write their own packages, they will need to learn much more about Julia, but most never will need to.

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


## Control Structures and Flow

- **Avoid:** the short-circuiting pattern unless it is very clear.  This is especially true for assertions which shouldn't be in the normal control flow
```julia
# BAD!
a < 0 && error("`a` parameter invalid")

# GOOD!
@assert a < 0 
```
- **Do not** use a `;` to call functions with keyword arguments unless it is required.  i.e. prefer `range(1, stop=5)` to `range(1; stop=5)`

## Dependencies

- **Use external packages** whenever possible, and never rewrite code that is available in a well-maintained external package (even if it is imperfect)
- The following packages can be used as a dependency without any concerns: `QuantEcon, Parameters, Optim, Roots, Expectations, NLsolve, DataFrames, Plots, ....`
- **Do** use `using` where possible (i.e. not `import`), and include the whole package as opposed to selecting only particular functions or types.
- **Prefer** to keep packages used throughout the lecture at the top of the first block (e.g. `using LinearAlgebra, Parameters, QuantEcon`)  but packages used only in a single place should have the `using` local to that use.
    - If `Plots` is only used lower down in the lecture, then try to have it local to that section to ensure faster loading time.