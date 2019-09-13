.. _fundamental_types:

.. include:: /_static/includes/header.raw

*****************************************************
Arrays, Tuples, Ranges, and Other Fundamental Types
*****************************************************

.. contents:: :depth: 2

.. epigraph::

    "Let's be clear: the work of science has nothing whatever to do with consensus.
    Consensus is the business of politics. Science, on the contrary, requires only
    one investigator who happens to be right, which means that he or she has
    results that are verifiable by reference to the real world. In science
    consensus is irrelevant. What is relevant is reproducible results." -- Michael Crichton

Overview
============================

In Julia, arrays and tuples are the most important data type for working with numerical data

In this lecture we give more details on

* creating and manipulating Julia arrays

* fundamental array processing operations

* basic matrix algebra

* tuples and named tuples

* ranges

* nothing, missing, and unions

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia

    using LinearAlgebra, Statistics

Array Basics
================

(`See multi-dimensional arrays documentation <https://docs.julialang.org/en/v1/manual/arrays/>`_)

Since it is one of the most important types, we will start with arrays

Later, we will see how arrays (and all other types in Julia) are handled in a generic and extensible way

Shape and Dimension
----------------------

We've already seen some Julia arrays in action


.. code-block:: julia

    a = [10, 20, 30]


.. code-block:: julia

    a = [1.0, 2.0, 3.0]


The output tells us that the arrays are of types ``Array{Int64,1}`` and ``Array{Float64,1}`` respectively

Here ``Int64`` and ``Float64`` are types for the elements inferred by the compiler

We'll talk more about types later

The ``1`` in ``Array{Int64,1}`` and ``Array{Any,1}`` indicates that the array is
one dimensional (i.e., a ``Vector``)

This is the default for many Julia functions that create arrays

.. code-block:: julia

    typeof(randn(100))

In Julia, one dimensional vectors are best interpreted as column vectors, which we will see when we take transposes

We can check the dimensions of ``a`` using ``size()`` and ``ndims()``
functions

.. code-block:: julia

    ndims(a)


.. code-block:: julia

    size(a)


The syntax ``(3,)`` displays a tuple containing one element -- the size along the one dimension that exists

Array vs Vector vs Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Julia, ``Vector`` and ``Matrix`` are just aliases for one- and two-dimensional arrays
respectively

.. code-block:: julia

    Array{Int64, 1} == Vector{Int64}
    Array{Int64, 2} == Matrix{Int64}


Vector construction with ``,`` is then interpreted as a column vector

To see this, we can create a column vector and row vector more directly

.. code-block:: julia

    [1, 2, 3] == [1; 2; 3]  # both column vectors

.. code-block:: julia

    [1 2 3]  # a row vector is 2-dimensional

As we've seen, in Julia we have both

* one-dimensional arrays (i.e., flat arrays)

* arrays of size ``(1, n)`` or ``(n, 1)`` that represent row and column vectors respectively

Why do we need both?

On one hand, dimension matters for matrix algebra

* Multiplying by a row vector is different to multiplying by a column vector

On the other, we use arrays in many settings that don't involve matrix algebra

In such cases, we don't care about the distinction between row and column vectors

This is why many Julia functions return flat arrays by default


.. _creating_arrays:

Creating Arrays
------------------


Functions that Create Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We've already seen some functions for creating a vector filled with ``0.0``

.. code-block:: julia

    zeros(3)

This generalizes to matrices and higher dimensional arrays

.. code-block:: julia

    zeros(2, 2)

To return an array filled with a single value, use ``fill``

.. code-block:: julia

    fill(5.0, 2, 2)

Finally, you can create an empty array using the ``Array()`` constructor

.. code-block:: julia

    x = Array{Float64}(undef, 2, 2)


The printed values you see here are just garbage values

(the existing contents of the allocated memory slots being interpreted as 64 bit floats)

If you need more control over the types, fill with a non-floating point

.. code-block:: julia

    fill(0, 2, 2)  # fills with 0, not 0.0

Or fill with a boolean type

.. code-block:: julia

    fill(false, 2, 2)  # produces a boolean matrix



Creating Arrays from Existing Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the most part, we will avoid directly specifying the types of arrays, and let the compiler deduce the optimal types on its own

The reasons for this, discussed in more detail in :doc:`this lecture <../more_julia/generic_programming>`, are to ensure both clarity and generality

One place this can be inconvenient is when we need to create an array based on an existing array

First, note that assignment in Julia binds a name to a value, but does not make a copy of that type

.. code-block:: julia

    x = [1, 2, 3]
    y = x
    y[1] = 2
    x

In the above, ``y = x`` simply creates a new named binding called ``y`` which refers to whatever ``x`` currently binds to

To copy the data, you need to be more explicit

.. code-block:: julia

    x = [1, 2, 3]
    y = copy(x)
    y[1] = 2
    x

However, rather than making a copy of ``x``, you may want to just have a similarly sized array

.. code-block:: julia

    x = [1, 2, 3]
    y = similar(x)
    y


We can also use ``similar`` to pre-allocate a vector with a different size, but the same shape

.. code-block:: julia

    x = [1, 2, 3]
    y = similar(x, 4)  # make a vector of length 4

Which generalizes to higher dimensions

.. code-block:: julia

    x = [1, 2, 3]
    y = similar(x, 2, 2)  # make a 2x2 matrix

Manual Array Definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we've seen, you can create one dimensional arrays from manually specified data like so

.. code-block:: julia

    a = [10, 20, 30, 40]

In two dimensions we can proceed as follows

.. code-block:: julia

    a = [10 20 30 40]  # two dimensional, shape is 1 x n


.. code-block:: julia

    ndims(a)


.. code-block:: julia

    a = [10 20; 30 40]  # 2 x 2


You might then assume that ``a = [10; 20; 30; 40]`` creates a two dimensional column vector but this isn't the case

.. code-block:: julia

    a = [10; 20; 30; 40]


.. code-block:: julia

    ndims(a)


Instead transpose the matrix (or adjoint if complex)

.. code-block:: julia

    a = [10 20 30 40]'


.. code-block:: julia

    ndims(a)


Array Indexing
-----------------

We've already seen the basics of array indexing

.. code-block:: julia

    a = [10 20 30 40]
    a[end-1]


.. code-block:: julia

    a[1:3]


For 2D arrays the index syntax is straightforward

.. code-block:: julia

    a = randn(2, 2)
    a[1, 1]


.. code-block:: julia

    a[1, :]  # first row


.. code-block:: julia

    a[:, 1]  # first column


Booleans can be used to extract elements

.. code-block:: julia

    a = randn(2, 2)


.. code-block:: julia

    b = [true false; false true]


.. code-block:: julia

    a[b]


This is useful for conditional extraction, as we'll see below

An aside: some or all elements of an array can be set equal to one number using slice notation


.. code-block:: julia

    a = zeros(4)


.. code-block:: julia

    a[2:end] .= 42


.. code-block:: julia

    a

Views and Slices
-----------------------------

Using the ``:`` notation provides a slice of an array, copying the sub-array to a new array with a similar type

.. code-block:: julia

    a = [1 2; 3 4]
    b = a[:, 2]
    @show b
    a[:, 2] = [4, 5] # modify a
    @show a
    @show b;

A **view** on the other hand does not copy the value

.. code-block:: julia

    a = [1 2; 3 4]
    @views b = a[:, 2]
    @show b
    a[:, 2] = [4, 5]
    @show a
    @show b;

Note that the only difference is the ``@views`` macro, which will replace any slices with views in the expression

An alternative is to call the ``view`` function directly -- though it is generally discouraged since it is a step away from the math

.. code-block:: julia

    @views b = a[:, 2]
    view(a, :, 2) == b

As with most programming in Julia, it is best to avoid prematurely assuming that ``@views`` will have a significant impact on performance, and stress code clarity above all else

Another important lesson about ``@views`` is that they **are not** normal, dense arrays

.. code-block:: julia

    a = [1 2; 3 4]
    b_slice = a[:, 2]
    @show typeof(b_slice)
    @show typeof(a)
    @views b = a[:, 2]
    @show typeof(b);

The type of ``b`` is a good example of how types are not as they may seem

Similarly

.. code-block:: julia

    a = [1 2; 3 4]
    b = a'   # transpose
    typeof(b)


To copy into a dense array

.. code-block:: julia

    a = [1 2; 3 4]
    b = a' # transpose
    c = Matrix(b)  # convert to matrix
    d = collect(b) # also `collect` works on any iterable
    c == d

Special Matrices
-----------------

As we saw with ``transpose``, sometimes types that look like matrices are not stored as a dense array

As an example, consider creating a diagonal matrix

.. code-block:: julia

    d = [1.0, 2.0]
    a = Diagonal(d)

As you can see, the type is ``2×2 Diagonal{Float64,Array{Float64,1}}``, which is not a 2-dimensional array

The reasons for this are both efficiency in storage, as well as efficiency in arithmetic and matrix operations

In every important sense, matrix types such as ``Diagonal`` are just as much a "matrix" as the dense matrices we have using (see the :doc:`introduction to types lecture <../getting_started_julia/introduction_to_types>` for more)

.. code-block:: julia

    @show 2a
    b = rand(2,2)
    @show b * a;

Another example is in the construction of an identity matrix, where a naive implementation is

.. code-block:: julia

    b = [1.0 2.0; 3.0 4.0]
    b - Diagonal([1.0, 1.0])  # poor style, inefficient code

Whereas you should instead use

.. code-block:: julia

    b = [1.0 2.0; 3.0 4.0]
    b - I  # good style, and note the lack of dimensions of I

While the implementation of ``I`` is a little abstract to go into at this point, a hint is:

.. code-block:: julia

    typeof(I)

This is a ``UniformScaling`` type rather than an identity matrix, making it much more powerful and general

Assignment and Passing Arrays
------------------------------

As discussed above, in Julia, the left hand side of an assignment is a "binding" or a label to a value

.. code-block:: julia

    x = [1 2 3]
    y = x  # name `y` binds to whatever value `x` bound to

The consequence of this, is that you can re-bind that name

.. code-block:: julia

    x = [1 2 3]
    y = x        # name `y` binds to whatever `x` bound to
    z = [2 3 4]
    y = z        # only changes name binding, not value!
    @show (x, y, z);

What this means is that if ``a`` is an array and we set ``b = a`` then ``a`` and ``b`` point to exactly the same data

In the above, suppose you had meant to change the value of ``x`` to the values of ``y``, you need to assign the values rather than the name

.. code-block:: julia

    x = [1 2 3]
    y = x       # name `y` binds to whatever `x` bound to
    z = [2 3 4]
    y .= z      # now dispatches the assignment of each element
    @show (x, y, z);

Alternatively, you could have used ``y[:] = z``

This applies to in-place functions as well

First, define a simple function for a linear map

.. code-block:: julia

    function f(x)
        return [1 2; 3 4] * x  # matrix * column vector
    end

    val = [1, 2]
    f(val)

In general, these "out-of-place" functions are preferred to "in-place" functions, which modify the arguments

.. code-block:: julia

    function f(x)
        return [1 2; 3 4] * x # matrix * column vector
    end

    val = [1, 2]
    y = similar(val)

    function f!(out, x)
        out .= [1 2; 3 4] * x
    end

    f!(y, val)
    y

This demonstrates a key convention in Julia: functions which modify any of the arguments have the name ending with ``!`` (e.g. ``push!``)

We can also see a common mistake, where instead of modifying the arguments, the name binding is swapped

.. code-block:: julia

    function f(x)
        return [1 2; 3 4] * x  # matrix * column vector
    end

    val = [1, 2]
    y = similar(val)

    function f!(out, x)
        out = [1 2; 3 4] * x   # MISTAKE! Should be .= or [:]
    end
    f!(y, val)
    y

The frequency of making this mistake is one of the reasons to avoid in-place functions, unless proven to be necessary by benchmarking

In-place and Immutable Types
------------------------------

Note that scalars are always immutable, such that

.. code-block:: julia

    y = [1 2]
    y .-= 2    # y .= y .- 2, no problem

    x = 5
    # x .-= 2  # Fails!
    x = x - 2  # subtle difference - creates a new value and rebinds the variable


In particular, there is no way to pass any immutable into a function and have it modified

.. code-block:: julia

    x = 2

    function f(x)
        x = 3     # MISTAKE! does not modify x, creates a new value!
    end

    f(x)          # cannot modify immutables in place
    @show x;

This is also true for other immutable types such as tuples, as well as some vector types

.. code-block:: julia

    using StaticArrays
    xdynamic = [1, 2]
    xstatic = @SVector [1, 2]  # turns it into a highly optimized static vector

    f(x) = 2x
    @show f(xdynamic)
    @show f(xstatic)

    # inplace version
    function g(x)
        x .= 2x
        return "Success!"
    end
    @show xdynamic
    @show g(xdynamic)
    @show xdynamic;

    # g(xstatic) # fails, static vectors are immutable

Operations on Arrays
================================

Array Methods
------------------

Julia provides standard functions for acting on arrays, some of which we've
already seen

.. code-block:: julia

    a = [-1, 0, 1]

    @show length(a)
    @show sum(a)
    @show mean(a)
    @show std(a)      # standard deviation
    @show var(a)      # variance
    @show maximum(a)
    @show minimum(a)
    @show extrema(a)  # (mimimum(a), maximum(a))


To sort an array

.. code-block:: julia

    b = sort(a, rev = true)  # returns new array, original not modified


.. code-block:: julia

    b = sort!(a, rev = true)  # returns *modified original* array


.. code-block:: julia

    b == a  # tests if have the same values


.. code-block:: julia

    b === a  # tests if arrays are identical (i.e share same memory)


Matrix Algebra
---------------------

For two dimensional arrays, ``*`` means matrix multiplication

.. code-block:: julia

    a = ones(1, 2)


.. code-block:: julia

    b = ones(2, 2)


.. code-block:: julia

    a * b


.. code-block:: julia

    b * a'


To solve the linear system :math:`A X = B` for :math:`X` use ``A \ B``

.. code-block:: julia

    A = [1 2; 2 3]


.. code-block:: julia

    B = ones(2, 2)


.. code-block:: julia

    A \ B


.. code-block:: julia

    inv(A) * B


Although the last two operations give the same result, the first one is numerically more stable and should be preferred in most cases

Multiplying two **one** dimensional vectors gives an error -- which is reasonable since the meaning is ambiguous

More precisely, the error is that there isn't an implementation of ``*`` for two one dimensional vectors

The output explains this, and lists some other methods of ``*`` which Julia thinks are close to what we want

.. code-block:: julia
    :class: skip-test

    ones(2) * ones(2)


If you want an inner product in this setting use ``dot()`` or the unicode ``\cdot<TAB>``

.. code-block:: julia

    dot(ones(2), ones(2))


Matrix multiplication using one dimensional vectors is a bit inconsistent --
pre-multiplication by the matrix is OK, but post-multiplication gives an error


.. code-block:: julia

    b = ones(2, 2)


.. code-block:: julia

    b * ones(2)


.. code-block:: julia
    :class: skip-test

    ones(2) * b


Elementwise Operations
------------------------

Algebraic Operations
^^^^^^^^^^^^^^^^^^^^^^^^

Suppose that we wish to multiply every element of matrix ``A`` with the corresponding element of matrix ``B``

In that case we need to replace ``*`` (matrix multiplication) with ``.*`` (elementwise multiplication)

For example, compare

.. code-block:: julia

    ones(2, 2) * ones(2, 2)   # matrix multiplication


.. code-block:: julia

    ones(2, 2) .* ones(2, 2)   # element by element multiplication


This is a general principle: ``.x`` means apply operator ``x`` elementwise


.. code-block:: julia

    A = -ones(2, 2)


.. code-block:: julia

    A.^2  # square every element

However in practice some operations are mathematically valid without broadcasting, and hence the ``.`` can be omitted

.. code-block:: julia

    ones(2, 2) + ones(2, 2)  # same as ones(2, 2) .+ ones(2, 2)


Scalar multiplication is similar

.. code-block:: julia

    A = ones(2, 2)


.. code-block:: julia

    2 * A  # same as 2 .* A


In fact you can omit the ``*`` altogether and just write ``2A``

Unlike MATLAB and other languages, scalar addition requires the ``.+`` in order to correctly broadcast

.. code-block:: julia

    x = [1, 2]
    x .+ 1     # not x + 1
    x .- 1     # not x - 1


Elementwise Comparisons
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elementwise comparisons also use the ``.x`` style notation

.. code-block:: julia

    a = [10, 20, 30]


.. code-block:: julia

    b = [-100, 0, 100]


.. code-block:: julia

    b .> a


.. code-block:: julia

    a .== b


We can also do comparisons against scalars with parallel syntax

.. code-block:: julia

    b


.. code-block:: julia

    b .> 1



This is particularly useful for *conditional extraction* -- extracting the elements of an array that satisfy a condition

.. code-block:: julia

    a = randn(4)


.. code-block:: julia

    a .< 0


.. code-block:: julia

    a[a .< 0]


Changing Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^

The primary function for changing the dimensions of an array is ``reshape()``


.. code-block:: julia

    a = [10, 20, 30, 40]


.. code-block:: julia

    b = reshape(a, 2, 2)


.. code-block:: julia

    b


Notice that this function returns a view on the existing array

This means that changing the data in the new array will modify the data in the
old one

.. code-block:: julia

    b[1, 1] = 100  # continuing the previous example


.. code-block:: julia

    b


.. code-block:: julia

    a


To collapse an array along one dimension you can use ``dropdims()``

.. code-block:: julia

    a = [1 2 3 4]  # two dimensional


.. code-block:: julia

    dropdims(a, dims = 1)


The return value is an array with the specified dimension "flattened"

Broadcasting Functions
--------------------------

Julia provides standard mathematical functions such as ``log``, ``exp``, ``sin``, etc.

.. code-block:: julia

    log(1.0)


By default, these functions act *elementwise* on arrays

.. code-block:: julia

    log.(1:4)

Note that we can get the same result as with a comprehension or more explicit loop


.. code-block:: julia

    [ log(x) for x in 1:4 ]


.. ACTUALLY, kind of the opposite, as in most languages.  "for" loops typically fastest when done correctly.  In Julia loops are typically fast and hence the need for vectorized functions is less intense than for some other high level languages

Nonetheless the syntax is convenient

Linear Algebra
-------------------

(`See linear algebra documentation <https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/>`_)


Julia provides some a great deal of additional functionality related to linear operations


.. code-block:: julia

    A = [1 2; 3 4]


.. code-block:: julia

    det(A)


.. code-block:: julia

    tr(A)


.. code-block:: julia

    eigvals(A)


.. code-block:: julia

    rank(A)


Ranges
==================

As with many other types, a ``Range`` can act as a vector

.. code-block:: julia

    a = 10:12        # a range, equivalent to 10:1:12
    @show Vector(a)  # can convert, but shouldn't

    b = Diagonal([1.0, 2.0, 3.0])
    b * a .- [1.0; 2.0; 3.0]

Ranges can also be created with floating point numbers using the same notation

.. code-block:: julia

    a = 0.0:0.1:1.0  # 0.0, 0.1, 0.2, ... 1.0

But care should be taken if the terminal node is not a multiple of the set sizes

.. code-block:: julia

    maxval = 1.0
    minval = 0.0
    stepsize = 0.15
    a = minval:stepsize:maxval # 0.0, 0.15, 0.3, ...
    maximum(a) == maxval

To evenly space points where the maximum value is important, i.e., ``linspace`` in other languages

.. code-block:: julia

    maxval = 1.0
    minval = 0.0
    numpoints = 10
    a = range(minval, maxval, length=numpoints)
    # or range(minval, stop=maxval, length=numpoints)

    maximum(a) == maxval

Tuples and Named Tuples
=========================

(`See tuples <https://docs.julialang.org/en/v1/manual/functions/#Tuples-1>`_ and `named tuples documentation <https://docs.julialang.org/en/v1/manual/functions/#Named-Tuples-1>`_)

We were introduced to tuples earlier, which provide high-performance immutable sets of distinct types

.. code-block:: julia

    t = (1.0, "test")
    t[1]            # access by index
    a, b = t        # unpack
    # t[1] = 3.0    # would fail as tuples are immutable
    println("a = $a and b = $b")

As well as **named tuples**, which extend tuples with names for each argument

.. code-block:: julia

    t = (val1 = 1.0, val2 = "test")
    t.val1      # access by index
    # a, b = t  # bad style, better to unpack by name with @unpack
    println("val1 = $(t.val1) and val1 = $(t.val1)") # access by name

While immutable, it is possible to manipulate tuples and generate new ones

.. code-block:: julia

    t2 = (val3 = 4, val4 = "test!!")
    t3 = merge(t, t2)  # new tuple


Named tuples are a convenient and high-performance way to manage and unpack sets of parameters


.. code-block:: julia

    function f(parameters)
        α, β = parameters.α, parameters.β  # poor style, error prone if adding parameters
        return α + β
    end

    parameters = (α = 0.1, β = 0.2)
    f(parameters)


This functionality is aided by the ``Parameters.jl`` package and the ``@unpack`` macro

.. code-block:: julia

    using Parameters

    function f(parameters)
        @unpack α, β = parameters  # good style, less sensitive to errors
        return α + β
    end

    parameters = (α = 0.1, β = 0.2)
    f(parameters)

In order to manage default values, use the ``@with_kw`` macro

.. code-block:: julia

    using Parameters
    paramgen = @with_kw (α = 0.1, β = 0.2)  # create named tuples with defaults

    # creates named tuples, replacing defaults
    @show paramgen()  # calling without arguments gives all defaults
    @show paramgen(α = 0.2)
    @show paramgen(α = 0.2, β = 0.5);

An alternative approach, defining a new type using ``struct`` tends to be more prone to accidental misuse, and leads to a great deal of boilerplate code

For that, and other reasons of generality, we will use named tuples for collections of parameters where possible

Nothing, Missing, and Unions
==============================

Sometimes a variable, return type from a function, or value in an array needs to represent the absence of a value rather than a particular value

There are two distinct use cases for this

#. ``nothing`` ("software engineers null"): used where no value makes sense in a particular context due to a failure in the code, a function parameter not passed in, etc.
#. ``missing`` ("data scientists null"): used when a value would make conceptual sense, but it isn't available

.. _error_handling:

Nothing and Basic Error Handling
----------------------------------

The value ``nothing`` is a single value of type ``Nothing``

.. code-block:: julia

    typeof(nothing)


An example of a reasonable use of ``nothing`` is if you need to have a variable defined in an outer scope, which may or may not be set in an inner one

.. code-block:: julia

    function f(y)
        x = nothing
        if y > 0.0
            # calculations to set `x`
            x = y
        end

        # later, can check `x`
        if isnothing(x)
            println("x was not set")
        else
            println("x = $x")
        end
        x
    end

    @show f(1.0)
    @show f(-1.0);

While in general you want to keep a variable name bound to a single type in Julia, this is a notable exception

Similarly, if needed, you can return a ``nothing`` from a function to indicate that it did not calculate as expected

.. code-block:: julia

    function f(x)
        if x > 0.0
            return sqrt(x)
        else
            return nothing
        end
    end
    x1 = 1.0
    x2 = -1.0
    y1 = f(x1)
    y2 = f(x2)

    # check results with isnothing
    if isnothing(y1)
        println("f($x2) successful")
    else
        println("f($x2) failed");
    end

As an aside, an equivalent way to write the above function is to use the
`ternary operator <https://docs.julialang.org/en/v1/manual/control-flow/index.html#man-conditional-evaluation-1>`_,
which gives a compact if/then/else structure

.. code-block:: julia

    function f(x)
        x > 0.0 ? sqrt(x) : nothing  # the "a ? b : c" pattern is the ternary
    end

    f(1.0)

We will sometimes use this form when it makes the code more clear (and it will occasionally make the code higher performance)

Regardless of how ``f(x)`` is written,  the return type is an example of a union, where the result could be one of an explicit set of types

In this particular case, the compiler would deduce that the type would be a ``Union{Nothing,Float64}`` -- that is, it returns either a floating point or a ``nothing``

You will see this type directly if you use an array containing both types

.. code-block:: julia

    x = [1.0, nothing]

When considering error handling, whether you want a function to return ``nothing`` or simply fail depends on whether the code calling ``f(x)`` is carefully checking the results

For example, if you were calling on an array of parameters where a priori you were not sure which ones will succeed, then

.. code-block:: julia

    x = [0.1, -1.0, 2.0, -2.0]
    y = f.(x)

    # presumably check `y`

On the other hand, if the parameter passed is invalid and you would prefer not to handle a graceful failure, then using an assertion is more appropriate

.. code-block:: julia

    function f(x)
        @assert x > 0.0
        sqrt(x)
    end

    f(1.0)

Finally, ``nothing`` is a good way to indicate an optional parameter in a function

.. code-block:: julia

    function f(x; z = nothing)

        if isnothing(z)
            println("No z given with $x")
        else
            println("z = $z given with $x")
        end
    end

    f(1.0)
    f(1.0, z=3.0)

An alternative to ``nothing``, which can be useful and sometimes higher performance,
is to use ``NaN`` to signal that a value is invalid returning from a function

.. code-block:: julia

    function f(x)
        if x > 0.0
            return x
        else
            return NaN
        end
    end

    f(0.1)
    f(-1.0)

    @show typeof(f(-1.0))
    @show f(-1.0) == NaN  # note, this fails!
    @show isnan(f(-1.0))  # check with this

Note that in this case, the return type is ``Float64`` regardless of the input for ``Float64`` input

Keep in mind, though, that this only works if the return type of a function is ``Float64``


Exceptions
----------------------------------

(See `exceptions documentation <https://docs.julialang.org/en/v1/manual/control-flow/index.html#Exception-Handling-1>`_)

While returning a ``nothing`` can be a good way to deal with functions which may or may not return values, a more robust error handling method is to use exceptions

Unless you are writing a package, you will rarely want to define and throw your own exceptions, but will need to deal with them from other libraries

The key distinction for when to use an exceptions vs. return a ``nothing`` is whether an error is unexpected rather than a normal path of execution

An example of an exception is a ``DomainError``, which signifies that a value passed to a function is invalid

.. code-block:: julia

    # throws exception, turned off to prevent breaking notebook
    # sqrt(-1.0)

    # to see the error
    try sqrt(-1.0); catch err; err end  # catches the exception and prints it


Another example you will see is when the compiler cannot convert between types

.. code-block:: julia

    # throws exception, turned off to prevent breaking notebook
    # convert(Int64, 3.12)

    # to see the error
    try convert(Int64, 3.12); catch err; err end  # catches the exception and prints it.

If these exceptions are generated from unexpected cases in your code, it may be appropriate simply let them occur and ensure you can read the error

Occasionally you will want to catch these errors and try to recover, as we did above in the ``try`` block

.. code-block:: julia

    function f(x)
        try
            sqrt(x)
        catch err                # enters if exception thrown
            sqrt(complex(x, 0))  # convert to complex number
        end
    end

    f(0.0)
    f(-1.0)

.. _missing:

Missing
----------------------------------

(see `"missing" documentation <https://docs.julialang.org/en/v1/manual/missing/>`_)

The value ``missing`` of type ``Missing`` is used to represent missing value in a statistical sense

For example, if you loaded data from a panel, and gaps existed

.. code-block:: julia

    x = [3.0, missing, 5.0, missing, missing]

A key feature of ``missing`` is that it propagates through other function calls - unlike ``nothing``

.. code-block:: julia

    f(x) = x^2

    @show missing + 1.0
    @show missing * 2
    @show missing * "test"
    @show f(missing);      # even user-defined functions
    @show mean(x);

The purpose of this is to ensure that failures do not silently fail and provide meaningless numerical results

This even applies for the comparison of values, which

.. code-block:: julia

    x = missing

    @show x == missing
    @show x === missing  # an exception
    @show ismissing(x);

Where ``ismissing`` is the canonical way to test the value

In the case where you would like to calculate a value without the missing values, you can use ``skipmissing``

.. code-block:: julia

    x = [1.0, missing, 2.0, missing, missing, 5.0]

    @show mean(x)
    @show mean(skipmissing(x))
    @show coalesce.(x, 0.0);  # replace missing with 0.0;

As ``missing`` is similar to R's ``NA`` type, we will see more of ``missing`` when we cover ``DataFrames``

Exercises
=============


.. _np_ex1:

Exercise 1
----------------

This exercise uses matrix operations that arise in certain problems,
including when dealing with linear stochastic difference equations

If you aren't familiar with all the terminology don't be concerned -- you can
skim read the background discussion and focus purely on the matrix exercise

With that said, consider the stochastic difference equation

.. math::
    :label: ja_sde

    X_{t+1} = A X_t + b + \Sigma W_{t+1}


Here

* :math:`X_t, b` and :math:`X_{t+1}` are :math:`n \times 1`

* :math:`A` is :math:`n \times n`

* :math:`\Sigma` is :math:`n \times k`

* :math:`W_t` is :math:`k \times 1` and :math:`\{W_t\}` is iid with zero mean and variance-covariance matrix equal to the identity matrix

Let :math:`S_t` denote the :math:`n \times n` variance-covariance matrix of :math:`X_t`

Using the rules for computing variances in matrix expressions, it can be shown from :eq:`ja_sde` that :math:`\{S_t\}` obeys

.. math::
    :label: ja_sde_v

    S_{t+1} = A S_t A' + \Sigma \Sigma'


It can be shown that, provided all eigenvalues of :math:`A` lie within the unit circle, the sequence :math:`\{S_t\}` converges to a unique limit :math:`S`

This is the **unconditional variance** or **asymptotic variance** of the stochastic difference equation

As an exercise, try writing a simple function that solves for the limit :math:`S` by iterating on :eq:`ja_sde_v` given :math:`A` and :math:`\Sigma`

To test your solution, observe that the limit :math:`S` is a solution to the matrix equation

.. math::
    :label: ja_dle

    S = A S A' + Q
    \quad \text{where} \quad Q := \Sigma \Sigma'


This kind of equation is known as a **discrete time Lyapunov equation**

The `QuantEcon package <http://quantecon.org/julia_index.html>`_
provides a function called ``solve_discrete_lyapunov`` that implements a fast
"doubling" algorithm to solve this equation

Test your iterative method against ``solve_discrete_lyapunov`` using matrices

.. math::

    A =
    \begin{bmatrix}
        0.8 & -0.2  \\
        -0.1 & 0.7
    \end{bmatrix}
    \qquad
    \Sigma =
    \begin{bmatrix}
        0.5 & 0.4 \\
        0.4 & 0.6
    \end{bmatrix}

Exercise 2
----------------

Take a stochastic process for :math:`\{y_t\}_{t=0}^T`

.. math::
    y_{t+1} = \gamma + \theta y_t + \sigma w_{t+1}

where

* :math:`w_{t+1}` is distributed ``Normal(0,1)``
* :math:`\gamma=1, \sigma=1, y_0 = 0`
* :math:`\theta \in \Theta \equiv \{0.8, 0.9, 0.98\}`

Given these parameters

* Simulate a single :math:`y_t` series for each :math:`\theta \in \Theta`
  for :math:`T = 150`.  Feel free to experiment with different :math:`T`
* Overlay plots of the rolling mean of the process for each :math:`\theta \in \Theta`,
  i.e. for each :math:`1 \leq \tau \leq T` plot

.. math::

    \frac{1}{\tau}\sum_{t=1}^{\tau}y_T

* Simulate :math:`N=200` paths of the stochastic process above to the :math:`T`,
  for each :math:`\theta \in \Theta`, where we refer to an element of a particular
  simulation as :math:`y^n_t`
* Overlay plots a histogram of the stationary distribution of the final
  :math:`y^n_T` for each :math:`\theta \in \Theta`.  Hint: pass ``alpha``
  to a plot to make it transparent (e.g. ``histogram(vals, alpha = 0.5)``) or
  use ``stephist(vals)`` to show just the step function for the histogram
* Numerically find the mean and variance of this as an ensemble average, i.e.
  :math:`\sum_{n=1}^N\frac{y^n_T}{N}` and
  :math:`\sum_{n=1}^N\frac{(y_T^n)^2}{N} -\left(\sum_{n=1}^N\frac{y^n_T}{N}\right)^2`

Later, we will interpret some of these in :doc:`this lecture <../tools_and_techniques/lln_clt>`

Exercise 3
--------------

Let the data generating process for a variable be

.. math::

    y = a x_1 + b x_1^2 + c x_2 + d + \sigma w

where :math:`y, x_1, x_2` are scalar observables, :math:`a,b,c,d` are parameters to estimate, and :math:`w` are iid normal with mean 0 and variance 1

First, let's simulate data we can use to estimate the parameters

* Draw :math:`N=50` values for :math:`x_1, x_2` from iid normal distributions

Then, simulate with different :math:`w`
* Draw a :math:`w` vector for the ``N`` values and then ``y`` from this simulated data if the parameters were :math:`a = 0.1, b = 0.2 c = 0.5, d = 1.0, \sigma = 0.1`
* Repeat that so you have ``M = 20`` different simulations of the ``y`` for the ``N`` values

Finally, calculate order least squares manually (i.e., put the observables
into matrices and vectors, and directly use the equations for
`OLS <https://en.wikipedia.org/wiki/Ordinary_least_squares>`_ rather than a package)

* For each of the ``M=20`` simulations, calculate the OLS estimates for :math:`a, b, c, d, \sigma`
* Plot a histogram of these estimates for each variable


Exercise 4
--------------

Redo Exercise 1 using the ``fixedpoint`` function from ``NLsolve`` :doc:`this lecture <julia_by_example>`

Compare the number of iterations of the NLsolve's Anderson Acceleration to the handcoded iteration used in Exercise 1


Solutions
==================

Exercise 1
----------

Here's the iterative approach

.. code-block:: julia

    function compute_asymptotic_var(A, Σ;
                                    S0 = Σ * Σ',
                                    tolerance = 1e-6,
                                    maxiter = 500)
        V = Σ * Σ'
        S = S0
        err = tolerance + 1
        i = 1
        while err > tolerance && i ≤ maxiter
            next_S = A * S * A' + V
            err = norm(S - next_S)
            S = next_S
            i += 1
        end
        return S
    end


.. code-block:: julia

    A = [0.8  -0.2;
         -0.1  0.7]

    Σ = [0.5 0.4;
         0.4 0.6]


Note that all eigenvalues of :math:`A` lie inside the unit disc


.. code-block:: julia

    maximum(abs, eigvals(A))

Let's compute the asymptotic variance


.. code-block:: julia

    our_solution = compute_asymptotic_var(A, Σ)


Now let's do the same thing using QuantEcon's ``solve_discrete_lyapunov()`` function and check we get the same result

.. code-block:: julia
    :class: hide-output

    using QuantEcon

.. code-block:: julia

    norm(our_solution - solve_discrete_lyapunov(A, Σ * Σ'))
