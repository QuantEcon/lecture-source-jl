.. _julia_arrays:

.. include:: /_static/includes/lecture_howto_jl.raw

*********************************
Vectors, Arrays and Matrices
*********************************

.. contents:: :depth: 2

.. epigraph::

    "Let's be clear: the work of science has nothing whatever to do with consensus.
    Consensus is the business of politics. Science, on the contrary, requires only
    one investigator who happens to be right, which means that he or she has
    results that are verifiable by reference to the real world. In science
    consensus is irrelevant. What is relevant is reproducible results." -- Michael Crichton

Overview
============================

In Julia, arrays are the most important data type for working with collections of numerical data

In this lecture we give more details on

* creating and manipulating Julia arrays

* fundamental array processing operations

* basic matrix algebra




.. _julia_array:

Array Basics
================


Shape and Dimension
----------------------


We've already seen some Julia arrays in action

.. code-block:: julia

    a = [10, 20, 30]


.. code-block:: julia

    a = ["foo", "bar", 10]
     


The REPL tells us that the arrays are of types ``Array{Int64,1}`` and ``Array{Any,1}`` respectively

Here ``Int64`` and ``Any`` are types for the elements inferred by the compiler

We'll talk more about types later on

The ``1`` in ``Array{Int64,1}`` and ``Array{Any,1}`` indicates that the array is 
one dimensional

This is the default for many Julia functions that create arrays



.. code-block:: julia

    typeof(randn(100))





To say that an array is one dimensional is to say that it is flat --- neither a row nor a column vector

We can also confirm that ``a`` is flat using the ``size()`` or ``ndims()``
functions

.. code-block:: julia

    size(a)






.. code-block:: julia

    ndims(a)






The syntax ``(3,)`` displays a tuple containing one element --- the size along the one dimension that exists

Here are some functions that create two-dimensional arrays

.. code-block:: julia

    eye(3)







.. code-block:: julia

    diagm([2, 4])








.. code-block:: julia

    size(eye(3))







Array vs Vector vs Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Julia, in addition to arrays you will see the types ``Vector`` and ``Matrix``

However, these are just aliases for one- and two-dimensional arrays
respectively


.. code-block:: julia

    Array{Int64, 1} == Vector{Int64}





.. code-block:: julia

    Array{Int64, 2} == Matrix{Int64}



.. code-block:: julia

    Array{Int64, 1} == Matrix{Int64}



.. code-block:: julia

    Array{Int64, 3} == Matrix{Int64}





In particular, a ``Vector`` in Julia is a flat array



Changing Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^

The primary function for changing the dimension of an array is ``reshape()``


.. code-block:: julia

    a = [10, 20, 30, 40]



.. code-block:: julia

    b = reshape(a, 2, 2)



.. code-block:: julia

    b



Notice that this function returns a "view" on the existing array

This means that changing the data in the new array will modify the data in the
old one:

.. code-block:: julia

    b[1, 1] = 100  # Continuing the previous example



.. code-block:: julia

    b



.. code-block:: julia

    a



To collapse an array along one dimension you can use ``squeeze()``

.. code-block:: julia

    a = [1 2 3 4]  # Two dimensional








.. code-block:: julia

    squeeze(a, 1)




The return value is an array with the specified dimension "flattened"



Why Flat Arrays?
^^^^^^^^^^^^^^^^^^^^^^^^

As we've seen, in Julia we have both

* one-dimensional arrays (i.e., flat arrays)

* arrays of size ``(1, n)`` or ``(n, 1)`` that represent row and column vectors respectively

Why do we need both?

On one hand, dimension matters when we come to matrix algebra

* Multiplying by a row vector is different to multiplication by a column vector

On the other, we use arrays in many settings that don't involve matrix algebra

In such cases, we don't care about the distinction between row and column vectors

This is why many Julia functions return flat arrays by default





.. _creating_arrays:

Creating Arrays
------------------



Functions that Return Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We've already seen some functions for creating arrays

.. code-block:: julia

    eye(2)



.. code-block:: julia

    zeros(3)




You can create an empty array using the ``Array()`` constructor

.. code-block:: julia

    x = Array{Float64}(2, 2)

        

The printed values you see here are just garbage values

(the existing contents of the allocated memory slots being interpreted as 64 bit floats)

Other important functions that return arrays are

.. code-block:: julia

    ones(2, 2)



.. code-block:: julia

    fill("foo", 2, 2)




Manual Array Definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we've seen, you can create one dimensional arrays from manually specified data like so

.. code-block:: julia

    a = [10, 20, 30, 40]



In two dimensions we can proceed as follows

.. code-block:: julia

    a = [10 20 30 40]  # Two dimensional, shape is 1 x n




.. code-block:: julia

    ndims(a)



.. code-block:: julia

    a = [10 20; 30 40]  # 2 x 2



You might then assume that ``a = [10; 20; 30; 40]`` creates a two dimensional column vector but unfortunately this isn't the case

.. code-block:: julia

    a = [10; 20; 30; 40]



.. code-block:: julia

    ndims(a)



Instead transpose the row vector

.. code-block:: julia

    a = [10 20 30 40]'



.. code-block:: julia

    ndims(a)





Array Indexing
-----------------

We've already seen the basics of array indexing

.. code-block:: julia

    a = collect(10:10:40)





.. code-block:: julia

    a[end-1]




.. code-block:: julia

    a[1:3]





For 2D arrays the index syntax is straightforward

.. code-block:: julia

    a = randn(2, 2)





.. code-block:: julia

    a[1, 1]






.. code-block:: julia

    a[1, :]  # First row






.. code-block:: julia

    a[:, 1]  # First column





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

    a = Array{Float64}(4)



.. code-block:: julia

    a[2:end] = 42



.. code-block:: julia

    a
      



Passing Arrays
--------------------

As in Python, all arrays are passed by reference

What this means is that if ``a`` is an array and we set ``b = a`` then ``a`` and ``b`` point to exactly the same data

Hence any change in ``b`` is reflected in ``a``

.. code-block:: julia

    a = ones(3)






.. code-block:: julia

    b = a



.. code-block:: julia

    b[3] = 44



.. code-block:: julia

    a



If you are a MATLAB programmer perhaps you are recoiling in horror at this
idea

But this is actually the more sensible default -- after all, it's very inefficient to copy arrays unnecessarily

If you do need an actual copy in Julia, just use ``copy()``

.. code-block:: julia

    a = ones(3)




.. code-block:: julia

    b = copy(a)



.. code-block:: julia

    b[3] = 44



.. code-block:: julia

    a





Operations on Arrays
================================


Array Methods
------------------

Julia provides standard functions for acting on arrays, some of which we've
already seen

.. code-block:: julia

    a = [-1, 0, 1]







.. code-block:: julia

    length(a)






.. code-block:: julia

    sum(a)






.. code-block:: julia

    mean(a)





.. code-block:: julia

    std(a)





.. code-block:: julia

    var(a)






.. code-block:: julia

    maximum(a)






.. code-block:: julia

    minimum(a)




.. code-block:: julia

    b = sort(a, rev=true)  # Returns new array, original not modified




.. code-block:: julia

    b === a  # === tests if arrays are identical (i.e share same memory)




.. code-block:: julia

    b = sort!(a, rev=true)  # Returns *modified original* array



.. code-block:: julia

    b === a








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






To solve the linear system ``A X = B`` for ``X`` use ``A \ B``

.. code-block:: julia

    A = [1 2; 2 3]








.. code-block:: julia

    B = ones(2, 2)






.. code-block:: julia

    A \ B








.. code-block:: julia

    inv(A) * B






Although the last two operations give the same result, the first one is numerically more stable and should be preferred in most cases

Multiplying two **one** dimensional vectors gives an error --- which is reasonable since the meaning is ambiguous

.. code-block:: julia
    :class: no-execute

    ones(2) * ones(2)



    
If you want an inner product in this setting use ``dot()``


.. code-block:: julia

    dot(ones(2), ones(2))







Matrix multiplication using one dimensional vectors is a bit inconsistent --- pre-multiplication by the matrix is OK, but post-multiplication gives an error


.. code-block:: julia

    b = ones(2, 2)






.. code-block:: julia

    b * ones(2)








.. code-block:: julia
    :class: no-execute

    ones(2) * b





It's probably best to give your vectors dimension before you multiply them against matrices


Elementwise Operations
------------------------



Algebraic Operations
^^^^^^^^^^^^^^^^^^^^^^^^

Suppose that we wish to multiply every element of matrix ``A`` with the corresponding element of matrix ``B``

In that case we need to replace ``*`` (matrix multiplication) with ``.*`` (elementwise multiplication)

For example, compare

.. code-block:: julia

    ones(2, 2) * ones(2, 2)   # Matrix multiplication






.. code-block:: julia

    ones(2, 2) .* ones(2, 2)   # Element by element multiplication







This is a general principle: ``.x`` means apply operator ``x`` elementwise


.. code-block:: julia

    A = -ones(2, 2)







.. code-block:: julia

    A.^2  # Square every element







However in practice some operations are unambiguous and hence the ``.`` can be omitted

.. code-block:: julia

    ones(2, 2) + ones(2, 2)  # Same as ones(2, 2) .+ ones(2, 2)






Scalar multiplication is similar

.. code-block:: julia

    A = ones(2, 2)








.. code-block:: julia

    2 * A  # Same as 2 .* A






In fact you can omit the ``*`` altogether and just write ``2A``


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



This is particularly useful for *conditional extraction* --- extracting the elements of an array that satisfy a condition

.. code-block:: julia

    a = randn(4)




.. code-block:: julia

    a .< 0



.. code-block:: julia

    a[a .< 0]




Vectorized Functions
--------------------------

Julia provides standard mathematical functions such as ``log``, ``exp``, ``sin``, etc.

.. code-block:: julia

    log(1.0)



By default, these functions act *elementwise* on arrays

.. code-block:: julia

    log.(ones(4))




Functions that act elementwise on arrays in this manner are called **vectorized functions**

Note that we can get the same result as with a comprehension or more explicit loop


.. code-block:: julia

    [log(x) for x in ones(4)]




In Julia loops are typically fast and hence the need for vectorized functions is less intense than for some other high level languages

Nonetheless the syntax is convenient



Linear Algebra
=======================


Julia provides some a great deal of additional functionality related to linear operations



.. code-block:: julia

    A = [1 2; 3 4]



.. code-block:: julia

    det(A)



.. code-block:: julia

    trace(A)


.. code-block:: julia

    eigvals(A)
 

.. code-block:: julia

    rank(A)




For more details see the `linear algebra section <https://docs.julialang.org/en/stable/manual/linear-algebra/>`_ of the standard library

Exercises
=============


.. _np_ex1:

Exercise 1
----------------

This exercise is on some matrix operations that arise in certain problems, including when dealing with linear stochastic difference equations

If you aren't familiar with all the terminology don't be concerned --- you can skim read the background discussion and focus purely on the matrix exercise

With that said, consider the stochastic difference equation

.. math::
    :label: ja_sde

    X_{t+1} = A X_t + b + \Sigma W_{t+1}


Here 

* :math:`X_t, b` and :math:`X_{t+1}` ar :math:`n \times 1`

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


Solutions
==================

Exercise 1
----------

Here's the iterative approach

.. code-block:: julia

    function compute_asymptotic_var(A, 
                                    Sigma, 
                                    S0=Sigma * Sigma', 
                                    tolerance=1e-6, 
                                    maxiter=500)
        V = Sigma * Sigma'
        S = S0
        err = tolerance + 1
        i = 1
        while err > tolerance && i <= maxiter
            next_S = A * S * A' + V
            err = norm(S - next_S)
            S = next_S
            i = i + 1
        end
        return S
    end






.. code-block:: julia

    A =     [0.8 -0.2; 
            -0.1 0.7]
    Sigma = [0.5 0.4;
             0.4 0.6]


Note that all eigenvalues of :math:`A` lie inside the unit disc:


.. code-block:: julia

    maximum(abs, eigvals(A))

Let's compute the asymptotic variance:






.. code-block:: julia

    compute_asymptotic_var(A, Sigma)




Now let's do the same thing using QuantEcon's `solve_discrete_lyapunov()` function and check we get the same result


.. code-block:: julia

    using QuantEcon

.. code-block:: julia

    solve_discrete_lyapunov(A, Sigma * Sigma')




