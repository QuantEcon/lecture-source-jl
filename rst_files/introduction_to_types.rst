.. _introduction_to_types:

.. include:: /_static/includes/lecture_howto_jl.raw

**********************************************
Introduction to Types and Generic Programming
**********************************************

.. contents:: :depth: 2

Overview
============================

In Julia, arrays and tuples are the most important data type sfor working with numerical data

In this lecture we give more details on

* declaring types

* abstract types

* motivation for generic programming

Finding and Interpreting Types
================================

Finding The Type
--------------------------------

Activate the project environment, ensuring that ``Project.toml`` and ``Manifest.toml`` are in the same location as your notebook

.. code-block:: julia

    using Pkg; Pkg.activate(@__DIR__); #activate environment in the notebook's location
    using LinearAlgebra, Statistics

As we have seen in the previous lectures, in Julia all values have a type, which can be queried using the ``typeof`` function

.. code-block:: julia

    @show typeof(1)
    @show typeof(1.0);

The harcoded values ``1`` and ``1.0`` are called literals in a programming language, and the compiler will deduce their types

The above types will be ``Int64`` and ``Float64`` respectively

You can also query the type of a variable which may 

.. code-block:: julia

    x = 1
    typeof(x)

Where the name ``x`` refers to the value ``1``, created as a literal

The next two types use curly bracket notation to express the fact that they are *parametric*

.. code-block:: julia

    @show typeof(1.0 + 1im)
    @show typeof(ones(2,2);

We will learn more details about  :doc:`generic programming <generic_programming>` later, but the key is to interpret the curly brackets as swappable parameters for a given type

For example, ``Array{Float64, 2}`` can be read as

#. ``Array`` is a parametric type representing a dense array, where the first parameter is the type stored, and the 2nd is the number of dimensions
#. ``Float64`` is a concrete type declaring that the data stored will be a particular size of floating point
#. ``2`` is the number of dimensions of that array

A concrete type is one which where values can be created by the compiler

Values of a **parametric type** cannot be concretely constructed unless all of the parameters are given (themselves with concrete types)

In the case of ``Complex{Float64}``
 
#. ``Complex`` is an abstract complex number type
#. ``Float64`` is a concrete type declaring what the type of the real and imaginary parts of the value should store

We will see later that both the ``Array``and ``Complex`` require that any type it uses for storage is a ``Real``--which ``Float64`` fulfills

Another type to consider is the ``Tuple`` and ``Named Tuple``

.. code-block:: julia

    x = (1, 2.0, "test")
    @show typeof(x)

In that case, the ``Tuple`` is the parametric type, and the 3 parameters are a list of the types of each value stored in ``x``

For a named tuple

.. code-block:: julia

    x = (a = 1, b = 2.0, c = "test")
    @show typeof(x)

The parametric ``NamedTuple`` type contains 2 parameters: first a list of names for each field of the tuple, and second the underlying ``Tuple`` type to store the values

Anytime a value is prefixed by a colon, as in the ``:a`` above, the type is ``Symbol``--a special kind of string used to make the code general and high-performance

.. code-block:: julia

    typeof(:a)


See `julia documentation <https://docs.julialang.org/en/v1/manual/types/#Parametric-Types-1>`_ for more on parametric types

**Remark:** Note that, by convention, type names use CamelCase ---  ``FloatingPoint``, ``Array``, ``AbstractArray``, etc.    

Variables, Types, and Values
--------------------------------

Since variables and functions are denoted in lower case, this can be used to easily identify types when reading code and output


After assigning a variable name to an value, we can query the type of the
value via the name

.. code-block:: julia

    x = 42
    @show typeof(x);

The type resides with the value itself, not with the name ``x``

Thus, ``x`` is just a symbol bound to an value of type ``Int64``

Indeed, we can *rebind* the symbol ``x`` to any other value, of the same type or otherwise

.. code-block:: julia

    x = 42.0


Now ``x`` "points to" another value, of type ``Float64``

.. code-block:: julia

    typeof(x)

Introduction to Types
======================

We will discuss this in detail in :doc:`this lecture <generic_programming>`, but much of its performance gains and generality of notation comes from Julia's type system

For example, as we have seen

.. code-block:: julia

    x1 = [1, 2, 3]
    x2 = [1.0, 2.0, 3.0]
    @show typeof(x1)
    @show typeof(x2)

These return ``Array{Int64,1}`` and ``Array{Float64,1}`` respectively, which the compiler is able to infer from the right hand side of the expressions

Given the information on the type, the compiler can work through the sequence of expressions to infer other types

.. code-block:: julia

    # define some function
    f(y) = 2y

    # call with an integer array
    x = [1, 2, 3]
    z = f(x) # compiler deduces type

Good Practices for Functions and Variables
--------------------------------------------

In order to keep many of the benefits of Julia, you will sometimes want to help the compiler ensure that it can always deduce a single type from any function or expression

As an example of bad practice, is to use an array to hold unrelated types

.. code-block:: julia

    x = [1.0, "test", 1] # typically poor style

The type of this is ``Array{Any,1}``, where the ``Any`` means the compiler has determined that any valid Julia types can be added to the array

While occasionally useful, this is to be avoided whenever possible in performance sensitive code

The other place this can come up is in the declaration of functions,

As an example, consider a function which returns different types depending on the arguments

.. code-block:: julia

    function f(x)
        if x > 0
            return 1.0
        else 
            return 0 # Probably meant 0.0
        end
    end
    @show f(1)
    @show f(-1)

The issue here is relatively subtle:  ``1.0`` is a floating point, while ``0`` is an integer

Consequently, given the type of ``x``, the compiler cannot in general determine what type the function will return

This issue, called **type stability** is at the heart of most Julia performance considerations

Luckily, the practice of trying to ensure that functions return the same types is also the most consistent with simple, clear code


Manually Declaring Types
-------------------------

While we keep talking about types, you will notice that we have never declared any types in the underlying code

This is intentional for exposition and "user" code of packages, rather than the writing of those packages themselves

It is also in contrast to some of the sample code you will see

To give an example of the declaration of types, the following are equivalent

.. code-block:: julia

    function f(x, A)
        b = [5.0; 6.0]
        return A * x .+ b
    end
    val = f([0.1, 2.0], [1.0 2.0; 3.0 4.0])

.. code-block:: julia

    function f2(x::Vector{Float64}, A::Matrix{Float64})::Vector{Float64} # argument and return types
        b::Vector{Float64} = [5.0; 6.0]
        return A * x .+ b
    end
    val = f2([0.1; 2.0], [1.0 2.0; 3.0 4.0])

While declaring the types may be verbose, would it ever generate faster code?

The answer is: almost never

Furthermore, it can lead to confusion and inefficiencies since many things that behave like vectors and matrices are not ``Matrix{Float64}`` and ``Vector{Float64}``

To see a few examples where the first works and the second fails

.. code-block:: julia

    @show f([0.1; 2.0], [1 2; 3 4])
    @show f([0.1; 2.0], Diagonal([1.0, 2.0]))

    #f2([0.1; 2.0], [1 2; 3 4]) # not a Float64
    #f2([0.1; 2.0], Diagonal([1.0, 2.0])) # not a Matrix{Float64}


Exercises
=============
