.. _introduction_to_types:

.. include:: /_static/includes/lecture_howto_jl.raw

**********************************************
Introduction to Types and Generic Programming
**********************************************

.. contents:: :depth: 2

Overview
============================

In Julia, arrays and tuples are the most important data type for working with numerical data

In this lecture we give more details on

* declaring types

* abstract types

* motivation for generic programming

Setup
------

.. literalinclude:: /_static/includes/deps.jl


Finding and Interpreting Types
================================

Finding The Type
--------------------------------

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
    @show typeof(ones(2,2));

We will learn more details about :doc:`generic programming <generic_programming>` later, but the key is to interpret the curly brackets as swappable parameters for a given type

For example, ``Array{Float64, 2}`` can be read as

#. ``Array`` is a parametric type representing a dense array, where the first parameter is the type stored, and the 2nd is the number of dimensions
#. ``Float64`` is a concrete type declaring that the data stored will be a particular size of floating point
#. ``2`` is the number of dimensions of that array

A concrete type is one which where values can be created by the compiler

Values of a **parametric type** cannot be concretely constructed unless all of the parameters are given (themselves with concrete types)

In the case of ``Complex{Float64}``
 
#. ``Complex`` is an abstract complex number type
#. ``Float64`` is a concrete type declaring what the type of the real and imaginary parts of the value should store

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

Anytime a value is prefixed by a colon, as in the ``:a`` above, the type is ``Symbol``--a special kind of string used by the compiler

.. code-block:: julia

    typeof(:a)


(See `parametric types documentation <https://docs.julialang.org/en/v1/manual/types/#Parametric-Types-1>`_)

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

However, outside of these sorts of examples and tests, it is usually a bad idea to change the type of a variable haphazardly

Beyond a few notable exceptions for error handling (e.g. ``nothing`` used for `error handling <error_handling>`_), changing types is usually a symptom of poorly organized code--and it makes compiler `type inference <type_inference>`_ more difficult

The Type Hierarchy
=====================

Let's discuss how types are organized in Julia

Abstract vs Concrete Types
---------------------------
(See `abstract types documentation  <https://docs.julialang.org/en/v1/manual/types/#Abstract-Types-1>`_)

In our example above, the Julia library code for ``Array``and ``Complex`` are written in a way such that it will work for any ``Real`` type--which ``Float64`` fulfills

In this case, ``Real`` is an **abstract type**, and a value of type ``Real`` can never be created directly 

Instead, it provides a way to write :doc:`generic <generic_programming>` code for specific to any concrete types based on ``Real``

We saw above that ``Float64`` is the standard type for representing a 64 bit
floating point number

But we've also seen references to types such as ``Real`` and ``AbstractFloat``

The former (i.e., ``Float64``) is an example of a **concrete type**, as is ``Int64`` or ``Float32``

The latter (i.e., ``Real``, ``AbstractFloat``) are examples of so-called **abstract types**

Concrete types are types that we can *instantiate* --- i.e., pair with data in memory

On the other hand, abstract types help us organize and work with related concrete types


The Type Hierarchy
----------------------

How exactly do abstract types organize or relate different concrete types?

The answer is that, in the Julia language specification, the types form a hierarchy

You can check if a type is a subtype of another with the ``<:`` operator

.. code-block:: julia

    @show Float64 <: Real
    @show Int64 <: Real
    @show Complex{Float64} <: Real
    @show Array <: Real;

In the above, both ``Float64`` and ``Int64`` are **subtypes** of ``Real``, whereas the ``Complex`` numbers are not

They are, however, all subtypes of ``Number``

.. code-block:: julia

    @show Real <: Number
    @show Float64 <: Number
    @show Int64 <: Number
    @show Complex{Float64} <: Number;


``Number`` in turn is a subtype of ``Any``, which is a parent of all types


.. code-block:: julia

    Number <: Any


In particular, the type tree is organized with ``Any`` at the top and the concrete types at the bottom

We never actually see *instances* of abstract types (i.e., ``typeof(x)`` never returns an abstract type)

The point of abstract types is to categorize the concrete types, as well as other abstract types that sit below them in the hierarchy

.. _type_inference:

Deducing and Declaring Types
=============================

We will discuss this in detail in :doc:`this lecture <generic_programming>`, but much of its performance gains and generality of notation comes from Julia's type system

For example, with

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

Analyzing Function Return Types (Advanced)
-------------------------------------------

For the most part, time spent "optimizing" julia code to run faster is able ensuring the compiler can correctly deduce types for all functions

We will discuss this in more detail in :doc:`this lecture <need_for_speed>`, but to give a hint

.. code-block:: julia

    x = [1, 2, 3]
    f(x) = 2x
    @code_warntype f(x)

Here, the ``Body::Array{Int64,1}`` tells us the type of the return value of the function when called with ``[1, 2, 3]`` is always a vector of integers

In contrast, consider a function potentially returning ``nothing``, as in :doc:`this lecture <fundamental_types>`

.. code-block:: julia

    f(x) = x > 0.0 ? x : nothing
    @code_warntype f(1)

This states that the compiler determines the return type could be one of two different types, ``Body::Union{Nothing, Int64}`` 


Good Practices for Functions and Variables
--------------------------------------------

In order to keep many of the benefits of Julia, you will sometimes want to help the compiler ensure that it can always deduce a single type from any function or expression

As an example of bad practice, is to use an array to hold unrelated types

.. code-block:: julia

    x = [1.0, "test", 1] # typically poor style

The type of this is ``Array{Any,1}``, where the ``Any`` means the compiler has determined that any valid Julia types can be added to the array

While occasionally useful, this is to be avoided whenever possible in performance sensitive code

The other place this can come up is in the declaration of functions

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

You will notice that in the lecture notes we have never directly declared any types

This is intentional for exposition and for serious "user" code of packages, rather than the writing of those packages themselves

It is also in contrast to some of the sample code you will see in other Julia sources, which you will need to be able to read

To give an example of the declaration of types, the following are equivalent

.. code-block:: julia

    function f(x, A)
        b = [5.0, 6.0]
        return A * x .+ b
    end
    val = f([0.1, 2.0], [1.0 2.0; 3.0 4.0])

.. code-block:: julia

    function f2(x::Vector{Float64}, A::Matrix{Float64})::Vector{Float64} # argument and return types
        b::Vector{Float64} = [5.0, 6.0]
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

Declaring Struct
-----------------

TODO: Another major diff


Multiple Dispatch
==================
use abs for numbers and complex numbers

special code for trapezoidal rule for a uniform vs. non-uniform grid

Exercises
=============

Implement the trap for both