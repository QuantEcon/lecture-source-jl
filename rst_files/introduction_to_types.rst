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

* multiple-dispatch

* building user-defined types

Setup
------

.. literalinclude:: /_static/includes/deps.jl

<<<<<<< HEAD
=======
    using InstantiateFromURL
    activate_github("QuantEcon/QuantEconLecturePackages", tag="v0.3.1")
    using LinearAlgebra, Statistics, Compat
>>>>>>> d5cba6c... Type hiearchy merged

Finding and Interpreting Types
================================

Finding The Type
--------------------------------

As we have seen in the previous lectures, in Julia all values have a type, which can be queried using the ``typeof`` function

.. code-block:: julia

    @show typeof(1)
    @show typeof(1.0);

The harcoded values ``1`` and ``1.0`` are called literals in a programming language, and the compiler will deduce their types (``Int64`` and ``Float64`` respectively in the example above)

You can also query the type of a value

.. code-block:: julia

    x = 1
    typeof(x)

Where the name ``x`` binds to the value ``1``, created as a literal

Parametric Types
--------------------------------

(See `parametric types documentation <https://docs.julialang.org/en/v1/manual/types/#Parametric-Types-1>`_)

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

**Remark:** Note that, by convention, type names use CamelCase ---  ``Array``, ``AbstractArray``, etc.

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


Good Practices for Functions and Variable Types
-------------------------------------------------

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
    @show f(-1);

The issue here is relatively subtle:  ``1.0`` is a floating point, while ``0`` is an integer

Consequently, given the type of ``x``, the compiler cannot in general determine what type the function will return

This issue, called **type stability** is at the heart of most Julia performance considerations

Luckily, the practice of trying to ensure that functions return the same types is also the most consistent with simple, clear code


Manually Declaring Function and Variable Types
-------------------------------------------------

(See `type declarations documentation <https://docs.julialang.org/en/v1/manual/types/#Type-Declarations-1>`_)

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


Creating New Types
====================

(See `type declarations documentation <https://docs.julialang.org/en/v1/manual/types/#Type-Declarations-1>`_)

Up until now, we have used ``NamedTuple`` to collect sets of parameters for our models and examples

There are many reasons to use that for the narrow purpose of maintaining values for model parameters, but you will eventually need to be able to read code that creates its own typse

Syntax for Creating Concrete Types
-------------------------------------

(See `composite types documentation <https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1>`_)

While other sorts of types exist, we almost always use the ``struct`` keyword, which is for creation of composite data types

Notes:

* "composite" refers to the fact that the data types in question can be used as collection of named fields

* the ``struct`` terminology is used in a number of programming languages to refer to composite data types

Let's start with a trivial example where the ``struct`` we build has fields named ``a, b, c``, are not typed

.. code-block:: julia

    struct FooNotTyped
        a # BAD! Not typed
        b
        c
    end

And another where the types of the fields are chosen

.. code-block:: julia

    struct Foo
        a::Float64 # or just `a` if not declaring type
        b::Int64
        c::Vector{Float64}
    end

In either case, the compiler generates a function to create new values of the data type, called a "constructor"

It has the same name as the data type but uses function call notion

.. code-block:: julia

    foo_nt = FooNotTyped(2.0, 3, [1.0, 2.0, 3.0]) # new FooNotTyped
    foo = Foo(2.0, 3, [1.0, 2.0, 3.0])  # creates a new Foo
    @show typeof(foo)
    @show foo.a # get the value for a field
    @show foo.b
    @show foo.c;

You will notice two differences above for the creation of a ``struct`` compared to our use of ``NamedTuple``

* Types are declared for the fields, rather than inferred by the compiler
* The construction of a new instance, has no named parameters to prevent accidental misuse by choosing the wrong order

Issues with Type Declarations
-------------------------------

Was it necessary to manually declare the types ``a::Float64`` in the above struct?

The answer, in practice, is usually yes

Without a declaration of the type, the compiler is unable to generate efficient code, and the use of a ``struct`` declared without types could drop performance by orders of magnitude

Moreover, it is very easy to use the wrong type, or unnecessarily constrain the types

The first example, which is usually just as low-performance as no declaration of types at all, is to accidentally declare it with an abstract type

.. code-block:: julia

    struct Foo2
        a::Float64
        b::Integer # BAD! Not a concrete type
        c::Vector{Real} # BAD! Not a concrete type
    end

The second issue is that by choosing a type (as in the ``Foo`` above), you may be constraining what is allowed more than is really necessary

.. code-block:: julia

    f(x) = x.a + x.b + sum(x.c) # use the type
    a = 2.0
    b = 3
    c = [1.0, 2.0, 3.0]
    foo = Foo(a, b, c)
    f(foo) # call with the foo, no problem

    # Some other typed for the values 
    a = 2 # not a floating point, but f() would work
    b = 3
    c = [1.0, 2.0, 3.0]' # transpose is not a `Vector`. But f() would work
    # foo = Foo(a, b, c) # fails to compile

    # works with the NotTyped version, but low performance
    foo_nt = FooNotTyped(a, b, c)

Declaring Parametric Types (Advanced)
----------------------------------------

(See `type parametric types documentation <https://docs.julialang.org/en/v1/manual/types/#Parametric-Types-1>`_)

Motivated by the above, we can create a type which can adapt to holding fields of different types

.. code-block:: julia

    struct Foo3{T1, T2, T3}
        a::T1 # could be any type
        b::T2
        c::T3
    end

    # Works fine
    a = 2
    b = 3
    c = [1.0, 2.0, 3.0]' # transpose is not a `Vector`. But f() would work
    foo = Foo3(a, b, c)
    f(foo)

Of course, this is probably too flexible, and the ``f`` function might not work on an arbitrary set of ``a, b, c``

You could constrain the types based on the abstract parent type using the ``<:`` operator

.. code-block:: julia

    struct Foo4{T1 <: Real, T2 <: Real, T3 <: AbstractVecOrMat{<:Real}}
        a::T1
        b::T2
        c::T3 # should check dimensions as well
    end
    foo = Foo4(a, b, c) # no problem, and high performance

This ensure that

* ``a`` and ``b`` are a subtype of ``Real``, which ensures that the ``+`` in the definition of ``f`` works
* ``c`` is a one dimensional abstract array of ``Real`` values

The code works, and is equivalent in performance to a ``NamedTuple``, but is becoming verbose and error prone

Keyword Argument Constructors (Advanced)
-------------------------------------------

There is no way around the difficult creation of parametric types to achieve high performance code

However, the other issue where constructor arguments are error-prone, can be remedied with the ``Parameters.jl`` library

.. code-block:: julia

    using Parameters
    @with_kw  struct Foo5
        a::Float64 = 2.0 # adds default value
        b::Int64
        c::Vector{Float64}
    end
    foo = Foo5(a = 0.1, b = 2, c = [1.0, 2.0, 3.0])
    foo2 = Foo5(c = [1.0, 2.0, 3.0], b = 2) # rearrange order, uses default values
    @show foo
    @show foo2

    function f(x)
        @unpack a, b, c = x # can use @unpack on any struct
        return a + b + sum(c) 
    end
    f(foo)


Introduction to Multiple Dispatch
===================================

One of the defining features of Julia is **multiple dispatch**, whereby the same function name can do different things depending on the underlying types

Without realizing it, in nearly every function call within packages or the standard library you have used this features

To see this in action, consider the absolute value function ``abs``

.. code-block:: julia

    @show abs(-1) # Int64
    @show abs(-1.0) # Float64
    @show abs(0.0 - 1.0im); # Complex{Float64}

In all of these cases, the ``abs`` function has specialized code depending on the type passed in

To do this, you need to specify different **methods** of the function which operate on a particular set of types

Unlike most cases we have seen before, this requires a type annotation

To rewrite the ``abs`` function

.. code-block:: julia

    function ourabs(x::Real)
        if x > zero(x) # note, not 0!
            return x
        else
            return -x
        end
    end

    function ourabs(x::Complex)
        sqrt(real(x)^2 + imag(x)^2)
    end

    @show ourabs(-1) # Int64
    @show ourabs(-1.0) # Float64
    @show ourabs(1.0 - 2.0im); # Complex{Float64}

Note that in the above, ``x`` works for any type of ``Real``, including ``Int64``, ``Float64``, and ones you may not have realized exist


.. code-block:: julia

    x = -2//3 # a Rational number
    @show typeof(x)
    @show ourabs(x)



Multiple Dispatch
==================
use abs for numbers and complex numbers

special code for trapezoidal rule for a uniform vs. non-uniform grid

Exercises
=============

.. special code for trapezoidal rule for a uniform vs. non-uniform grid
