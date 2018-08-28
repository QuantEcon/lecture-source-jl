.. _types_methods:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Types, Methods and Dispatch
******************************************

.. TODO: Add some discussion about names and scope?  Or to julia_essentials?

.. contents:: :depth: 2

Overview
============================

In this lecture we delve more deeply into the structure of Julia, and in particular into

* the concept of types

* methods and multiple dispatch

* building user-defined types


These concepts relate to the way that Julia stores and acts on data

Understanding them will help you

* Write "well organized" Julia code that's easy to read, modify, maintain and debug

* Improve the speed at which your code runs

* Read Julia code written by other programmers


.. _julia_types:



Types and Multiple Dispatch
===========================


Common Types
---------------

In Julia all objects have a type, which can be queried using the ``typeof()`` function


.. code-block:: julia

    typeof(0.5)


.. code-block:: julia

    typeof(5)


.. code-block:: julia

    typeof("foo")


.. code-block:: julia

    typeof('c')


The next two types use curly bracket notation to express the fact that they are *parametric*


.. code-block:: julia

    typeof(1 + 1im)


.. code-block:: julia

    typeof(Matrix{Float64}(undef, 2, 2))


We'll return to parametric types later in this lecture


Remark: Note that, by convention, type names use CamelCase ---  ``FloatingPoint``, ``Array``, ``AbstractArray``, etc.


Variables and Type
--------------------


After assigning a variable name to an object, we can query the type of the
object via the name

.. code-block:: julia

    x = 42


.. code-block:: julia

    typeof(x)


The type resides with the object itself, not with the name ``x``

Thus, ``x`` is just a symbol bound to an object of type ``Int64``

Indeed, we can *rebind* the symbol ``x`` to any other object, of the same type or otherwise

.. code-block:: julia

    x = 42.0


Now ``x`` "points to" another object, of type ``Float64``

.. code-block:: julia

    typeof(x)


Multiple Dispatch
------------------


When we process data with a computer, the precise data type is important --- sometimes more than we realize

For example, on an abstract mathematical level we don't distinguish between
`1 + 1` and `1.0 + 1.0`

But for a CPU, integer and floating point addition are different things, using
a different set of instructions

Julia handles this problem by storing multiple, specialized versions of functions like addition, one for each data type or set of data types

These individual specialized versions are called **methods**

When an operation like addition is requested, the Julia runtime environment inspects the type of data to be acted on and hands it out to the appropriate method

This process is called **multiple dispatch**

Example 1
^^^^^^^^^

In Julia, `1 + 1` has the alternative syntax `+(1, 1)`

.. code-block:: julia

    +(1, 1)


This operator `+` is itself a function with multiple methods

We can investigate them using the `@which` macro, which shows the method to which a given call is dispatched

.. code-block:: julia

    x, y = 1.0, 1.0
    @which +(x, y)


We see that the operation is sent to the ``+`` method that specializes in adding
floating point numbers

Here's the integer case

.. code-block:: julia

    x, y = 1, 1
    @which +(x, y)


This (slightly edited) output says that the call has been dispatched to the `+` method
responsible for handling integer values

(We'll learn more about the details of this syntax below)

Here's another example, with complex numbers

.. code-block:: julia

    x, y = 1.0 + 1.0im, 1.0 + 1.0im
    @which +(x, y)


Again, the call has been dispatched to a `+` method specifically designed for handling the given data type


Example 2
^^^^^^^^^

The `isless` function also has multiple methods

.. code-block:: julia

    isless(1.0, 2.0)  # Applied to two floats


.. code-block:: julia

    @which isless(1.0, 2.0)


Now let's try with integers

.. code-block:: julia

    @which isless(1, 2)


The `Real` data type we haven't met yet --- it's an example of an *abstract* type, and encompasses both floats and integers

We'll learn more about abstract types below


Example 3
^^^^^^^^^^^^^^


The function ``isfinite()`` has multiple methods too

.. code-block:: julia

    @which isfinite(1) # Call isfinite on an integer


.. code-block:: julia

    @which isfinite(1.0) # Call isfinite on a float


Here ``AbstractFloat`` is another abstract data type, this time encompassing all floats

We can list all the methods of ``isfinite`` as follows

.. code-block:: julia

    methods(isfinite)


We'll discuss some of the more complicated data types you see here later on


Adding Methods
-------------------------------

It's straightforward to add methods to existing functions

For example, we can't at present add an integer and a string in Julia

.. code-block:: julia
    :class: no-execute

    +(100, "100")


This is sensible behavior, but if you want to change it there's nothing to stop you:


.. code-block:: julia

    import Base: +  #  Gives access to + so that we can add a method

    +(x::Integer, y::String) = x + parse(Int, y)


.. code-block:: julia

    +(100, "100")


.. code-block:: julia

    100 + "100"


Dispatch and User-Defined Functions
------------------------------------

You can exploit multiple dispatch in user-defined functions

Here's a trivial example (we'll see many realistic examples later)

.. code-block:: julia

    function h(a::Float64)
        println("You have called the method for handling Float64s")
    end

    function h(a::Int64)
        println("You have called the method for handling Int64s")
    end


The `h` that gets invoked depends on the data type that you call it with:

.. code-block:: julia

    h(1.0)


.. code-block:: julia

    h(1)


Actually, as we'll see when we :doc:`discuss JIT compilation
<need_for_speed>`, this process is partly automated

For example, if we write a function that can handle either floating point or integer arguments and then call it with floating point arguments, a specialized method for applying our function to floats will be constructed and stored in memory

* Inside the method, operations such as addition, multiplication, etc. will be specialized to their floating point versions

If we next call it with integer arguments, the process will be repeated but now
specialized to integers

* Inside the method, operations such as addition, multiplication, etc. will be specialized to their integer versions


Subsequent calls will be routed automatically to the most appropriate method


Comments on Efficiency
------------------------


Julia's multiple dispatch approach to handling data differs from the approach
used by many other languages

It is, however, well thought out and well suited to scientific computing

The reason is that many methods, being specialized to specific data types, are highly optimized for the kind of data that they act on

We can likewise build specialized methods and hence generate fast code

We'll see how this enables Julia to easily generate highly efficient machine code in :doc:`later on <need_for_speed>`


The Type Hierarchy
=====================

Let's discuss how types are organized in Julia


Abstract vs Concrete Types
---------------------------

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

For example, ``Float64`` and ``Int64`` are **subtypes** of ``Real``

.. code-block:: julia

    Float64 <: Real

.. code-block:: julia

    Int64 <: Real


On the other hand, 64 bit complex numbers are not reals

.. code-block:: julia

    ComplexF32 <: Real


They are, however, subtypes of ``Number``

.. code-block:: julia

    ComplexF32 <: Number


``Number`` in turn is a subtype of ``Any``, which is a parent of all types


.. code-block:: julia

    Number <: Any


In particular, the type tree is organized with ``Any`` at the top and the concrete types at the bottom


We never actually see *instances* of abstract types (i.e., ``typeof(x)`` never returns an abstract type)

The point of abstract types is to categorize the concrete types, as well as other abstract types that sit below them in the hierarchy


Back to Multiple Dispatch
-----------------------------------

We can now be a little bit clearer about what happens when you call a function on given types

Suppose we execute the function call ``f(a, b)`` where ``a`` and ``b``
are of concrete types ``S`` and ``T`` respectively

The Julia interpreter first queries the types of ``a`` and ``b`` to obtain the tuple ``(S, T)``

It then parses the list of methods belonging to ``f``, searching for a match

If it finds a method matching ``(S, T)`` it calls that method

If not, it looks to see whether the pair ``(S, T)`` matches any method defined for *immediate parent types*

For example, if ``S`` is ``Float64`` and ``T`` is ``ComplexF32`` then the
immediate parents are ``AbstractFloat`` and ``Number`` respectively


.. code-block:: julia

    supertype(Float64)


.. code-block:: julia

    supertype(ComplexF32)


Hence the interpreter looks next for a method of the form ``f(x::AbstractFloat, y::Number)``

If the interpreter can't find a match in immediate parents (supertypes) it proceeds up the tree, looking at the parents of the last type it checked at each iteration

* If it eventually finds a matching method it invokes that method

* If not, we get an error

This is the process that leads to the following error

.. code-block:: julia

    +(100, "100")


Because the dispatch procedure starts from concrete types and works upwards, dispatch always invokes the *most specific method* available

For example, if you have methods for function ``f`` that handle

#.  ``(Float64, Int64)`` pairs

#.  ``(Number, Number)`` pairs

and you call ``f`` with ``f(0.5, 1)`` then the first method will be invoked

This makes sense because (hopefully) the first method is optimized for
exactly this kind of data

The second method is probably more of a "catch all" method that handles other
data in a less optimal way


Here's another simple example, involving a user-defined function

.. code-block:: julia

    function f(x)
        println("Generic function invoked")
    end

    function f(x::Number)
        println("Number method invoked")
    end

    function f(x::Integer)
        println("Integer method invoked")
    end

Let's now run this and see how it relates to our discussion of method dispatch
above

.. code-block:: julia

    f(3)


.. code-block:: julia

    f(3.0)


.. code-block:: julia

    f("foo")


Since

* ``3`` is an ``Int64`` and ``Int64 <: Integer <: Number``

the call ``f(3)`` proceeds up the tree to ``Integer`` and invokes ``f(x::Integer)``

On the other hand, ``3.0`` is a ``Float64``, which is not a subtype of  ``Integer``

Hence the call ``f(3.0)`` continues up to ``f(x::Number)``

Finally, ``f("foo")`` is handled by the generic function, since ``String`` is not a subtype of ``Number``


User-Defined Types
==============================

Let's have a look at defining our own data types


Motivation
----------------------

At our respective homes we both have draws full of fishing gear

Of course we have draws full of other things too, like kitchen utensils, or clothes

Are these draws really necessary?

Perhaps not, but who wants to search the whole house for their fishing reel when the fish are biting?

Certainly not us

Just as it's convenient to store household objects in draws, it's also
convenient to organize the objects in your program into
designated "containers"

The first step is to design and build the containers

We do this by declaring and using our own types

For example,

* a ``Firm`` type might store parameters for objects that represent firms in a given model

* an ``EstimationResults`` type might store output from some statistical procedure, etc.


Once those types are declared, we can create instances of the type

For example,

  ``results = EstimationResults(y, X)``

might create an instances of ``EstimationResults`` that stores estimated coefficients and other information from a given regression exercise involving data `y, X`


Syntax
---------

While there are multiple ways to create new types, we almost always use the ``struct`` keyword, which is for creation of composite data types

Notes:

* "composite" refers to the fact that the data types in question can be used as "containers" that hold a variety of data

* the ``struct`` terminology is used in a number of programming languages to refer to composite data types

Let's start with a trivial example where the ``struct`` we build is empty

.. code-block:: julia

    struct Foo  # A useless data type that stores no data
    end

When a new data type is defined in this way, the interpreter also creates a *default constructor* for the data type

This constructor is a function for generating new instances of the data type in question

It has the same name as the data type but uses function call notion:

.. code-block:: julia

    foo = Foo()  # Call default constructor, make a new Foo

A new instance of type ``Foo`` is created and the name ``foo`` is bound to
that instance

.. code-block:: julia

    typeof(foo)


Adding Methods
--------------

We can now create functions that act on instances of ``Foo``

.. code-block:: julia

    foofunc(x::Foo) = "onefoo"


.. code-block:: julia

    foofunc(foo)


Or we can add new methods for acting on Foos to existing functions, such as `+`

.. code-block:: julia

    +(x::Foo, y::Foo) = "twofoos"


.. code-block:: julia

    foo1, foo2 = Foo(), Foo()  # Create two Foos


.. code-block:: julia

    +(foo1, foo2)


A Less Trivial Example
-------------------------

Let's say we are doing a lot of work with AR(1) processes, which
are random sequences :math:`\{X_t\}` that follow the law of motion

.. math::
    X_{t+1} = a X_t + b + \sigma W_{t+1}
    :label: tm_ar1

Here

* :math:`a`, :math:`b` and :math:`\sigma` are scalars and

* :math:`\{W_t\}` is an iid sequence of shocks with some given distribution :math:`\phi`

Let's take these primitives :math:`a`, :math:`b`, :math:`\sigma` and :math:`\phi`
and organize them into a single entity like so

.. code-block:: julia

    mutable struct AR1
        a
        b
        σ
        ϕ
    end


Here ``mutable`` means that we can change (mutate) data while the object is live in memory -- see below


For the distribution ``ϕ`` we'll assign a ``Distribution`` from the `Distributions <https://github.com/JuliaStats/Distributions.jl>`__ package

.. code-block:: julia

    using Distributions

.. code-block:: julia

    m = AR1(0.9, 1, 1, Beta(5, 5))


In this call to the constructor we've created an instance of ``AR1`` and bound the name ``m`` to it

We can access the fields of ``m`` using their names and "dotted attribute" notation

.. code-block:: julia

    m.a


.. code-block:: julia

    m.b


.. code-block:: julia

    m.σ


.. code-block:: julia

    m.ϕ


For example, the attribute ``m.ϕ`` points to an instance of ``Beta``, which is in turn a subtype of ``Distribution`` as defined in the Distributions package

.. code-block:: julia

    typeof(m.ϕ)


.. code-block:: julia

    m.ϕ isa Distribution


We can reach into ``m`` and change this if we want to

.. code-block:: julia

    m.ϕ = Exponential(0.5)


.. _spec_field_types:


Specifying Field Types
^^^^^^^^^^^^^^^^^^^^^^^^^


In our type definition we can be explicit that we want ``ϕ`` to be a
``Distribution`` and the other elements to be floats

.. code-block:: julia

    struct AR1_explicit
        a::Float64
        b::Float64
        σ::Float64
        ϕ::Distribution
    end

(In this case, ``mutable`` is removed since we do not intend to make any changes to the elements of ``AR1_explicit``)

Now the constructor will complain if we try to use the wrong data type

.. code-block:: julia
    :class: no-execute

    m = AR1_explicit(0.9, 1, "foo", Beta(5, 5))


This can be useful in terms of failing early on incorrect data, rather than
deeper into execution

At the same time, `AR1_explicit` is not as generic as `AR1`, and hence less flexible

For example, suppose that we want to allow `a`, `b` and `σ` to take any
value that is `<: Real`

We could achieve this by the new definition

.. code-block:: julia

    struct AR1_real
        a::Real
        b::Real
        σ::Real
        ϕ::Distribution
    end


But it turns out that using abstract types inside user-defined types adversely
affects performance --- more about that :doc:`soon <need_for_speed>`

Fortunately, there's another approach that both

* preserves the use of concrete types for internal data and

* allows flexibility across multiple concrete data types

This approach uses *type parameters*, a topic we turn to now


Type Parameters
-------------------

Consider the following output

.. code-block:: julia

    typeof([10, 20, 30])


Here ``Array`` is one of Julia's predefined types (``Array <: DenseArray <: AbstractArray <: Any``)

The ``Int64,1`` in curly brackets are **type parameters**

In this case they are the element type and the dimension

Many other types have type parameters too

.. code-block:: julia

    typeof(1.0 + 1.0im)


.. code-block:: julia

    typeof(1 + 1im)


Types with parameters are therefore in fact an indexed family of types, one for each possible value of the parameter


We can use parametric types in our own type definitions, as the next example shows


Back to the AR1 Example
-------------------------

Recall our AR(1) example, where we considered different restrictions on internal data

For the coefficients `a`, `b` and `σ`  we considered

* allowing them to be any type

* forcing them to be of type `Float64`

* allowing them to be any `Real`

The last option is a nice balance between specific and flexible

For example, using `Real` in the type definition tells us that, while these values should be scalars, integer values and floats are both OK

However, as mentioned above, using abstract types for fields of user-defined types impacts negatively on performance

For now it suffices to observe that we can achieve flexibility and eliminate
abstract types on `a`, `b`, `σ`, and `ϕ` by the following declaration


.. code-block:: julia

    struct AR1_best{T <: Real, D <: Distribution}
        a::T
        b::T
        σ::T
        ϕ::D
    end

If we create an instance using `Float64` values and a `Beta` distribution then the instance has type
`AR1_best{Float64,Beta}`

It is worth nothing that under this definition, the instance can only be created by
providing `a`, `b`, and `σ` of the same type. One could make it flexible enough to
parameterize on different values or providing a constructor that converts the inputs
to the same type (e.g., using `promote_type`)

.. code-block:: julia

    m = AR1_best(0.9, 1.0, 1.0, Beta(5, 5))


Exercises
===========


Exercise 1
---------------

Write a function with the signature ``simulate(m::AR1, n::Integer, x0::Real)``
that takes as arguments

* an instance ``m`` of ``AR1`` (see above)
* an integer ``n``
* a real number ``x0``

and returns an array containing a time series of length ``n`` generated according to :eq:`tm_ar1` where

* the primitives of the AR(1) process are as specified in ``m``

* the initial condition :math:`X_0` is set equal to ``x0``

Hint: If ``d`` is an instance of ``Distribution`` then ``rand(d)`` generates one random draw from the distribution specified in ``d``


Solutions
==========


Exercise 1
----------

Let's start with the AR1 definition as specified in the lecture

.. code-block:: julia

    struct AR1_ex1{T <: Real, D <: Distribution}
        a::T
        b::T
        σ::T
        ϕ::D
    end

Now let's write the function to simulate AR1s

.. code-block:: julia

    function simulate(m::AR1_ex1, n::Integer, x0::Real)
        X = zeros(n)
        X[1] = x0
        for t ∈ 1:(n-1)
            X[t+1] = m.a * X[t] + m.b + m.σ * rand(m.ϕ)
        end
        return X
    end


Let's test it out on the AR(1) process discussed in the lecture

.. code-block:: julia

    m = AR1_ex1(0.9, 1.0, 1.0, Beta(5, 5))
    X = simulate(m, 100, 0.0)


Next let's plot the time series to see what it looks like

.. code-block:: julia

    using Plots

    plot(X, legend=:none)
