.. _generic_programming:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Generic and Functional Programming
******************************************

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

Thanks to Jeffrey Sarnoff

Setup
------

.. literalinclude:: /_static/includes/deps.jl


Generic Programming Foundations and Mathematics
==================================================

.. Un-learn Object-Oriented Programming
.. -----------------------------------------
.. 
.. The superficial similarity can lead to misuse
.. 
.. Types are *not* classes, and methods are not simply member functions

At the root of all types is ``Any``

There are a (very limited) set of operations which are available for any type, including a ``show`` function and ``typeof``

.. code-block:: julia

    using Distributions
    x = 1
    y = Normal()
    @show x #
    @show y
    @show typeof(x)
    @show typeof(y)
    @show supertype(typeof(x))
    @show typeof(x) |> supertype # pipe just applies a function around another functoin
    @show typeof(y) |> supertype;

We will investigate some of the sub-types of ``Any``

Example: `Number` and Algebraic Structures
----------------------------------------------

In mathematics, a `Ring <https://en.wikipedia.org/wiki/Ring_(mathematics)>`_ is a set with two binary operators (:math:`+` and :math:`\cdot`, called the additive and multiplicative operators) where there is an
* additive operator is associative and commutative
* multiplicative operator is associative and and distributive with respect to the additive operator
* additive identity element, usually denoted ``0`` such that :math:`a + 0 = a` for any :math:`a` in the set
* additive inverse of each element, i.e. :math:`-a` such that :math:`a + (-a) = 0`
* multiplicative identity element, usually denoted ``1`` such that :math:`a \cdot 1 = a`

This is skipping over a few other key parts of the definition, but it is also useful to say what is not needed
* A total or partial ordering is not required (i.e. there does not need to be any meaningful ``<`` operator defined)
* A multiplicative inverse is not required

This algebraic structure provides motivation for the abstract ``Number`` type in Julia
**Remark** We use motivation here because they are not formally connected and the mapping is imperfect
* The main difficulty when dealing with numbers that can be concretely created on a computer is that the closure requirements are difficult to ensure (e.g. floating points have finite numbers of bits of information)

Let ``typeof(a) = typeof(b) = T <: Number``, then under an implicit definition of the **generic interface** for ``Number`` the following must be defined
* the additive operator: ``a + b``
* the multiplicative operator: ``a * b``
* an additive inverse operator: ``-a``
* an inverse operation for addition ``a - b = a + (-b)``
* an additive identity: ``zero(T)`` or ``zero(a)`` for convenience
* a multiplicative identity: ``one(T)`` or ``one(a)`` for convenience


The core of generic programmig is that, given the knowledge that a value is of type ``Number``, we can write to that generic interface

To demonstrate these for a complex number or a big integer (i.e., two types other than the standard ``Float64`` ``Int64`` you may associate with numbers)

.. code-block:: julia

    a = 1.0 + 1.0im
    b = 0.0 + 2.0im
    @show typeof(a)
    @show typeof(a) <: Number
    @show a + b
    @show a * b
    @show -a
    @show a - b
    @show zero(a)
    @show one(a);

.. code-block:: julia

    a = BigInt(10)
    b = BigInt(4)
    @show typeof(a)
    @show typeof(a) <: Number
    @show a + b
    @show a * b
    @show -a
    @show a - b
    @show zero(a)
    @show one(a);


Thinking back to the mathematical motivation, a `Field <https://en.wikipedia.org/wiki/Field_(mathematics)>`_ is an `Ring` with a few additional properties, among them
* a multiplicative inverse: :math:`a^{-1}`
* an inverse operation for multiplication: :math:`a / b = a \cdot b^{-1}`
* a (not strictly required) total ordering operation: :math:`a < b`

where the  `total ordering <https://en.wikipedia.org/wiki/Total_order#Strict_total_order>`_  operation would fulfill the requirements for a Totally Ordered Field

This type gives some motivation for the operations and properties of the ``Real`` type 

Of course, ``Complex{Float64} <: Number`` but not ``Real``, which makes sense since those operations are not defined for the basic Complex number type in mathematics

These operations are implemented in any subtype of ``Real`` through
* the multiplicative inverse: ``inv(a)``
* the multiplicative inverse operation: ``a / b = a * inv(b)``
* an ordering ``a < b``

To demonstrate with the ``Rational`` type

.. code-block:: julia

    a = 1 // 10
    b = 4 // 6
    @show typeof(a)
    @show typeof(a) <: Number
    @show typeof(a) <: Real
    @show inv(a)
    @show a / b
    @show a < b;

**Remark** Here we see where and how the precise connection to the mathematics for number types breaks down for practical reasons, in particular
* ``Integer`` types (i.e. ``Int64 <: Integer``) do not have a a multiplicative inverse with closure in the set
* However, it is necessary in practice for integer division to be defined, and return back a member of the ``Reals``
* in computer science, this is called type promotion, where a type can be converted to another to ensure an operation is possible

Moving further down the tree of types provides more operations, which start to become more tied to the computational implementation than the mathematics

For example, floating point numbers all have a machine precision below which there is no way to further differentiate numbers

.. code-block:: julia

    @show Float64 <: AbstractFloat
    @show BigFloat <: AbstractFloat
    @show eps(Float64)
    @show eps(BigFloat);

Example: Distributions
----------------------------------------------

First, lets look at the tree of types for a ``Normal`` distribution, as implemented in the ``Distributions.jl`` package

.. code-block:: julia

    using Distributions
    d1 = Normal(1.0, 2.0) # an example type to explore
    @show d1
    @show typeof(d1)
    @show typeof(d1) |> supertype # i.e. supertype(typeof(d1))
    @show typeof(d1) |> supertype |> supertype
    @show typeof(d1) |> supertype |> supertype |> supertype;

The ``Sampleable{Univariate,Continuous}`` type has a limited number of functions, chiefly the ability to draw a random number

.. code-block:: julia

    @show rand(d1);

The purpose of that abstract type is to provide an interface for drawing from a variety of distributions, some of which may not have a well-defined predefined

If you were writing a function a stochastic process with an arbitrary `iid` shocks, where you did not need to assume an existing of a ``pdf`` etc., this is a natural candidate

For example, to simulate :math:`x_{t+1} = a x_t + b \epsilon_{t+1}` where :math:`\epsilon \sim D` for any `D` we can draw from

.. code-block:: julia

    function simulateprocess(x₀; a = 1.0, b = 1.0, N = 10, d::Sampleable{Univariate,Continuous})
        x = zeros(typeof(x₀), N+1) # preallocate vector, careful on the type
        x[1] = x₀
        for t in 2:N+1
            x[t] = a * x[t-1] + b * rand(d) # draw
        end
        return x
    end
    @show simulateprocess(0.0, d=Normal(0.2, 2.0), N=5);
    # @show simulateprocess(0.0, d=Normal(0.2, 2.0), N=5); #add example of something with pdf

The ``Sampleable{Univariate,Continuous}`` and, especially, the ``Sampleable{Multivariate,Continuous}`` abstract types are useful generic interfaces for monte-carlo and Bayesian methods, in particular, where you can often draw from a distribution, but can do little else  

Moving down the tree, the ``Distributions{Univariate, Continuous}`` abstract type has certain functions we would expect to operate with it

These match the mathematics, such as ``pdf, cdf, quantile, support, minimum, maximum`` and a few others

.. code-block:: julia

    d1 = Normal(1.0, 2.0)
    d2 = Exponential(0.1)
    @show d1
    @show d2
    @show supertype(typeof(d1))
    @show supertype(typeof(d2))

    @show pdf(d1, 0.1)
    @show pdf(d2, 0.1)
    @show cdf(d1, 0.1)
    @show cdf(d2, 0.1)
    @show support(d1)
    @show support(d2)
    @show minimum(d1)
    @show minimum(d2)
    @show maximum(d1)
    @show maximum(d2);

You could create your own ``Distributions{Univariate, Continuous}`` type, if you implemented all of those functions, as is described in `the documentation <https://juliastats.github.io/Distributions.jl/latest/extends.html>`_  

If you fulfill all of the conditions of a particular interface, you (or anyone else) could use code written for the abstract ``Distributions{Univariate, Continuous}`` type without any modifications

As an example, consider the `StatPlots <https://github.com/JuliaPlots/StatPlots.jl>`_ package


.. code-block:: julia

    using StatPlots
        

Limitations of these Structures in Julia
------------------------------------------

You will notice that types in Julia represent a tree with ``Any`` at the root

The tree structure has worked well for the above examples, but it doesn't allow us to associate multiple categorizations of types

For example, a semigroup type would be very useful for a writing generic code (e.g. continuous-time solutions for ODEs and matrix-free methods) but cannot be implemented rigorously since the ``Matrix`` type is a semigroup, but it is also an ``AbstractArray``

In the future, the way to implement this is with a feature called traits


Example: ``isless``
-------------------

The ``isless`` function also has multiple methods


First let's try with integers

.. code-block:: julia

    @which isless(1, 2)

As we saw previously, the `Real` data type is an *abstract* type, and encompasses both floats and integers

If we go to the provided link in the source, we see the entirety of the function is

.. code-block:: julia
    :class: no-execute

    isless(x::Real, y::Real) = x<y


That is, for any values where ``typeof(x) <: Real`` and ``typeof(y) <: Real``, the definition relies on ``<``

We know that ``<`` is defined for the types because it is part of the informal interface for the ``Real`` abstract type

Note that this is not defined for ``Number`` because not all ``Number`` types have the ``<`` operator defined (e.g. ``Complex``)

Of course, in order to generate fast code, 

.. code-block:: julia

    isless(1.0, 2.0)  # Applied to two floats


.. code-block:: julia

    @which isless(1.0, 2.0)

Note that the reason  ``Float64 <: Real``

Understanding Multiple Dispatch in Julia
===============================================

This section provides more background on how methods, functions, and types are connected


Methods and Functions
----------------------

The precise data type is important, for reasons of both efficiency and mathematical correctness

For example consider `1 + 1` vs. `1.0 + 1.0` or `[1 0] + [0 1]`

On a CPU, integer and floating point addition are different things, using a different set of instructions

Julia handles this problem by storing multiple, specialized versions of functions like addition, one for each data type or set of data types

These individual specialized versions are called **methods**

When an operation like addition is requested, the Julia compiler inspects the type of data to be acted on and hands it out to the appropriate method

This process is called **multiple dispatch**


Like all "infix" operators, `1 + 1` has the alternative syntax `+(1, 1)`

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


This output says that the call has been dispatched to the `+` method
responsible for handling integer values

(We'll learn more about the details of this syntax below)

Here's another example, with complex numbers

.. code-block:: julia

    x, y = 1.0 + 1.0im, 1.0 + 1.0im
    @which +(x, y)


Again, the call has been dispatched to a `+` method specifically designed for handling the given data type

.. 
..  Example 3
..  ^^^^^^^^^^^^^^
.. 
.. 
.. The function ``isfinite()`` has multiple methods too
.. 
.. .. code-block:: julia
.. 
..     @which isfinite(1) # Call isfinite on an integer
.. 
.. 
.. .. code-block:: julia
.. 
..     @which isfinite(1.0) # Call isfinite on a float
.. 
.. 
.. Here ``AbstractFloat`` is another abstract data type, this time encompassing all floats
.. 
.. We can list all the methods of ``isfinite`` as follows
.. 
.. .. code-block:: julia
.. 
..     methods(isfinite)
.. 
.. 
.. We'll discuss some of the more complicated data types you see here later on
.. 

Adding Methods
^^^^^^^^^^^^^^^^^^

It's straightforward to add methods to existing functions

For example, we can't at present add an integer and a string in Julia (i.e. ``100 + "100"`` is not valid syntax)

This is sensible behavior, but if you want to change it there's nothing to stop you:


.. code-block:: julia

    import Base: +  #  Gives access to + so that we can add a method

    +(x::Integer, y::String) = x + parse(Int, y)

    @show +(100, "100")
    @show 100 + "100"; #equivalent


.. If we write a function that can handle either floating point or integer arguments and then call it with floating point arguments, a specialized method for applying our function to floats will be constructed and stored in memory
.. 
.. * Inside the method, operations such as addition, multiplication, etc. will be specialized to their floating point versions
.. 
.. If we next call it with integer arguments, the process will be repeated but now
.. specialized to integers
.. 
.. * Inside the method, operations such as addition, multiplication, etc. will be specialized to their integer versions
.. 
.. 
.. Subsequent calls will be routed automatically to the most appropriate method


.. Comments on Efficiency
.. ------------------------
.. 
.. 
.. We'll see how this enables Julia to easily generate highly efficient machine code in :doc:`later on <need_for_speed>`




Understanding the Compilation Process
---------------------------------------

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

.. 
.. User-Defined Types
.. ==============================
.. 
.. Let's have a look at defining our own data types
.. 
.. 
.. Motivation
.. ----------------------
.. 
.. At our respective homes we both have draws full of fishing gear
.. 
.. Of course we have draws full of other things too, like kitchen utensils, or clothes
.. 
.. Are these draws really necessary?
.. 
.. Perhaps not, but who wants to search the whole house for their fishing reel when the fish are biting?
.. 
.. Certainly not us
.. 
.. Just as it's convenient to store household values in draws, it's also
.. convenient to organize the values in your program into
.. designated "containers"
.. 
.. The first step is to design and build the containers
.. 
.. We do this by declaring and using our own types
.. 
.. For example,
.. 
.. * a ``Firm`` type might store parameters for values that represent firms in a given model
.. 
.. * an ``EstimationResults`` type might store output from some statistical procedure, etc.
.. 
.. 
.. Once those types are declared, we can create instances of the type
.. 
.. For example,
.. 
..   ``results = EstimationResults(y, X)``
.. 
.. might create an instances of ``EstimationResults`` that stores estimated coefficients and other information from a given regression exercise involving data `y, X`

.. 
.. Adding Methods
.. --------------
.. 
.. We can now create functions that act on instances of ``Foo``
.. 
.. .. code-block:: julia
.. 
..     foofunc(x::Foo) = "onefoo"
.. 
.. 
.. .. code-block:: julia
.. 
..     foofunc(foo)
.. 
.. 
.. Or we can add new methods for acting on Foos to existing functions, such as `+`
.. 
.. .. code-block:: julia
.. 
..     +(x::Foo, y::Foo) = "twofoos"
.. 
.. 
.. .. code-block:: julia
.. 
..     foo1, foo2 = Foo(), Foo()  # Create two Foos
.. 
.. 
.. .. code-block:: julia
.. 
..     +(foo1, foo2)
.. 
.. 
.. A Less Trivial Example
.. -------------------------
.. 
.. Let's say we are doing a lot of work with AR(1) processes, which
.. are random sequences :math:`\{X_t\}` that follow the law of motion
.. 
.. .. math::
..     X_{t+1} = a X_t + b + \sigma W_{t+1}
..     :label: tm_ar1
.. 
.. Here
.. 
.. * :math:`a`, :math:`b` and :math:`\sigma` are scalars and
.. 
.. * :math:`\{W_t\}` is an iid sequence of shocks with some given distribution :math:`\phi`
.. 
.. Let's take these primitives :math:`a`, :math:`b`, :math:`\sigma` and :math:`\phi`
.. and organize them into a single entity like so
.. 
.. .. code-block:: julia
.. 
..     mutable struct AR1
..         a
..         b
..         σ
..         ϕ
..     end
.. 
.. 
.. Here ``mutable`` means that we can change (mutate) data while the value is live in memory -- see below
.. 
.. 
.. For the distribution ``ϕ`` we'll assign a ``Distribution`` from the `Distributions <https://github.com/JuliaStats/Distributions.jl>`__ package
.. 
.. .. code-block:: julia
.. 
..     using Distributions
.. 
.. .. code-block:: julia
.. 
..     m = AR1(0.9, 1, 1, Beta(5, 5))
.. 
.. 
.. In this call to the constructor we've created an instance of ``AR1`` and bound the name ``m`` to it
.. 
.. We can access the fields of ``m`` using their names and "dotted attribute" notation
.. 
.. .. code-block:: julia
.. 
..     m.a
.. 
.. 
.. .. code-block:: julia
.. 
..     m.b
.. 
.. 
.. .. code-block:: julia
.. 
..     m.σ
.. 
.. 
.. .. code-block:: julia
.. 
..     m.ϕ
.. 
.. 
.. For example, the attribute ``m.ϕ`` points to an instance of ``Beta``, which is in turn a subtype of ``Distribution`` as defined in the Distributions package
.. 
.. .. code-block:: julia
.. 
..     typeof(m.ϕ)
.. 
.. 
.. .. code-block:: julia
.. 
..     m.ϕ isa Distribution
.. 
.. 
.. We can reach into ``m`` and change this if we want to
.. 
.. .. code-block:: julia
.. 
..     m.ϕ = Exponential(0.5)
.. 
.. 
.. .. _spec_field_types:
.. 
.. 
.. Specifying Field Types
.. ^^^^^^^^^^^^^^^^^^^^^^^^^
.. 
.. 
.. In our type definition we can be explicit that we want ``ϕ`` to be a
.. ``Distribution`` and the other elements to be floats
.. 
.. .. code-block:: julia
.. 
..     struct AR1_explicit
..         a::Float64
..         b::Float64
..         σ::Float64
..         ϕ::Distribution
..     end
.. 
.. (In this case, ``mutable`` is removed since we do not intend to make any changes to the elements of ``AR1_explicit``)
.. 
.. Now the constructor will complain if we try to use the wrong data type
.. 
.. .. code-block:: julia
..     :class: no-execute
.. 
..     m = AR1_explicit(0.9, 1, "foo", Beta(5, 5))
.. 
.. 
.. This can be useful in terms of failing early on incorrect data, rather than
.. deeper into execution
.. 
.. At the same time, `AR1_explicit` is not as generic as `AR1`, and hence less flexible
.. 
.. For example, suppose that we want to allow `a`, `b` and `σ` to take any
.. value that is `<: Real`
.. 
.. We could achieve this by the new definition
.. 
.. .. code-block:: julia
.. 
..     struct AR1_real
..         a::Real
..         b::Real
..         σ::Real
..         ϕ::Distribution
..     end
.. 
.. 
.. But it turns out that using abstract types inside user-defined types adversely
.. affects performance --- more about that :doc:`soon <need_for_speed>`
.. 
.. Fortunately, there's another approach that both
.. 
.. * preserves the use of concrete types for internal data and
.. 
.. * allows flexibility across multiple concrete data types
.. 
.. This approach uses *type parameters*, a topic we turn to now
.. 
.. 
.. Type Parameters
.. -------------------
.. 
.. Consider the following output
.. 
.. .. code-block:: julia
.. 
..     typeof([10, 20, 30])
.. 
.. 
.. Here ``Array`` is one of Julia's predefined types (``Array <: DenseArray <: AbstractArray <: Any``)
.. 
.. The ``Int64,1`` in curly brackets are **type parameters**
.. 
.. In this case they are the element type and the dimension
.. 
.. Many other types have type parameters too
.. 
.. .. code-block:: julia
.. 
..     typeof(1.0 + 1.0im)
.. 
.. 
.. .. code-block:: julia
.. 
..     typeof(1 + 1im)
.. 
.. 
.. Types with parameters are therefore in fact an indexed family of types, one for each possible value of the parameter
.. 
.. 
.. We can use parametric types in our own type definitions, as the next example shows
.. 
.. 
.. Back to the AR1 Example
.. -------------------------
.. 
.. Recall our AR(1) example, where we considered different restrictions on internal data
.. 
.. For the coefficients `a`, `b` and `σ`  we considered
.. 
.. * allowing them to be any type
.. 
.. * forcing them to be of type `Float64`
.. 
.. * allowing them to be any `Real`
.. 
.. The last option is a nice balance between specific and flexible
.. 
.. For example, using `Real` in the type definition tells us that, while these values should be scalars, integer values and floats are both OK
.. 
.. However, as mentioned above, using abstract types for fields of user-defined types impacts negatively on performance
.. 
.. For now it suffices to observe that we can achieve flexibility and eliminate
.. abstract types on `a`, `b`, `σ`, and `ϕ` by the following declaration
.. 
.. 
.. .. code-block:: julia
.. 
..     struct AR1_best{T <: Real, D <: Distribution}
..         a::T
..         b::T
..         σ::T
..         ϕ::D
..     end
.. 
.. If we create an instance using `Float64` values and a `Beta` distribution then the instance has type
.. `AR1_best{Float64,Beta}`
.. 
.. It is worth nothing that under this definition, the instance can only be created by
.. providing `a`, `b`, and `σ` of the same type. One could make it flexible enough to
.. parameterize on different values or providing a constructor that converts the inputs
.. to the same type (e.g., using `promote_type`)
.. 
.. .. code-block:: julia
.. 
..     m = AR1_best(0.9, 1.0, 1.0, Beta(5, 5))
.. 
.. 
.. Exercises
.. ===========
.. 
.. 
.. Exercise 1
.. ---------------
.. 
.. Write a function with the signature ``simulate(m::AR1, n::Integer, x0::Real)``
.. that takes as arguments
.. 
.. * an instance ``m`` of ``AR1`` (see above)
.. * an integer ``n``
.. * a real number ``x0``
.. 
.. and returns an array containing a time series of length ``n`` generated according to :eq:`tm_ar1` where
.. 
.. * the primitives of the AR(1) process are as specified in ``m``
.. 
.. * the initial condition :math:`X_0` is set equal to ``x0``
.. 
.. Hint: If ``d`` is an instance of ``Distribution`` then ``rand(d)`` generates one random draw from the distribution specified in ``d``
.. 
.. 
.. Solutions
.. ==========
.. 
.. 
.. Exercise 1
.. ----------
.. 
.. Let's start with the AR1 definition as specified in the lecture
.. .. 
.. .. code-block:: julia
.. 
..     struct AR1_ex1{T <: Real, D <: Distribution}
..         a::T
..         b::T
..         σ::T
..         ϕ::D
..     end
.. 
.. Now let's write the function to simulate AR1s
.. 
.. .. code-block:: julia
.. 
..     function simulate(m::AR1_ex1, n::Integer, x0::Real)
..         X = zeros(n)
..         X[1] = x0
..         for t ∈ 1:(n-1)
..             X[t+1] = m.a * X[t] + m.b + m.σ * rand(m.ϕ)
..         end
..         return X
..     end
.. 
.. 
.. Let's test it out on the AR(1) process discussed in the lecture
.. 
.. .. code-block:: julia
.. 
..     m = AR1_ex1(0.9, 1.0, 1.0, Beta(5, 5))
..     X = simulate(m, 100, 0.0)
.. 
.. 
.. Next let's plot the time series to see what it looks like
.. 
.. .. code-block:: julia
.. 
..     using Plots
..     gr(fmt=:png)
..     plot(X, legend=:none)

