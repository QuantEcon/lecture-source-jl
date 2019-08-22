.. _types_methods:

.. include:: /_static/includes/header.raw

******************************************
The Need for Speed
******************************************


.. contents:: :depth: 2

Overview
============================

Computer scientists often classify programming languages according to the following two categories

*High level languages* aim to maximize productivity by

* being easy to read, write and debug

* automating standard tasks (e.g., memory management)

* being interactive, etc.

*Low level languages* aim for speed and control, which they achieve by

* being closer to the metal (direct access to CPU, memory, etc.)

* requiring a relatively large amount of information from the user (e.g., all data types must be specified)

Traditionally we understand this as a trade off

* high productivity or high performance

* optimized for humans or optimized for machines

One of the great strengths of Julia is that it pushes out the curve, achieving
both high productivity and high performance with relatively little fuss

The word "relatively" is important here, however...

In simple programs, excellent performance is often trivial to achieve

For longer, more sophisticated programs, you need to be aware of potential stumbling blocks

This lecture covers the key points

Requirements
-------------

You should read our :doc:`earlier lecture <../more_julia/generic_programming>` on types, methods and multiple dispatch before this one

Setup
------------------

.. literalinclude:: /_static/includes/deps_generic.jl
     :class: hide-output

.. code-block:: julia

    using LinearAlgebra, Statistics, Compat

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

This is sensible behavior, but if you want to change it there's nothing to stop you

.. code-block:: julia

    import Base: +  # enables adding methods to the + function

    +(x::Integer, y::String) = x + parse(Int, y)

    @show +(100, "100")
    @show 100 + "100";  # equivalent

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

* If it eventually finds a matching method, it invokes that method

* If not, we get an error

This is the process that leads to the following error (since we only added the ``+`` for adding ``Integer`` and ``String`` above)

.. code-block:: julia
    :class: skip-test

    @show (typeof(100.0) <: Integer) == false
    100.0 + "100"

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

    function q(x)  # or q(x::Any)
        println("Default (Any) method invoked")
    end

    function q(x::Number)
        println("Number method invoked")
    end

    function q(x::Integer)
        println("Integer method invoked")
    end

Let's now run this and see how it relates to our discussion of method dispatch
above

.. code-block:: julia

    q(3)

.. code-block:: julia

    q(3.0)

.. code-block:: julia

    q("foo")

Since ``typeof(3) <: Int64 <: Integer <: Number``, the call ``q(3)`` proceeds up the tree to ``Integer`` and invokes ``q(x::Integer)``

On the other hand, ``3.0`` is a ``Float64``, which is not a subtype of  ``Integer``

Hence the call ``q(3.0)`` continues up to ``q(x::Number)``

Finally, ``q("foo")`` is handled by the function operating on ``Any``, since ``String`` is not a subtype of ``Number`` or ``Integer``

Analyzing Function Return Types
-------------------------------------------

For the most part, time spent "optimizing" Julia code to run faster is about ensuring the compiler can correctly deduce types for all functions

The macro ``@code_warntype`` gives us a hint

.. code-block:: julia

    x = [1, 2, 3]
    f(x) = 2x
    @code_warntype f(x)

The ``@code_warntype`` macro compiles ``f(x)`` using the type of ``x`` as an example -- i.e., the ``[1, 2, 3]`` is used as a prototype for analyzing the compilation, rather than simply calculating the value

Here, the ``Body::Array{Int64,1}`` tells us the type of the return value of the
function, when called with types like ``[1, 2, 3]``, is always a vector of integers

In contrast, consider a function potentially returning ``nothing``, as in :doc:`this lecture <../getting_started_julia/fundamental_types>`

.. code-block:: julia

    f(x) = x > 0.0 ? x : nothing
    @code_warntype f(1)

This states that the compiler determines the return type when called with an integer (like ``1``) could be one of two different types, ``Body::Union{Nothing, Int64}``

A final example is a variation on the above, which returns the maximum of ``x`` and ``0``

.. code-block:: julia

    f(x) = x > 0.0 ? x : 0.0
    @code_warntype f(1)

Which shows that, when called with an integer, the type could be that integer or the floating point ``0.0``

On the other hand, if we use change the function to return ``0`` if `x <= 0`, it is type-unstable with  floating point

.. code-block:: julia

    f(x) = x > 0.0 ? x : 0
    @code_warntype f(1.0)

The solution is to use the ``zero(x)`` function which returns the additive identity element of type ``x``

On the other hand, if we change the function to return ``0`` if ``x <= 0``, it is type-unstable with  floating point

.. code-block:: julia

    @show zero(2.3)
    @show zero(4)
    @show zero(2.0 + 3im)

    f(x) = x > 0.0 ? x : zero(x)
    @code_warntype f(1.0)

Foundations
============================

Let's think about how quickly code runs, taking as given

* hardware configuration

* algorithm (i.e., set of instructions to be executed)


We'll start by discussing the kinds of instructions that machines understand


Machine Code
-------------

All instructions for computers end up as *machine code*

Writing fast code --- expressing a given algorithm so that it runs quickly --- boils down to producing efficient machine code

You can do this yourself, by hand, if you want to

Typically this is done by writing `assembly <https://en.wikipedia.org/wiki/Assembly_language>`__, which is a symbolic representation of machine code

Here's some assembly code implementing a function that takes arguments :math:`a, b` and returns :math:`2a + 8b`


.. code-block:: asm
    :class: no-execute

	pushq	%rbp
	movq	%rsp, %rbp
	addq	%rdi, %rdi
	leaq	(%rdi,%rsi,8), %rax
	popq	%rbp
	retq
	nopl	(%rax)

Note that this code is specific to one particular piece of hardware that we use --- different machines require different machine code

If you ever feel tempted to start rewriting your economic model in assembly, please restrain yourself

It's far more sensible to give these instructions in a language like Julia,
where they can be easily written and understood

.. code-block:: julia

    function f(a, b)
        y = 2a + 8b
        return y
    end

or Python

.. code-block:: python
    :class: no-execute

    def f(a, b):
        y = 2 * a + 8 * b
        return y

or even C

.. code-block:: c
    :class: no-execute

    int f(int a, int b) {
        int y = 2 * a + 8 * b;
        return y;
    }

In any of these languages we end up with code that is much easier for humans to write, read, share and debug

We leave it up to the machine itself to turn our code into machine code

How exactly does this happen?


Generating Machine Code
---------------------------


The process for turning high level code into machine code differs across
languages

Let's look at some of the options and how they differ from one another


AOT Compiled Languages
^^^^^^^^^^^^^^^^^^^^^^^

Traditional compiled languages like Fortran, C and C++ are a reasonable option for writing fast code

Indeed, the standard benchmark for performance is still well-written C or Fortran

These languages compile down to efficient machine code because users are forced to provide a lot of detail on data types and how the code will execute

The compiler therefore has ample information for building the corresponding machine code ahead of time (AOT) in a way that

* organizes the data optimally in memory and

* implements efficient operations as required for the task in hand

At the same time, the syntax and semantics of C and Fortran are verbose and unwieldy when compared to something like Julia

Moreover, these low level languages lack the interactivity that's so crucial for scientific work


Interpreted Languages
^^^^^^^^^^^^^^^^^^^^^^

Interpreted languages like Python generate machine code "on the fly", during program execution

This allows them to be flexible and interactive

Moreover, programmers can leave many tedious details to the runtime environment, such as

* specifying variable types

* memory allocation/deallocation, etc.

But all this convenience and flexibility comes at a cost: it's hard to turn
instructions written in these languages into efficient machine code

For example, consider what happens when Python adds a long list of numbers
together

Typically the runtime environment has to check the type of these objects one by one before it figures out how to add them

This involves substantial overheads

There are also significant overheads associated with accessing the data values themselves, which might not be stored contiguously in memory

The resulting machine code is often complex and slow


Just-in-time compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just-in-time (JIT) compilation is an alternative approach that marries some of
the advantages of AOT compilation and interpreted languages

The basic idea is that functions for specific tasks are compiled as requested

As long as the compiler has enough information about what the function does,
it can in principle generate efficient machine code

In some instances, all the information is supplied by the programmer

In other cases, the compiler will attempt to infer missing information on the fly based on usage

Through this approach, computing environments built around JIT compilers aim to

* provide all the benefits of high level languages discussed above and, at the same time,

* produce efficient instruction sets when functions are compiled down to machine code


JIT Compilation in Julia
==========================

JIT compilation is the approach used by Julia

In an ideal setting, all information necessary to generate efficient native machine code is supplied or inferred

In such a setting, Julia will be on par with machine code from low level languages

An Example
--------------

Consider the function

.. code-block:: julia

    function f(a, b)
        y = (a + 8b)^2
        return 7y
    end

Suppose we call ``f`` with integer arguments (e.g., ``z = f(1, 2)``)

The JIT compiler now knows the types of ``a`` and ``b``

Moreover, it can infer types for other variables inside the function

* e.g., ``y`` will also be an integer

It then compiles a specialized version of the function to handle integers and
stores it in memory

We can view the corresponding machine code using the `@code_native` macro

.. code-block:: julia

    @code_native f(1, 2)


If we now call ``f`` again, but this time with floating point arguments, the JIT compiler will once more infer types for the other variables inside the function

* e.g., ``y`` will also be a float

It then compiles a new version to handle this type of argument

.. code-block:: julia

    @code_native f(1.0, 2.0)


Subsequent calls using either floats or integers are now routed to the appropriate compiled code


Potential Problems
---------------------

In some senses, what we saw above was a best case scenario

Sometimes the JIT compiler produces messy, slow machine code

This happens when type inference fails or the compiler has insufficient information to optimize effectively

The next section looks at situations where these problems arise and how to get around them


Fast and Slow Julia Code
==========================

To summarize what we've learned so far, Julia provides a platform for generating highly efficient machine code with relatively little effort by combining

#. JIT compilation

#. Optional type declarations and type inference to pin down the types of variables and hence compile efficient code

#. Multiple dispatch to facilitate specialization and optimization of compiled code for different data types

But the process is not flawless, and hiccups can occur

The purpose of this section is to highlight potential issues and show you how
to circumvent them


BenchmarkTools
------------------

The main Julia package for benchmarking is `BenchmarkTools.jl <https://www.github.com/JuliaCI/BenchmarkTools.jl>`_

Below, we'll use the ``@btime`` macro it exports to evaluate the performance of Julia code

As mentioned in an :doc:`earlier lecture <../more_julia/testing>`, we can also save benchmark results to a file and guard against performance regressions in code

For more, see the package docs

Global Variables
-----------------

Global variables are names assigned to values outside of any function or type definition

The are convenient and novice programmers typically use them with abandon

But global variables are also dangerous, especially in medium to large size programs, since

* they can affect what happens in any part of your program

* they can be changed by any function

This makes it much harder to be certain about what some  small part of a given piece of code actually commands

Here's a `useful discussion on the topic <http://wiki.c2.com/?GlobalVariablesAreBad>`__

When it comes to JIT compilation, global variables create further problems

The reason is that the compiler can never be sure of the type of the global
variable, or even that the type will stay constant while a given function runs

To illustrate, consider this code, where ``b`` is global

.. code-block:: julia

    b = 1.0
    function g(a)
        global b
        for i ∈ 1:1_000_000
            tmp = a + b
        end
    end

The code executes relatively slowly and uses a huge amount of memory

.. code-block:: julia

    using BenchmarkTools

    @btime g(1.0)


If you look at the corresponding machine code you will see that it's a mess

.. code-block:: julia

    @code_native g(1.0)


If we eliminate the global variable like so

.. code-block:: julia

    function g(a, b)
        for i ∈ 1:1_000_000
            tmp = a + b
        end
    end

then execution speed improves dramatically

.. code-block:: julia

    @btime g(1.0, 1.0)

Note that the second run was dramatically faster than the first

That's because the first call included the time for JIT compilaiton

Notice also how small the memory footprint of the execution is

Also, the machine code is simple and clean

.. code-block:: julia

    @code_native g(1.0, 1.0)


Now the compiler is certain of types throughout execution of the function and
hence can optimize accordingly


The ``const`` keyword
^^^^^^^^^^^^^^^^^^^^^^^^^

Another way to stabilize the code above is to maintain the global variable but
prepend it with ``const``

.. code-block:: julia

    const b_const = 1.0
    function g(a)
        global b_const
        for i ∈ 1:1_000_000
            tmp = a + b_const
        end
    end


Now the compiler can again generate efficient machine code

We'll leave you to experiment with it


Composite Types with Abstract Field Types
--------------------------------------------

Another scenario that trips up the JIT compiler is when composite types have
fields with abstract types

We met this issue :ref:`earlier <spec_field_types>`, when we discussed AR(1) models

Let's experiment, using, respectively,

* an untyped field

* a field with abstract type, and

* parametric typing

As we'll see, the last of options these gives us the best performance, while still maintaining significant flexibility

Here's the untyped case

.. code-block:: julia

    struct Foo_generic
        a
    end

Here's the case of an abstract type on the field ``a``

.. code-block:: julia

    struct Foo_abstract
        a::Real
    end

Finally, here's the parametrically typed case

.. code-block:: julia

    struct Foo_concrete{T <: Real}
        a::T
    end

Now we generate instances

.. code-block:: julia

    fg = Foo_generic(1.0)
    fa = Foo_abstract(1.0)
    fc = Foo_concrete(1.0)


In the last case, concrete type information for the fields is embedded in the object

.. code-block:: julia

    typeof(fc)

This is significant because such information is detected by the compiler

Timing
^^^^^^^^^

Here's a function that uses the field ``a`` of our objects

.. code-block:: julia

    function f(foo)
        for i ∈ 1:1_000_000
            tmp = i + foo.a
        end
    end


Let's try timing our code, starting with the generic case:


.. code-block:: julia

    @btime f($fg)

The timing is not very impressive

Here's the nasty looking machine code

.. code-block:: julia

    @code_native f(fg)


The abstract case is similar

.. code-block:: julia

    @btime f($fa)


Note the large memory footprint

The machine code is also long and complex, although we omit details

Finally, let's look at the parametrically typed version

.. code-block:: julia

    @btime f($fc)


Some of this time is JIT compilation, and one more execution gets us down to


Here's the corresponding machine code

.. code-block:: julia

    @code_native f(fc)


Much nicer...


Abstract Containers
----------------------

Another way we can run into trouble is with abstract container types

Consider the following function, which essentially does the same job as Julia's ``sum()`` function but acts only on floating point data


.. code-block:: julia

    function sum_float_array(x::AbstractVector{<:Number})
        sum = 0.0
        for i ∈ eachindex(x)
            sum += x[i]
        end
        return sum
    end


Calls to this function run very quickly

.. code-block:: julia

    x = range(0,  1, length = Int(1e6))
    x = collect(x)
    typeof(x)


.. code-block:: julia

    @btime sum_float_array($x)


When Julia compiles this function, it knows that the data passed in as ``x`` will be an array of 64 bit floats

Hence it's known to the compiler that the relevant method for ``+`` is always addition of floating point numbers

Moreover, the data can be arranged into continuous 64 bit blocks of memory to simplify memory access

Finally, data types are stable --- for example, the local variable ``sum`` starts off as a float and remains a float throughout


Type Inferences
^^^^^^^^^^^^^^^^^^^

Here's the same function minus the type annotation in the function signature

.. code-block:: julia

    function sum_array(x)
        sum = 0.0
        for i ∈ eachindex(x)
            sum += x[i]
        end
        return sum
    end

When we run it with the same array of floating point numbers it executes at a
similar speed as the function with type information

.. code-block:: julia

    @btime sum_array($x)


The reason is that when ``sum_array()`` is first called on a vector of a given
data type, a newly compiled version of the function is produced to handle that
type

In this case, since we're calling the function on a vector of floats, we get a compiled version of the function with essentially the same internal representation as ``sum_float_array()``

An Abstract Container
^^^^^^^^^^^^^^^^^^^^^^^^^


Things get tougher for the interpreter when the data type within the array is imprecise

For example, the following snippet creates an array where the element type is ``Any``

.. code-block:: julia

    x = Any[ 1/i for i ∈ 1:1e6 ];

.. code-block:: julia

    eltype(x)


Now summation is much slower and memory management is less efficient

.. code-block:: julia

    @btime sum_array($x)


Further Comments
===================


Here are some final comments on performance


Explicit Typing
----------------

Writing fast Julia code amounts to writing Julia from which the compiler can
generate efficient machine code

For this, Julia needs to know about the type of data it's processing as early as possible

We could hard code the type of all variables and function arguments but this comes at a cost

Our code becomes more cumbersome and less generic

We are starting to loose the advantages that drew us to Julia in the first place

Moreover, explicitly typing everything is not necessary for optimal performance

The Julia compiler is smart and can often infer types perfectly well, without
any performance cost

What we really want to do is

* keep our code simple, elegant and generic

* help the compiler out in situations where it's liable to get tripped up

Summary and Tips
-----------------------

Use functions to segregate operations into logically distinct blocks

Data types will be determined at function boundaries

If types are not supplied then they will be inferred

If types are stable and can be inferred effectively your functions will run fast


Further Reading
-----------------------


A good next stop for further reading is the `relevant part <https://docs.julialang.org/en/v1/manual/performance-tips/>`_ of the Julia documentation


.. http://www.informit.com/articles/article.aspx?p=1215438&seqNum=2


.. http://stackoverflow.com/questions/10268028/julia-compiles-the-script-everytime


.. This is how Julia gets good performance even when code is written without type annotations: if you call f(1) you get code specialized for Int64 — the type of 1 on 64-bit systems; if you call f(1.0) you get a newly jitted version that is specialized for Float64 — the type of 1.0 on all systems. Since each compiled version of the function knows what types it will be getting, it can run at C-like speed. You can sabotage this by writing and using "type-unstable" functions whose return type depends on run-time data, rather than just types, but we've taken great care not to do that in designing the core language and standard library.


.. http://scientopia.org/blogs/goodmath/2014/02/04/everyone-stop-implementing-programming-languages-right-now-its-been-solved/


.. ADD EXERCISES
