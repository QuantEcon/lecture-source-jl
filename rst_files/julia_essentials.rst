.. _julia_essentials:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Julia Essentials
******************************************

.. contents:: :depth: 2

Having covered a few examples, let's now turn to a more systematic exposition
of the essential features of the language

Overview
============

Topics:

* Common data types
* Basic file I/O
* Iteration
* More on user-defined functions
* Comparisons and logic


Common Data Types
======================


Like most languages, Julia language defines and provides functions for operating on standard data types such as

* integers
* floats
* strings
* arrays, etc...

Let's learn a bit more about them

Primitive Data Types
-------------------------

A particularly simple data type is a Boolean value, which can be either ``true`` or
``false``

Activate the project environment, ensuring that ``Project.toml`` and ``Manifest.toml`` are in the same location as your notebook

.. code-block:: julia

    using Pkg; Pkg.activate(@__DIR__); #activate environment in the notebook's location

.. code-block:: julia

    x = true


.. code-block:: julia

    typeof(x)


.. code-block:: julia

    y = 1 > 2  # Now y = false


Under addition, ``true`` is converted to ``1`` and ``false`` is converted to ``0``

.. code-block:: julia

    true + false


.. code-block:: julia

    sum([true, false, false, true])


The two most common data types used to represent numbers are integers and
floats

(Computers distinguish between floats and integers because arithmetic is
handled in a different way)

.. code-block:: julia

    typeof(1.0)


.. code-block:: julia

    typeof(1)


If you're running a 32 bit system you'll still see ``Float64``, but you will see ``Int32`` instead of ``Int64`` (see `the section on Integer types <https://docs.julialang.org/en/stable/manual/integers-and-floating-point-numbers/#Integers-1>`_ from the Julia manual)

Arithmetic operations are fairly standard


.. code-block:: julia

    x = 2; y = 1.0


.. code-block:: julia

    x * y


.. code-block:: julia

    x^2


.. code-block:: julia

    y / x


Although the ``*`` can be omitted for multiplication between a numeric literal and a variable


.. code-block:: julia

    2x - 3y


Also, you can use function (instead of infix) notation if you so desire

.. code-block:: julia

    +(10, 20)


.. code-block:: julia

    *(10, 20)


Complex numbers are another primitive data type, with the imaginary part being specified by ``im``


.. code-block:: julia

    x = 1 + 2im


.. code-block:: julia

    y = 1 - 2im


.. code-block:: julia

    x * y  # Complex multiplication


There are several more primitive data types that we'll introduce as necessary


Strings
----------

A string is a data type for storing a sequence of characters


.. code-block:: julia

    x = "foobar"


.. code-block:: julia

    typeof(x)


You've already seen examples of Julia's simple string formatting operations

.. code-block:: julia

    x = 10; y = 20


.. code-block:: julia

    "x = $x"


.. code-block:: julia

    "x + y = $(x + y)"


To concatenate strings use ``*``

.. code-block:: julia

    "foo" * "bar"


Julia provides many functions for working with strings

.. code-block:: julia

    s = "Charlie don't surf"


.. code-block:: julia

    split(s)


.. code-block:: julia

    replace(s, "surf" => "ski")


.. code-block:: julia

    split("fee,fi,fo", ",")


.. code-block:: julia

    strip(" foobar ")  # Remove whitespace


Julia can also find and replace using `regular expressions <https://en.wikipedia.org/wiki/Regular_expression>`_ (`see the documentation <https://docs.julialang.org/en/stable/manual/strings/#Regular-Expressions-1>`_ on regular expressions for more info)

.. code-block:: julia

    match(r"(\d+)", "Top 10")  # Find digits in string


Containers
--------------

Julia has several basic types for storing collections of data

We have already discussed arrays

A related data type is **tuples**, which can act like "immutable" arrays

.. code-block:: julia

    x = ("foo", "bar")


.. code-block:: julia

    typeof(x)


An immutable object is one that cannot be altered once it resides in memory

In particular, tuples do not support item assignment:

.. code-block:: julia
    :class: no-execute

    x[1] = 42


This is similar to Python, as is the fact that the parenthesis can be omitted


.. code-block:: julia

    x = "foo", "bar"


Another similarity with Python is tuple unpacking, which means that the
following convenient syntax is valid


.. code-block:: julia

    x = ("foo", "bar")


.. code-block:: julia

    word1, word2 = x


.. code-block:: julia

    word1


.. code-block:: julia

    word2


Referencing Items
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The last element of a sequence type can be accessed with the keyword ``end``

.. code-block:: julia

    x = [10, 20, 30, 40]


.. code-block:: julia

    x[end]


.. code-block:: julia

    x[end-1]


To access multiple elements of an array or tuple, you can use slice notation

.. code-block:: julia

    x[1:3]


.. code-block:: julia

    x[2:end]


The same slice notation works on strings

.. code-block:: julia

    "foobar"[3:end]


Dictionaries
^^^^^^^^^^^^^^^^^^^^^^^^^^

Another container type worth mentioning is dictionaries

Dictionaries are like arrays except that the items are named instead of numbered

.. code-block:: julia

    d = Dict("name" => "Frodo", "age" => 33)


.. code-block:: julia

    d["age"]


The strings ``name`` and ``age`` are called the **keys**

The objects that the keys are mapped to (``"Frodo"`` and ``33``) are called the **values**

They can be accessed via ``keys(d)`` and ``values(d)`` respectively


Input and Output
====================


Let's have a quick look at reading from and writing to text files

We'll start with writing


.. code-block:: julia

    f = open("newfile.txt", "w")  # "w" for writing


.. code-block:: julia

    write(f, "testing\n")         # \n for newline


.. code-block:: julia

    write(f, "more testing\n")


.. code-block:: julia

    close(f)

The effect of this is to create a file called ``newfile.txt`` in your present
working directory with contents


We can read the contents of ``newline.txt`` as follows

.. code-block:: julia

    f = open("newfile.txt", "r")  # Open for reading


.. code-block:: julia

    print(read(f, String))


.. code-block:: julia

    close(f)


.. TODO Should we include an example of the ``open("newfile.txt", "r") do f`` syntax here?
.. I personally like it because it manages closing the file for me and feels a
.. lot like python context managers (which I also like)

Often when reading from a file we want to step through the lines of a file, performing an action on each one

There's a neat interface to this in Julia, which takes us to our next topic


.. _iterating_version_1:

Iterating
========================

One of the most important tasks in computing is stepping through a
sequence of data and performing a given action

Julia's provides neat, flexible tools for iteration as we now discuss

Iterables
----------------

An iterable is something you can put on the right hand side of ``for`` and loop over

These include sequence data types like arrays

.. code-block:: julia

    let
        actions = ["surf", "ski"]
        for action ∈ actions
            println("Charlie don't $action")
        end
    end


They also include so-called **iterators**

You've already come across these types of objects

.. code-block:: julia

    let
        for i ∈ 1:3
            print(i)
        end
    end


If you ask for the keys of dictionary you get an iterator


.. code-block:: julia

    d = Dict("name" => "Frodo", "age" => 33)


.. code-block:: julia

    keys(d)


This makes sense, since the most common thing you want to do with keys is loop over them

The benefit of providing an iterator rather than an array, say, is that the former is more memory efficient

Should you need to transform an iterator into an array you can always use ``collect()``


.. code-block:: julia

    collect(keys(d))


Looping without Indices
------------------------

You can loop over sequences without explicit indexing, which often leads to
neater code

For example compare

.. code-block:: julia

    x_values = range(0, stop = 3, length = 10)

.. code-block:: julia

    for x ∈ x_values
        println(x * x)
    end


.. code-block:: julia

    for i ∈ eachindex(x_values)
        println(x_values[i] * x_values[i])
    end


Julia provides some functional-style helper functions (similar to Python) to facilitate looping without indices

One is ``zip()``, which is used for stepping through pairs from two sequences

For example, try running the following code

.. code-block:: julia

    let
        countries = ("Japan", "Korea", "China")
        cities = ("Tokyo", "Seoul", "Beijing")
        for (country, city) ∈ zip(countries, cities)
            println("The capital of $country is $city")
        end
    end


If we happen to need the index as well as the value, one option is to use ``enumerate()``

The following snippet will give you the idea

.. code-block:: julia

    let
        countries = ("Japan", "Korea", "China")
        cities = ("Tokyo", "Seoul", "Beijing")
        for (i, country) ∈ enumerate(countries)
            city = cities[i]
            println("The capital of $country is $city")
        end
    end


Comprehensions
------------------

Comprehensions are an elegant tool for creating new arrays or dictionaries from iterables

Here's some examples

.. code-block:: julia

    doubles = [ 2i for i ∈ 1:4 ]


.. code-block:: julia

    animals = ["dog", "cat", "bird"];   # Semicolon suppresses output


.. code-block:: julia

    plurals = [ animal * "s" for animal ∈ animals ]


.. code-block:: julia

    [ i + j for i ∈ 1:3, j ∈ 4:6 ]


.. code-block:: julia

    [ i + j + k for i ∈ 1:3, j ∈ 4:6, k ∈ 7:9 ]


The same kind of expression works for dictionaries

.. code-block:: julia

    Dict(string(i) => i for i ∈ 1:3)


Comparisons and Logical Operators
===================================

Comparisons
---------------------------------

As we saw earlier, when testing for equality we use ``==``


.. code-block:: julia

    x = 1


.. code-block:: julia

    x == 2


For "not equal" use ``!=`` or ``≠``

.. code-block:: julia

    x ≠ 3


We can chain inequalities:


.. code-block:: julia

    1 < 2 < 3


.. code-block:: julia

    1 ≤ 2 ≤ 3


In many languages you can use integers or other values when testing conditions but Julia is more fussy

.. code-block:: julia
    :class: no-execute

    while 0 println("foo") end


.. code-block:: julia
    :class: no-execute

    if 1 print("foo") end


Combining Expressions
------------------------

Here are the standard logical connectives (conjunction, disjunction)


.. code-block:: julia

    true && false


.. code-block:: julia

    true || false


Remember

* ``P && Q`` is ``true`` if both are ``true``, otherwise it's ``false``

* ``P || Q`` is ``false`` if both are ``false``, otherwise it's ``true``


User-Defined Functions
========================


Let's talk a little more about user-defined functions


User-defined functions are important for improving the clarity of your code by

* separating different strands of logic

* facilitating code reuse (writing the same thing twice is always a bad idea)


Julia functions are convenient:

* Any number of functions can be defined in a given file

* Any "value" can be passed to a function as an argument, including other functions

* Functions can be (and often are) defined inside other functions

* A function can return any kind of value, including functions

We'll see many examples of these structures in the following lectures


For now let's just cover some of the different ways of defining functions


Return Statement
------------------

In Julia, the ``return`` statement is optional, so that the following functions
have identical behavior

.. code-block:: julia

    function f1(a, b)
        return a * b
    end

    function f2(a, b)
        a * b
    end

When no return statement is present, the last value obtained when executing the code block is returned

Although some prefer the second option, we often favor the former on the basis that explicit is better than implicit

A function can have arbitrarily many ``return`` statements, with execution terminating when the first return is hit

You can see this in action when experimenting with the following function


.. code-block:: julia

    function foo(x)
        if x > 0
            return "positive"
        end
        return "nonpositive"
    end



Other Syntax for Defining Functions
--------------------------------------

For short function definitions Julia offers some attractive simplified syntax

First, when the function body is a simple expression, it can be defined
without the ``function`` keyword or ``end``

.. code-block:: julia

    ff(x) = sin(1 / x)


Let's check that it works


.. code-block:: julia

    ff(1 / pi)


Julia also allows for you to define anonymous functions

For example, to define ``f(x) = sin(1 / x)`` you can use ``x -> sin(1 / x)``

The difference is that the second function has no name bound to it

How can you use a function with no name?

Typically it's as an argument to another function

.. code-block:: julia

    map(x -> sin(1 / x), randn(3))  # Apply function to each element


Optional and Keyword Arguments
------------------------------------

Function arguments can be given default values

.. code-block:: julia

    fff(x, a = 1) = exp(cos(a * x))

If the argument is not supplied the default value is substituted

.. code-block:: julia

    fff(pi)


.. code-block:: julia

    fff(pi, 2)


Another option is to use **keyword** arguments

The difference between keyword and standard (positional) arguments is that
they are parsed and bound by name rather than order in the function call

For example, in the call

.. code-block:: julia
    :class: no-execute

    simulate(param1, param2, max_iterations=100, error_tolerance=0.01)

the last two arguments are keyword arguments and their order is irrelevant (as
long as they come after the positional arguments)

To define a function with keyword arguments you need to use ``;`` like so

.. code-block:: julia
    :class: no-execute

    function simulate_kw(param1, param2;
                         max_iterations = 100,
                         error_tolerance = 0.01)
        # Function body here
    end


Vectorized Functions
====================


A common scenario in computing is that

* we have a function ``f`` such that ``f(x)`` returns a number for any number ``x``

* we wish to apply ``f`` to every element of a vector ``x_vec`` to produce a new vector ``y_vec``

In Julia loops are fast and we can do this easily enough with a loop

For example, suppose that we want to apply ``sin`` to ``x_vec = [2.0, 4.0, 6.0, 8.0]``

The following code will do the job

.. code-block:: julia

    x_vec = [2.0, 4.0, 6.0, 8.0]
    y_vec = similar(x_vec)
    for (i, x) ∈ enumerate(x_vec)
        y_vec[i] = sin(x)
    end

But this is a bit unwieldy so Julia offers the alternative syntax

.. code-block:: julia

    y_vec = sin.(x_vec)

More generally, if ``f`` is any Julia function, then ``f.`` references the vectorized version

Conveniently, this applies to user-defined functions as well

To illustrate, let's write a function ``chisq`` such that ``chisq(k)`` returns a chi-squared random variable with ``k`` degrees of freedom when ``k`` is an integer

In doing this we'll exploit the fact that, if we take ``k`` independent standard normals, square them all and sum, we get a chi-squared with ``k`` degrees of freedom


.. code-block:: julia

    function chisq(k::Integer)
        k > 0 || throw(ArgumentError("$k must be a natural number"))
        z = randn(k)
        return sum(z -> z^2, z) # same as `sum(x^2 for x ∈ z)`
    end


.. code-block:: julia

    chisq(3)


Note that calls with integers less than 1 will trigger an assertion failure inside
the function body

.. code-block:: julia
    :class: no-execute

    chisq(-2)


Let's try this out on an array of integers, adding the vectorized notation

.. code-block:: julia

    chisq.([2, 4, 6])


Exercises
============


.. _pyess_ex1:

Exercise 1
---------------

Part 1: Given two numeric arrays or tuples ``x_vals`` and ``y_vals`` of equal length, compute
their inner product using ``zip()``

Part 2: Using a comprehension, count the number of even numbers between 0 and 99

* Hint: ``iseven`` returns ``true`` for even numbers and ``false`` for odds.

Part 3: Using a comprehension, take ``pairs = ((2, 5), (4, 2), (9, 8), (12, 10))`` and count the number of pairs ``(a, b)`` such that both ``a`` and ``b`` are even


.. _pyess_ex2:

Exercise 2
------------

Consider the polynomial

.. math::
    :label: polynom0

    p(x)
    = a_0 + a_1 x + a_2 x^2 + \cdots a_n x^n
    = \sum_{i=0}^n a_i x^i


Using ``enumerate()`` in your loop, write a function ``p`` such that ``p(x, coeff)`` computes the value in :eq:`polynom0` given a point ``x`` and an array of coefficients ``coeff``


.. _pyess_ex3:

Exercise 3
--------------

Write a function that takes a string as an argument and returns the number of capital letters in the string

Hint: ``uppercase("foo")`` returns ``"FOO"``


.. _pyess_ex4:

Exercise 4
------------

Write a function that takes two sequences ``seq_a`` and ``seq_b`` as arguments and
returns ``true`` if every element in ``seq_a`` is also an element of ``seq_b``, else
``false``

* By "sequence" we mean an array, tuple or string


.. _pyess_ex5:

Exercise 5
------------

The Julia libraries include functions for interpolation and approximation

Nevertheless, let's write our own function approximation routine as an exercise

In particular, write a function ``linapprox`` that takes as arguments

* A function ``f`` mapping some interval :math:`[a, b]` into :math:`\mathbb R`

* two scalars ``a`` and ``b`` providing the limits of this interval

* An integer ``n`` determining the number of grid points

* A number ``x`` satisfying ``a ≤ x ≤ b``

and returns the `piecewise linear interpolation <https://en.wikipedia.org/wiki/Linear_interpolation>`_ of ``f`` at ``x``, based on ``n`` evenly spaced grid points ``a = point[1] < point[2] < ... < point[n] = b``

Aim for clarity, not efficiency

Exercise 6
---------------------------------

The following data lists US cities and their populations

Copy this text into a text file called ``us_cities.txt`` and save it in your present working directory

* That is, save it in the location Julia returns when you call ``pwd()``

This can also be achieved by running the following Julia code:

.. code-block:: julia

    open("us_cities.txt", "w") do f
      write(f,
    "new york: 8244910
    los angeles: 3819702
    chicago: 2707120
    houston: 2145146
    philadelphia: 1536471
    phoenix: 1469471
    san antonio: 1359758
    san diego: 1326179
    dallas: 1223229")
    end


Write a program to calculate total population across these cities

Hints:

* If ``f`` is a file object then ``eachline(f)`` provides an iterable that steps you through the lines in the file

* ``parse(Int, "100")`` converts the string ``"100"`` into an integer


Solutions
==========

Exercise 1
----------

Part 1 solution:

Here's one possible solution

.. code-block:: julia

    x_vals = [1, 2, 3]
    y_vals = [1, 1, 1]
    sum(x * y for (x, y) ∈ zip(x_vals, y_vals))


Part 2 solution:

One solution is

.. code-block:: julia

    sum(iseven, 0:99)


Part 3 solution:

Here's one possibility

.. code-block:: julia

    pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
    sum(xy -> all(iseven, xy), pairs)


Exercise 2
----------

.. code-block:: julia

    p(x, coeff) = sum(a * x^(i-1) for (i, a) ∈ enumerate(coeff))


.. code-block:: julia

    p(1, (2, 4))


Exercise 3
----------

Here's one solutions:

.. code-block:: julia

    function f_ex3(string)
        count = 0
        for letter ∈ string
            if (letter == uppercase(letter)) && isletter(letter)
                count += 1
            end
        end
        return count
    end

    f_ex3("The Rain in Spain")


Exercise 4
----------

Here's one solutions:

.. code-block:: julia

    function f_ex4(seq_a, seq_b)
        is_subset = true
        for a ∈ seq_a
            if a ∉ seq_b
                is_subset = false
            end
        end
        return is_subset
    end

    # == test == #

    println(f_ex4([1, 2], [1, 2, 3]))
    println(f_ex4([1, 2, 3], [1, 2]))


if we use the `Set` data type then the solution is easier

.. code-block:: julia

    f_ex4_2(seq_a, seq_b) = Set(seq_a) ⊆ Set(seq_b) # \subseteq (⊆) is unicode for `issubset`

    println(f_ex4_2([1, 2], [1, 2, 3]))
    println(f_ex4_2([1, 2, 3], [1, 2]))


Exercise 5
----------

.. code-block:: julia

    function linapprox(f, a, b, n, x)
        #=
        Evaluates the piecewise linear interpolant of f at x on the interval
        [a, b], with n evenly spaced grid points.

        =#
        length_of_interval = b - a
        num_subintervals = n - 1
        step = length_of_interval / num_subintervals

        # === find first grid point larger than x === #
        point = a
        while point ≤ x
            point += step
        end

        # === x must lie between the gridpoints (point - step) and point === #
        u, v = point - step, point

        return f(u) + (x - u) * (f(v) - f(u)) / (v - u)
    end

Let's test it

.. code-block:: julia

    f_ex5(x) = x^2
    g_ex5(x) = linapprox(f_ex5, -1, 1, 3, x)


.. code-block:: julia

    using Plots


.. code-block:: julia

    let
        x_grid = range(-1, stop = 1, length = 100)
        y_vals = f_ex5.(x_grid)
        y_approx = g_ex5.(x_grid)
        plot(x_grid, y_vals, label = "true")
        plot!(x_grid, y_approx, label = "approximation")
    end


Exercise 6
----------

.. code-block:: julia

    let
        f_ex6 = open("us_cities.txt", "r")
        total_pop = 0
        for line ∈ eachline(f_ex6)
            city, population = split(line, ':')            # Tuple unpacking
            total_pop += parse(Int, population)
        end
        close(f_ex6)
        println("Total population = $total_pop")
    end
