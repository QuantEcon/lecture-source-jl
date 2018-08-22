.. _julia_by_example:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
An Introductory Example
******************************************

.. contents:: :depth: 2

Overview 
==================


We're now ready to start learning the Julia language itself


Level
-------

Our approach is aimed at those who already have at least some knowledge of programming --- perhaps experience with Python, MATLAB, R, C or similar

In particular, we assume you have some familiarity with fundamental programming concepts such as

* variables

* loops

* conditionals (if/else)

If you have no such programming experience, then one option is to try Python first

Python is a great first language and there are many introductory treatments

Otherwise, just dive in and see how you go...


Approach
-------------

In this lecture we will write and then pick apart small Julia programs

At this stage the objective is to introduce you to basic syntax and data structures

Deeper concepts---how things work---will be covered in later lectures

Since we are looking for simplicity the examples are a little contrived

Set Up
--------

We assume that you've worked your way through :doc:`our getting started lecture <getting_started>` already

For this lecture, we recommend that you work in a Jupyter notebook, as described :ref:`here <jl_jupyter>`



Other References
--------------------

The definitive reference is `Julia's own documentation <https://docs.julialang.org/en/stable/>`_

The manual is thoughtfully written but also quite dense (and somewhat evangelical)

The presentation in this and our remaining lectures is more of a tutorial style based around examples



Example: Plotting a White Noise Process
================================================


To begin, let's suppose that we want to simulate and plot the white noise
process :math:`\epsilon_0, \epsilon_1, \ldots, \epsilon_T`, where each draw :math:`\epsilon_t` is independent standard normal 

In other words, we want to generate figures that look something like this:

.. figure:: /_static/figures/test_program_1.png
   :scale: 100%

This is straightforward using `Plots.jl`, which was discussed in our :doc:`set up lecture <getting_started>`

Fire up a :ref:`Jupyter notebook <jl_jupyter>` and enter the following in a cell

.. code-block:: julia

    using Plots
    ts_length = 100
    ϵ_values = randn(ts_length)
    plot(ϵ_values, color="blue")



Let's break this down and see how it works


.. _import:

Importing Functions
---------------------


The effect of the statement ``using Plots`` is to make all the names exported by the ``Plots`` module available in the global scope

If you prefer to be more selective you can replace ``using Plots`` with ``import Plots: plot``

Now only the ``plot`` function is accessible

Since our program uses only the plot function from this module, either would have worked in the previous example


Arrays
--------

The function call ``ϵ_values = randn(ts_length)`` creates one of the
most fundamental Julia data types: an array


.. code-block:: julia

    typeof(ϵ_values)




.. code-block:: julia

    ϵ_values





The information from ``typeof()`` tells us that ``ϵ_values`` is an array of 64 bit floating point values, of dimension 1
 
Julia arrays are quite flexible --- they can store heterogeneous data for example

.. code-block:: julia

    x = [10, "foo", false]

 

Notice now that the data type is recorded as ``Any``, since the array contains mixed data

The first element of ``x`` is an integer

.. code-block:: julia

    typeof(x[1])



The second is a string


.. code-block:: julia

    typeof(x[2])



The third is the boolean value ``false``

.. code-block:: julia

    typeof(x[3])



Notice from the above that 

* array indices start at 1 (unlike Python, where arrays are zero-based)

* array elements are referenced using square brackets (unlike MATLAB and Fortran)

Julia contains many functions for acting on arrays --- we'll review them later

For now here's several examples, applied to the same list ``x = [10, "foo", false]``


.. code-block:: julia

    length(x)





.. code-block:: julia

    pop!(x)




.. code-block:: julia

    x





.. code-block:: julia

    push!(x, "bar")







.. code-block:: julia

    x






The first example just returns the length of the list

The second, ``pop!()``, pops the last element off the list and returns it

In doing so it changes the list (by dropping the last element)

Because of this we call ``pop!`` a **mutating method**

It's conventional in Julia that mutating methods end in ``!`` to remind the user that the function has other effects beyond just returning a value

The function ``push!()`` is similar, except that it appends its second argument to the array


For Loops
---------------

Although there's no need in terms of what we wanted to achieve with our
program, for the sake of learning syntax let's rewrite our program to use a
``for`` loop


.. code-block:: julia

    ts_length = 100
    ϵ_values = Array{Float64}(ts_length)
    for i in 1:ts_length
        ϵ_values[i] = randn()
    end
    plot(ϵ_values, color="blue")
    


Here we first declared ``ϵ_values`` to be an empty array for storing 64 bit floating point numbers

The ``for`` loop then populates this array by successive calls to ``randn()``

* Called without an argument, ``randn()`` returns a single float


Like all code blocks in Julia, the end of the ``for`` loop code block (which is just one line here) is indicated by the keyword ``end``

The word ``in`` from the ``for`` loop can be replaced by symbol ``=``

The expression ``1:ts_length`` creates an **iterator** that is looped over --- in this case the integers from ``1`` to ``ts_length``

Iterators are memory efficient because the elements are generated on the fly rather than stored in memory

In Julia you can also loop directly over arrays themselves, like so

.. code-block:: julia

    words = ["foo", "bar"]
    for word in words
        println("Hello $word")
    end




While Loops
---------------------

The syntax for the while loop contains no surprises


.. code-block:: julia

    ts_length = 100
    ϵ_values = Array{Float64}(ts_length)
    i = 1
    while i <= ts_length
        ϵ_values[i] = randn()
        i = i + 1
    end
    plot(ϵ_values, color="blue")



The next example does the same thing with a condition and the ``break``
statement



.. code-block:: julia

    ts_length = 100
    ϵ_values = Array{Float64}(ts_length)
    i = 1
    while true
        ϵ_values[i] = randn()
        i = i + 1
        if i > ts_length
            break
        end
    end
    plot(ϵ_values, color="blue")
    


.. _user_defined_functions:

User-Defined Functions
----------------------------

For the sake of the exercise, let's now go back to the ``for`` loop but restructure our program so that generation of random variables takes place within a user-defined function

.. code-block:: julia

    function generate_data(n)
        ϵ_values = Array{Float64}(n)
        for i = 1:n
            ϵ_values[i] = randn()
        end
        return ϵ_values
    end
    
    ts_length = 100
    data = generate_data(ts_length)
    plot(data, color="blue")


     
Here 

* ``function`` is a Julia keyword that indicates the start of a function definition

* ``generate_data`` is an arbitrary name for the function

* ``return`` is a keyword indicating the return value

A Slightly More Useful Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Of course the function ``generate_data`` is completely contrived

We could just write the following and be done

.. code-block:: julia

    ts_length = 100
    data = randn(ts_length)
    plot(data, color="blue")
    

Let's make a slightly more useful function 

This function will be passed a choice of probability distribution and respond by plotting a histogram of observations

In doing so we'll make use of the Distributions package


.. code-block:: julia

    Pkg.add("Distributions")

    
Here's the code

.. code-block:: julia

    using Distributions
    
    function plot_histogram(distribution, n)
        ϵ_values = rand(distribution, n)  # n draws from distribution
        histogram(ϵ_values)
    end
    
    lp = Laplace()
    plot_histogram(lp, 500)

The resulting figure looks like this



Let's have a casual discussion of how all this works while leaving technical details for later in the lectures

First, ``lp = Laplace()`` creates an instance of a data type defined
in the Distributions module that represents the Laplace distribution

The name ``lp`` is bound to this object

When we make the function call ``plot_histogram(lp, 500)`` the code in the body
of the function ``plot_histogram`` is run with

* the name ``distribution`` bound to the same object as ``lp``

* the name ``n`` bound to the integer ``500``

A Mystery
^^^^^^^^^^^

Now consider the function call ``rand(distribution, n)``

This looks like something of a mystery

The function ``rand()`` is defined in the base library such that ``rand(n)`` returns ``n`` uniform random variables on :math:`[0, 1)`

.. code-block:: julia

    rand(3)






On the other hand, ``distribution`` points to a data type representing the Laplace distribution that has been defined in a third party package

So how can it be that ``rand()`` is able to take this kind of object as an
argument and return the output that we want?

The answer in a nutshell is **multiple dispatch**

This refers to the idea that functions in Julia can have different behavior
depending on the particular arguments that they're passed

Hence in Julia we can take an existing function and give it a new behavior by defining how it acts on a new type of object

The interpreter knows which function definition to apply in a given setting by looking at the types of the objects the function is called on

In Julia these alternative versions of a function are called **methods**




Exercises
===============

.. _jbe_ex1:

Exercise 1
-----------------

Recall that :math:`n!` is read as ":math:`n` factorial" and defined as
:math:`n! = n \times (n - 1) \times \cdots \times 2 \times 1`

In Julia you can compute this value with ``factorial(n)``

Write your own version of this function, called ``factorial2``, using a ``for`` loop 



.. _jbe_ex2:

Exercise 2
--------------

The `binomial random variable <https://en.wikipedia.org/wiki/Binomial_distribution>`_ :math:`Y \sim Bin(n, p)` represents

* number of successes in :math:`n` binary trials

* each trial succeeds with probability :math:`p`

Using only ``rand()`` from the set of Julia's built-in random number
generators (not the Distributions package), write a function ``binomial_rv`` such that ``binomial_rv(n, p)`` generates one draw of :math:`Y`

Hint: If :math:`U` is uniform on :math:`(0, 1)` and :math:`p \in (0,1)`, then the expression ``U < p`` evaluates to ``true`` with probability :math:`p`




.. _jbe_ex3:

Exercise 3
--------------

Compute an approximation to :math:`\pi` using Monte Carlo

For random number generation use only ``rand()``

Your hints are as follows:

* If :math:`U` is a bivariate uniform random variable on the unit square :math:`(0, 1)^2`, then the probability that :math:`U` lies in a subset :math:`B` of :math:`(0,1)^2` is equal to the area of :math:`B`

* If :math:`U_1,\ldots,U_n` are iid copies of :math:`U`, then, as :math:`n` gets large, the fraction that falls in :math:`B` converges to the probability of landing in :math:`B`

* For a circle, area = π * radius^2




.. _jbe_ex4:

Exercise 4
--------------

Write a program that prints one realization of the following random device:

* Flip an unbiased coin 10 times
* If 3 consecutive heads occur one or more times within this sequence, pay one dollar
* If not, pay nothing

Once again use only ``rand()`` as your random number generator



.. _jbe_ex5:

Exercise 5
----------------------------------

Simulate and plot the correlated time series

.. math::

    x_{t+1} = \alpha \, x_t + \epsilon_{t+1}
    \quad \text{where} \quad
    x_0 = 0 
    \quad \text{and} \quad t = 0,\ldots,T


The sequence of shocks :math:`\{\epsilon_t\}` is assumed to be iid and standard normal

Set :math:`T=200` and :math:`\alpha = 0.9`




.. _jbe_ex6:

Exercise 6
----------------------------------

Plot three simulated time series, one for each of the cases :math:`\alpha=0`, :math:`\alpha=0.8` and :math:`\alpha=0.98`

In particular, you should produce (modulo randomness) a figure that looks as follows


(The figure illustrates how time series with the same one-step-ahead conditional volatilities, as these three processes have, can have very different unconditional volatilities)



Solutions
=========

Exercise 1
----------

.. code-block:: julia

    function factorial2(n)
        k = 1
        for i in 1:n
            k = k * i
        end
        return k
    end
    
    factorial2(4)






.. code-block:: julia

    factorial(4)  # Built-in function







Exercise 2
----------

.. code-block:: julia

    function binomial_rv(n, p)
        count = 0
        U = rand(n)
        for i in 1:n
            if U[i] < p
                count = count + 1    # Or count += 1
            end
        end
        return count
    end
    
    for j in 1:25
        b = binomial_rv(10, 0.5)
        print("$b, ")
    end



Exercise 3
----------

Consider the circle of diameter 1 embedded in the unit square

Let :math:`A` be its area and let :math:`r=1/2` be its radius

If we know :math:`\pi` then we can compute :math:`A` via
:math:`A = \pi r^2`

But here the point is to compute :math:`\pi`, which we can do by
:math:`\pi = A / r^2`

Summary: If we can estimate the area of the unit circle, then dividing
by :math:`r^2 = (1/2)^2 = 1/4` gives an estimate of :math:`\pi`

We estimate the area by sampling bivariate uniforms and looking at the
fraction that fall into the unit circle

.. code-block:: julia

    n = 1000000
    
    count = 0
    for i in 1:n
        u, v = rand(2)
        d = sqrt((u - 0.5)^2 + (v - 0.5)^2)  # Distance from middle of square
        if d < 0.5
            count += 1
        end
    end
    
    area_estimate = count / n
    
    print(area_estimate * 4)  # dividing by radius**2




Exercise 4
----------

.. code-block:: julia

    payoff = 0
    count = 0
    
    print("Count = ")
    
    for i in 1:10
        U = rand()
        if U < 0.5
            count += 1
        else
            count = 0
        end
        print(count)
        if count == 3
            payoff = 1
        end
    end
    print("\n")
    println("payoff = $payoff")




We can simplify this somewhat using the **ternary operator**. Here's
some examples

.. code-block:: julia

    a = 1 < 2 ? "foo" : "bar"
    a






.. code-block:: julia

    a = 1 > 2 ? "foo" : "bar"
    a






Using this construction:

.. code-block:: julia

    payoff = 0
    count = 0
    
    print("Count = ")
    
    for i in 1:10
        U = rand()
        count = U < 0.5 ? count + 1 : 0  
        print(count)
        if count == 3
            payoff = 1
        end
    end
    print("\n")
    println("payoff = $payoff")




Exercise 5
----------

Here's one solution

.. code-block:: julia

    α = 0.9
    T = 200
    x = zeros(T + 1)
    
    for t in 1:T
        x[t+1] = α * x[t] + randn()
    end
    plot(x, color="blue")





Exercise 6
----------

.. code-block:: julia

    αs = [0.0, 0.8, 0.98]
    T = 200
    
    series = []
    labels = []
    
    for α in αs
        x = zeros(T + 1)
        x[1] = 0
        for t in 1:T
            x[t+1] = α * x[t] + randn()
        end
        push!(series, x)
        push!(labels, "α = $α")
    end
    
    plot(series, label=reshape(labels, 1, length(labels)))





