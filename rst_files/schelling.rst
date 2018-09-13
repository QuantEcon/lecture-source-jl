.. _schelling:

.. include:: /_static/includes/lecture_howto_jl.raw

.. highlight:: julia

********************************
Schelling's Segregation Model
********************************

.. index::
    single: Schelling Segregation Model

.. index::
    single: Models; Schelling's Segregation Model

.. contents:: :depth: 2

Outline
=======================


In 1969, Thomas C. Schelling developed a simple but striking model of racial segregation :cite:`Schelling1969`

His model studies the dynamics of racially mixed neighborhoods

Like much of Schelling's work, the model shows how local interactions can lead to surprising aggregate structure

In particular, it shows that relatively mild preference for neighbors of similar race can lead in aggregate to the collapse of mixed neighborhoods, and high levels of segregation

In recognition of this and other research, Schelling was awarded the 2005 Nobel Prize in Economic Sciences (joint with Robert Aumann)

In this lecture we (in fact you) will build and run a version of Schelling's model



The Model
=======================

We will cover a variation of Schelling's model that is easy to program and captures the main idea

Set Up
---------

Suppose we have two types of people: orange people and green people

For the purpose of this lecture, we will assume there are 250 of each type

These agents all live on a single unit square

The location of an agent is just a point :math:`(x, y)`,  where :math:`0 < x, y < 1`


Preferences
-------------

We will say that an agent is *happy* if half or more of her 10 nearest neighbors are of the same type

Here 'nearest' is in terms of `Euclidean distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_

An agent who is not happy is called *unhappy*

An important point here is that agents are not averse to living in mixed areas

They are perfectly happy if half their neighbors are of the other color


Behavior
-----------

Initially, agents are mixed together (integrated)

In particular, the initial location of each agent is an independent draw from a bivariate uniform distribution on :math:`S = (0, 1)^2`

Now, cycling through the set of all agents, each agent is now given the chance to stay or move

We assume that each agent will stay put if they are happy and move if unhappy

The algorithm for moving is as follows

#. Draw a random location in :math:`S`

#. If happy at new location, move there

#. Else, go to step 1

In this way, we cycle continuously through the agents, moving as required

We continue to cycle until no one wishes to move



Results
=============

Let's have a look at the results we got when we coded and ran this model

As discussed above, agents are initially mixed randomly together


.. figure:: /_static/figures/schelling_fig1.png

But after several cycles they become segregated into distinct regions

.. figure:: /_static/figures/schelling_fig2.png

.. figure:: /_static/figures/schelling_fig3.png

.. figure:: /_static/figures/schelling_fig4.png

In this instance, the program terminated after 4 cycles through the set of
agents, indicating that all agents had reached a state of happiness

What is striking about the pictures is how rapidly racial integration breaks down

This is despite the fact that people in the model don't actually mind living mixed with the other type

Even with these preferences, the outcome is a high degree of segregation


Exercises
===============


.. _schelling_ex1:

Exercise 1
------------

Implement and run this simulation for yourself

Use 250 agents of each type

Solutions
==========

Exercise 1
----------

Here's one solution that does the job we want. If you feel like a
further exercise you can probably speed up some of the computations and
then increase the number of agents.

.. code-block:: julia
  :class: test

  using Test

.. code-block:: julia

    using Plots, Random, LinearAlgebra
    Random.seed!(42)  # set seed for random numbers. Reproducible output


.. code-block:: julia

    mutable struct Agent{TI<:Integer, TF<:AbstractFloat}
        kind::TI
        location::Vector{TF}
    end

    # constructor
    Agent(k) = Agent(k, rand(2))


    function draw_location!(a)
        a.location = rand(2)
        nothing
    end

    # distance is just 2 norm: uses our subtraction function
    get_distance(a, o) = norm(a.location - o.location)

    function is_happy(a, others)
        "True if sufficient number of nearest neighbors are of the same type."
        # distances is a list of pairs (d, agent), where d is distance from
        # agent to self
        distances = Vector{Tuple{Float64, Agent}}()

        for agent in others
            if a != agent
                dist = get_distance(a, agent)
                push!(distances, (dist, agent))
            end
        end

        # == Sort from smallest to largest, according to distance == #
        sort!(distances)

        # == Extract the neighboring agents == #
        neighbors = [agent for (d, agent) in distances[1:num_neighbors]]

        # == Count how many neighbors have the same type as self == #
        num_same_type = sum(a.kind == other.kind for other in neighbors)

        return num_same_type ≥ require_same_type
    end

    function update!(a, others)
        "If not happy, then randomly choose new locations until happy."
        while !is_happy(a, others)
            draw_location!(a)
        end
        return nothing
    end

    function plot_distribution(agents, cycle_num)
        x_vals_0, y_vals_0 = Float64[], Float64[]
        x_vals_1, y_vals_1 = Float64[], Float64[]

        # == Obtain locations of each type == #
        for agent in agents
            x, y = agent.location
            if agent.kind == 0
                push!(x_vals_0, x)
                push!(y_vals_0, y)
            else
                push!(x_vals_1, x)
                push!(y_vals_1, y)
            end
        end

        p = scatter(x_vals_0, y_vals_0, color=:orange, markersize=8, alpha=0.6)
        scatter!(x_vals_1, y_vals_1, color=:green, markersize=8, alpha=0.6)
        plot!(title="Cycle $(cycle_num)", legend=:none)

        return p
    end;

.. code-block:: julia

    # == Main == #

    num_of_type_0 = 250
    num_of_type_1 = 250
    num_neighbors = 10      # Number of agents regarded as neighbors
    require_same_type = 5   # Want at least this many neighbors to be same type

    # == Create a list of agents == #
    agents = Agent[Agent(0) for i in 1:num_of_type_0]
    push!(agents, [Agent(1) for i in 1:num_of_type_1]...)

    count = 1

    # ==  Loop until none wishes to move == #
    while true
        println("Entering loop $count")
        p = plot_distribution(agents, count)
        display(p)
        count += 1
        no_one_moved = true
        movers = 0
        for agent in agents
            old_location = agent.location
            update!(agent, agents)
            if !isapprox(0.0, maximum(old_location - agent.location))
                no_one_moved = false
            end
        end
        if no_one_moved
            break
        end
    end

    println("Converged, terminating")


.. code-block:: julia
  :class: test

  @test sum(agent.kind for agent in agents) == 250
