.. _short_path:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************************
Shortest Paths
******************************************************

.. index::
	single: Dynamic Programming; Shortest Paths

.. contents:: :depth: 2

Overview
=============

The shortest path problem is a `classic problem <https://en.wikipedia.org/wiki/Shortest_path>`_ in mathematics and computer science with applications in

* Economics (sequential decision making, analysis of social networks, etc.)

* Operations research and transportation

* Robotics and artificial intelligence

* Telecommunication network design and routing

* etc., etc.


Variations of the methods we discuss in this lecture are used millions of times every day, in applications such as

* Google Maps

* routing packets on the internet


For us, the shortest path problem also provides a nice introduction to the logic of **dynamic programming**

Dynamic programming is an extremely powerful optimization technique that we apply in many lectures on this site





Outline of the Problem
=========================

The shortest path problem is one of finding how to traverse a `graph <https://en.wikipedia.org/wiki/Graph_%28mathematics%29>`_ from one specified node to another at minimum cost

Consider the following graph

.. figure:: /_static/figures/graph.png
   :scale: 100%

We wish to travel from node (vertex) A to node G at minimum cost

* Arrows (edges) indicate the movements we can take
* Numbers on edges indicate the cost of traveling that edge

Possible interpretations of the graph include

* Minimum cost for supplier to reach a destination
* Routing of packets on the internet (minimize time)
* Etc., etc.

For this simple graph, a quick scan of the edges shows that the optimal paths are

* A, C, F, G at cost 8

.. figure:: /_static/figures/graph4.png
   :scale: 100%

* A, D, F, G at cost 8

.. figure:: /_static/figures/graph3.png
   :scale: 100%

Finding Least-Cost Paths
===========================

For large graphs we need a systematic solution

Let :math:`J(v)` denote the minimum cost-to-go from node :math:`v`, understood as the total cost from :math:`v` if we take the best route

Suppose that we know :math:`J(v)` for each node :math:`v`, as shown below for the graph from the preceding example


.. figure:: /_static/figures/graph2.png
   :scale: 100%

Note that :math:`J(G) = 0`

The best path can now be found as follows

* Start at A

* From node v, move to any node that solves

.. math::
    :label: spprebell

    \min_{w \in F_v} \{ c(v, w) + J(w) \}


where

* :math:`F_v` is the set of nodes that can be reached from :math:`v` in one step

* :math:`c(v, w)` is the cost of traveling from :math:`v` to :math:`w`

Hence, if we know the function :math:`J`, then finding the best path is almost trivial

But how to find :math:`J`?

Some thought will convince you that, for every node :math:`v`,
the function :math:`J` satisfies

.. math::
    :label: spbell

    J(v) = \min_{w \in F_v} \{ c(v, w) + J(w) \}


This is known as the *Bellman equation*, after the mathematician Richard Bellman



Solving for :math:`J`
=========================

The standard algorithm for finding :math:`J` is to start with

.. math::
    :label: spguess

    J_0(v) = M \text{ if } v \not= \text{ destination, else } J_0(v) = 0


where :math:`M` is some large number

Now we use the following algorithm

#. Set :math:`n = 0`
#. Set :math:`J_{n+1} (v) = \min_{w \in F_v} \{ c(v, w) + J_n(w) \}` for all :math:`v`
#. If :math:`J_{n+1}` and :math:`J_n` are not equal then increment :math:`n`, go to 2

In general, this sequence converges to :math:`J`---the proof is omitted






Exercises
============

.. _short_path_ex1:

Exercise 1
------------

Use the algorithm given above to find the optimal path (and its cost) for the
following graph

You can put it in a Jupyter notebook cell and hit Shift-Enter --- it
will be saved in the local directory as file `graph.txt`

.. code-block:: python3

    %%file graph.txt
    node0, node1 0.04, node8 11.11, node14 72.21
    node1, node46 1247.25, node6 20.59, node13 64.94
    node2, node66 54.18, node31 166.80, node45 1561.45
    node3, node20 133.65, node6 2.06, node11 42.43
    node4, node75 3706.67, node5 0.73, node7 1.02
    node5, node45 1382.97, node7 3.33, node11 34.54
    node6, node31 63.17, node9 0.72, node10 13.10
    node7, node50 478.14, node9 3.15, node10 5.85
    node8, node69 577.91, node11 7.45, node12 3.18
    node9, node70 2454.28, node13 4.42, node20 16.53
    node10, node89 5352.79, node12 1.87, node16 25.16
    node11, node94 4961.32, node18 37.55, node20 65.08
    node12, node84 3914.62, node24 34.32, node28 170.04
    node13, node60 2135.95, node38 236.33, node40 475.33
    node14, node67 1878.96, node16 2.70, node24 38.65
    node15, node91 3597.11, node17 1.01, node18 2.57
    node16, node36 392.92, node19 3.49, node38 278.71
    node17, node76 783.29, node22 24.78, node23 26.45
    node18, node91 3363.17, node23 16.23, node28 55.84
    node19, node26 20.09, node20 0.24, node28 70.54
    node20, node98 3523.33, node24 9.81, node33 145.80
    node21, node56 626.04, node28 36.65, node31 27.06
    node22, node72 1447.22, node39 136.32, node40 124.22
    node23, node52 336.73, node26 2.66, node33 22.37
    node24, node66 875.19, node26 1.80, node28 14.25
    node25, node70 1343.63, node32 36.58, node35 45.55
    node26, node47 135.78, node27 0.01, node42 122.00
    node27, node65 480.55, node35 48.10, node43 246.24
    node28, node82 2538.18, node34 21.79, node36 15.52
    node29, node64 635.52, node32 4.22, node33 12.61
    node30, node98 2616.03, node33 5.61, node35 13.95
    node31, node98 3350.98, node36 20.44, node44 125.88
    node32, node97 2613.92, node34 3.33, node35 1.46
    node33, node81 1854.73, node41 3.23, node47 111.54
    node34, node73 1075.38, node42 51.52, node48 129.45
    node35, node52 17.57, node41 2.09, node50 78.81
    node36, node71 1171.60, node54 101.08, node57 260.46
    node37, node75 269.97, node38 0.36, node46 80.49
    node38, node93 2767.85, node40 1.79, node42 8.78
    node39, node50 39.88, node40 0.95, node41 1.34
    node40, node75 548.68, node47 28.57, node54 53.46
    node41, node53 18.23, node46 0.28, node54 162.24
    node42, node59 141.86, node47 10.08, node72 437.49
    node43, node98 2984.83, node54 95.06, node60 116.23
    node44, node91 807.39, node46 1.56, node47 2.14
    node45, node58 79.93, node47 3.68, node49 15.51
    node46, node52 22.68, node57 27.50, node67 65.48
    node47, node50 2.82, node56 49.31, node61 172.64
    node48, node99 2564.12, node59 34.52, node60 66.44
    node49, node78 53.79, node50 0.51, node56 10.89
    node50, node85 251.76, node53 1.38, node55 20.10
    node51, node98 2110.67, node59 23.67, node60 73.79
    node52, node94 1471.80, node64 102.41, node66 123.03
    node53, node72 22.85, node56 4.33, node67 88.35
    node54, node88 967.59, node59 24.30, node73 238.61
    node55, node84 86.09, node57 2.13, node64 60.80
    node56, node76 197.03, node57 0.02, node61 11.06
    node57, node86 701.09, node58 0.46, node60 7.01
    node58, node83 556.70, node64 29.85, node65 34.32
    node59, node90 820.66, node60 0.72, node71 0.67
    node60, node76 48.03, node65 4.76, node67 1.63
    node61, node98 1057.59, node63 0.95, node64 4.88
    node62, node91 132.23, node64 2.94, node76 38.43
    node63, node66 4.43, node72 70.08, node75 56.34
    node64, node80 47.73, node65 0.30, node76 11.98
    node65, node94 594.93, node66 0.64, node73 33.23
    node66, node98 395.63, node68 2.66, node73 37.53
    node67, node82 153.53, node68 0.09, node70 0.98
    node68, node94 232.10, node70 3.35, node71 1.66
    node69, node99 247.80, node70 0.06, node73 8.99
    node70, node76 27.18, node72 1.50, node73 8.37
    node71, node89 104.50, node74 8.86, node91 284.64
    node72, node76 15.32, node84 102.77, node92 133.06
    node73, node83 52.22, node76 1.40, node90 243.00
    node74, node81 1.07, node76 0.52, node78 8.08
    node75, node92 68.53, node76 0.81, node77 1.19
    node76, node85 13.18, node77 0.45, node78 2.36
    node77, node80 8.94, node78 0.98, node86 64.32
    node78, node98 355.90, node81 2.59
    node79, node81 0.09, node85 1.45, node91 22.35
    node80, node92 121.87, node88 28.78, node98 264.34
    node81, node94 99.78, node89 39.52, node92 99.89
    node82, node91 47.44, node88 28.05, node93 11.99
    node83, node94 114.95, node86 8.75, node88 5.78
    node84, node89 19.14, node94 30.41, node98 121.05
    node85, node97 94.51, node87 2.66, node89 4.90
    node86, node97 85.09
    node87, node88 0.21, node91 11.14, node92 21.23
    node88, node93 1.31, node91 6.83, node98 6.12
    node89, node97 36.97, node99 82.12
    node90, node96 23.53, node94 10.47, node99 50.99
    node91, node97 22.17
    node92, node96 10.83, node97 11.24, node99 34.68
    node93, node94 0.19, node97 6.71, node99 32.77
    node94, node98 5.91, node96 2.03
    node95, node98 6.17, node99 0.27
    node96, node98 3.32, node97 0.43, node99 5.87
    node97, node98 0.30
    node98, node99 0.33
    node99,

Here the line ``node0, node1 0.04, node8 11.11, node14 72.21`` means that from `node0` we can go to

* `node1` at cost 0.04
* `node8` at cost 11.11
* `node14` at cost 72.21

and so on

According to our calculations, the optimal path and its cost are like `this <https://github.com/QuantEcon/QuantEcon.lectures.code/blob/master/short_path/graph_out.txt>`__

Your code should replicate this result



Solutions
==========




Exercise 1
----------

If you Shift-Enter in the next cell you'll save the data we want to work
with in the local directory --- then scroll down for the solution.

.. code-block:: julia

    f = open("graph.txt", "w")
    contents = """node0, node1 0.04, node8 11.11, node14 72.21
    node1, node46 1247.25, node6 20.59, node13 64.94
    node2, node66 54.18, node31 166.80, node45 1561.45
    node3, node20 133.65, node6 2.06, node11 42.43
    node4, node75 3706.67, node5 0.73, node7 1.02
    node5, node45 1382.97, node7 3.33, node11 34.54
    node6, node31 63.17, node9 0.72, node10 13.10
    node7, node50 478.14, node9 3.15, node10 5.85
    node8, node69 577.91, node11 7.45, node12 3.18
    node9, node70 2454.28, node13 4.42, node20 16.53
    node10, node89 5352.79, node12 1.87, node16 25.16
    node11, node94 4961.32, node18 37.55, node20 65.08
    node12, node84 3914.62, node24 34.32, node28 170.04
    node13, node60 2135.95, node38 236.33, node40 475.33
    node14, node67 1878.96, node16 2.70, node24 38.65
    node15, node91 3597.11, node17 1.01, node18 2.57
    node16, node36 392.92, node19 3.49, node38 278.71
    node17, node76 783.29, node22 24.78, node23 26.45
    node18, node91 3363.17, node23 16.23, node28 55.84
    node19, node26 20.09, node20 0.24, node28 70.54
    node20, node98 3523.33, node24 9.81, node33 145.80
    node21, node56 626.04, node28 36.65, node31 27.06
    node22, node72 1447.22, node39 136.32, node40 124.22
    node23, node52 336.73, node26 2.66, node33 22.37
    node24, node66 875.19, node26 1.80, node28 14.25
    node25, node70 1343.63, node32 36.58, node35 45.55
    node26, node47 135.78, node27 0.01, node42 122.00
    node27, node65 480.55, node35 48.10, node43 246.24
    node28, node82 2538.18, node34 21.79, node36 15.52
    node29, node64 635.52, node32 4.22, node33 12.61
    node30, node98 2616.03, node33 5.61, node35 13.95
    node31, node98 3350.98, node36 20.44, node44 125.88
    node32, node97 2613.92, node34 3.33, node35 1.46
    node33, node81 1854.73, node41 3.23, node47 111.54
    node34, node73 1075.38, node42 51.52, node48 129.45
    node35, node52 17.57, node41 2.09, node50 78.81
    node36, node71 1171.60, node54 101.08, node57 260.46
    node37, node75 269.97, node38 0.36, node46 80.49
    node38, node93 2767.85, node40 1.79, node42 8.78
    node39, node50 39.88, node40 0.95, node41 1.34
    node40, node75 548.68, node47 28.57, node54 53.46
    node41, node53 18.23, node46 0.28, node54 162.24
    node42, node59 141.86, node47 10.08, node72 437.49
    node43, node98 2984.83, node54 95.06, node60 116.23
    node44, node91 807.39, node46 1.56, node47 2.14
    node45, node58 79.93, node47 3.68, node49 15.51
    node46, node52 22.68, node57 27.50, node67 65.48
    node47, node50 2.82, node56 49.31, node61 172.64
    node48, node99 2564.12, node59 34.52, node60 66.44
    node49, node78 53.79, node50 0.51, node56 10.89
    node50, node85 251.76, node53 1.38, node55 20.10
    node51, node98 2110.67, node59 23.67, node60 73.79
    node52, node94 1471.80, node64 102.41, node66 123.03
    node53, node72 22.85, node56 4.33, node67 88.35
    node54, node88 967.59, node59 24.30, node73 238.61
    node55, node84 86.09, node57 2.13, node64 60.80
    node56, node76 197.03, node57 0.02, node61 11.06
    node57, node86 701.09, node58 0.46, node60 7.01
    node58, node83 556.70, node64 29.85, node65 34.32
    node59, node90 820.66, node60 0.72, node71 0.67
    node60, node76 48.03, node65 4.76, node67 1.63
    node61, node98 1057.59, node63 0.95, node64 4.88
    node62, node91 132.23, node64 2.94, node76 38.43
    node63, node66 4.43, node72 70.08, node75 56.34
    node64, node80 47.73, node65 0.30, node76 11.98
    node65, node94 594.93, node66 0.64, node73 33.23
    node66, node98 395.63, node68 2.66, node73 37.53
    node67, node82 153.53, node68 0.09, node70 0.98
    node68, node94 232.10, node70 3.35, node71 1.66
    node69, node99 247.80, node70 0.06, node73 8.99
    node70, node76 27.18, node72 1.50, node73 8.37
    node71, node89 104.50, node74 8.86, node91 284.64
    node72, node76 15.32, node84 102.77, node92 133.06
    node73, node83 52.22, node76 1.40, node90 243.00
    node74, node81 1.07, node76 0.52, node78 8.08
    node75, node92 68.53, node76 0.81, node77 1.19
    node76, node85 13.18, node77 0.45, node78 2.36
    node77, node80 8.94, node78 0.98, node86 64.32
    node78, node98 355.90, node81 2.59
    node79, node81 0.09, node85 1.45, node91 22.35
    node80, node92 121.87, node88 28.78, node98 264.34
    node81, node94 99.78, node89 39.52, node92 99.89
    node82, node91 47.44, node88 28.05, node93 11.99
    node83, node94 114.95, node86 8.75, node88 5.78
    node84, node89 19.14, node94 30.41, node98 121.05
    node85, node97 94.51, node87 2.66, node89 4.90
    node86, node97 85.09
    node87, node88 0.21, node91 11.14, node92 21.23
    node88, node93 1.31, node91 6.83, node98 6.12
    node89, node97 36.97, node99 82.12
    node90, node96 23.53, node94 10.47, node99 50.99
    node91, node97 22.17
    node92, node96 10.83, node97 11.24, node99 34.68
    node93, node94 0.19, node97 6.71, node99 32.77
    node94, node98 5.91, node96 2.03
    node95, node98 6.17, node99 0.27
    node96, node98 3.32, node97 0.43, node99 5.87
    node97, node98 0.30
    node98, node99 0.33
    node99, 
    """
    write(f, contents)
    close(f)

.. code-block:: julia

    using Printf
    function read_graph(in_file)
        graph = Dict()
        infile = open(in_file, "r")
        for line in readlines(infile)
            elements = reverse!(split(line, ','))
            node = strip(pop!(elements))
            graph[node] = []
            if node != "node99"
                for element in elements
                    dest, cost = split(element)
                    push!(graph[node], [strip(dest), parse(Float64, cost)])
                end
            end
            
        end
        close(infile)
        return graph
    end
    
    
    function update_J(J, graph)
        next_J = Dict()
        for node in keys(graph)
            if node == "node99"
                next_J[node] = 0
            else
                next_J[node] = minimum([cost + J[dest] for (dest, cost) in graph[node]])
            end
        end
        return next_J
    end
    
    
    function print_best_path(J, graph)
        sum_costs = 0
        current_location = "node0"
        while current_location != "node99"
            println(current_location)
            running_min = 1e10
            minimizer_dest = "none"
            minimizer_cost = 1e10
            for (dest, cost) in graph[current_location]
                cost_of_path = cost + J[dest]
                if cost_of_path < running_min
                    running_min = cost_of_path
                    minimizer_cost = cost
                    minimizer_dest = dest
                end
            end
            
            current_location = minimizer_dest
            sum_costs += minimizer_cost
        end
        
        println("node99")
        @printf "\nCost: %.2f" sum_costs
    end
    
    graph = read_graph("graph.txt")
    M = 1e10
    J = Dict()
    for node in keys(graph)
        J[node] = M
    end
    J["node99"] = 0
    
    while true
        next_J = update_J(J, graph)
        if next_J == J
            break
        else
            J = next_J
        end
    end
    
    print_best_path(J, graph)


