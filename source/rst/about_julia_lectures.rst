About these Lectures
=====================

Overview
--------

Programming, mathematics and statistics are powerful tools for analyzing the functioning of economies.

This lecture series provides a hands-on instruction manual.

Topics include

-  algorithms and numerical methods for studying economic problems,

-  related mathematical and statistical concepts, and

-  basics of coding skills and software engineering.

The intended audience is undergraduate students, graduate students and
researchers in economics, finance and related fields.


Julia
--------

The coding language for this lecture series is Julia.

Note that there's also a related set of `Python lectures <add link here>`.

In terms of the differences,

* Python is a general purpose language featuring a huge user community in the sciences and an outstanding scientific ecosystem.

* Julia is more language more focused for technical and scientific computing, with an outstanding ecosystem for cutting-edge methods and algorithms.

Both are modern, open source, high productivity languages with all the key features needed for 
high performance computing.


While Julia has many features of a general purpose language, its specialization makes it much closer to
using Matlab than using a general purpose language - giving it an advantage in being closer to mathematical notation

A Word of Warning
-----------------

The disadvantage of specialization is that Julia tends to be used by domain experts, and consequently
the ecosystem and language for non-mathematical/non-scientfic computing tasks is inferior to Python

Another disadvantage is that, since it tends to be used by experts and is on the cutting edge, the tooling is
much more fragile and rudimentary than Python or Matlab

Luckily, this no longer applies to the language itself, which is now completely stable, but for casual users the environment
can be a detterent

For that reason, Julia is most appropriate at this time for researchers who (1) are investing in the future; (2) want
to use one of the many amazing packages that Julia makes possible (and are sometimes impossible in other languages); or (3) write
sufficiently specialized algorithms that the quirks of the environment are much less important than the end-result


Composition of Packages
-----------------------

Julia has the advantage that third party libraries are often written entirely in Julia itself.  Even 


Open Source
-----------

All the computing environments we work with are free and open source.

This means that you, your coauthors and your students can install them and their libraries on all of your computers without cost or concern about licenses.

Another advantage of open source libraries is that you can read them and learn
how they work.

For example, let’s say you want to know exactly how `pandas <http://pandas.pydata.org/>`__ computes Newey-West covariance matrices.

No problem: You can go ahead and `read the code <https://github.com/pydata/pandas/blob/master/pandas/stats/math.py>`__.

While dipping into external code libraries takes a bit of coding maturity, it’s very useful for

#. helping you understand the details of a particular implementation, and

#. building your programming skills by showing you code written by first rate programmers.

Also, you can modify the library to suit your needs: if the functionality provided is not exactly what you want, you are free to change it.

Another, more philosophical advantage of open source software is that it conforms to the `scientific ideal of reproducibility <https://en.wikipedia.org/wiki/Scientific_method>`__.



How about Other Languages?
--------------------------

But why don't you use language XYZ?



MATLAB
~~~~~~

While MATLAB has many nice features, it's starting to show its age.

It can no longer match Python or Julia in terms of performance and design.

MATLAB is also proprietary, which comes with its own set of disadvantages.

In particular, the Achilles Heel of Matlab is its lack of a package management
system, which means that either (1) you need to rely on Matlab's own packages; (2) you
need to write the code yourself; or (3) you rely on unreliable and manual ways to share code

With the expansion in complexity of numerical methods, and the need for researchers to
collaborate on code, this makes Matlab unsustainable


Given what’s available now, it’s hard to find any good reasons to invest in MATLAB.

Incidentally, if you decide to jump from MATLAB to Julia, `this cheat-sheet <http://cheatsheets.quantecon.org/>`__ will be useful.




R
~

`R <https://cran.r-project.org/>`__ is a very useful open source statistical environment and programming language

Its primary strength is its `vast collection <https://cran.r-project.org/web/packages>`__ of extension packages

Julia is more general purpose than R and hence a better fit for this course



C / C++ / Fortran? 
~~~~~~~~~~~~~~~~~~

Isn’t Fortran / C / C++ faster than Julia? In which case it must be better, right?

No, you can achieve speeds equal to or faster than those of compiled languages in Julia through just-in-time compilation --- we'll talk about how later on.

Second, remember that the correct objective function to minimize is

::

    total time = development time + execution time

In assessing this trade off, it’s necessary to bear in mind that

-  Your time is a far more valuable resource than the computer’s time.

-  Languages like Julia are much faster to write and debug in.

-  In any one program, the vast majority of CPU time will be spent iterating over just a few lines of your code.

The other issue with all three languages, as with Matlab, is the lack of a package management system

Collaborating on C++ or Fortran packages and distributing code between researchers is difficult


Last Word
^^^^^^^^^

Writing your entire program in Fortran / C / C++ is best thought of as “premature optimization”

On this topic we quote the godfather:

    We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil. – `Donald Knuth <https://en.wikipedia.org/wiki/Donald_Knuth>`__

But, to put the final part of the quote

    ... Yet we should not pass up our opportunities in that critical 3%. – `Donald Knuth <https://en.wikipedia.org/wiki/Donald_Knuth>`__

Julia is an excellent language to attain those last few percent, without having to occaisonally resort to C code or
fight against the design of the particular language

Credits
-------

These lectures have benefited greatly from comments and suggestions from our
colleagues, students and friends. Special thanks are due to our sponsoring
organization the Alfred P. Sloan Foundation and our research assistants Chase
Coleman, Spencer Lyon and Matthew McKay for innumerable contributions to the
code library and functioning of the website.

We also thank `Andrij Stachurski <http://drdrij.com/>`__ for his great web
skills, and the many others who have contributed suggestions, bug fixes or
improvements. They include but are not limited to Anmol Bhandari, Long Bui,
Jeong-Hun Choi, David Evans, Shunsuke Hori, Chenghan Hou, Doc-Jin Jang,
Qingyin Ma, Akira Matsushita, Tomohito Okabe, Daisuke Oyama, David Pugh, Alex
Olssen, Nathan Palmer, Bill Tubbs, Natasha Watkins, Pablo Winant and Yixiao
Zhou.
TODO:  ADD MORE!!!!!!!!!!!!

