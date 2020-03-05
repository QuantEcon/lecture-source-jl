.. _troubleshooting:

.. include:: /_static/includes/header.raw

.. highlight:: python3

***************
Troubleshooting
***************

.. contents:: :depth: 2

This troubleshooting page is to help ensure you software environment is setup correctly
to run this lecture set locally on your machine.

Fixing Your Local Environment
=========================================

To set up a standard desktop environment, you can run the instructions in our `local setup lecture <https://julia.quantecon.org/getting_started_julia/getting_started.html#Desktop-Installation-of-Julia-and-Jupyter>`__. 

If you already have, make sure to try deleting your ``.julia`` directory (the "user depot," where packages are stored) and re-running the lectures (after running ``] add InstantiateFromURL``!). 

You can find this directory by running ``DEPOT_PATH[1]`` in a Julia REPL. 

Upgrading Julia 
=================

See the :ref:`lecture section <upgrading_julia>` on getting Atom and Jupyter working with a new version.

Fixing Atom
=============

See the :ref:`lecture section <atom_troubleshooting>` on troubleshooting Atom. 

Resetting a JupyterHub Lecture Set 
===================================

The lectures are delivered to JupyterHubs (like the `QuantEcon Syzygy server <https://quantecon.syzygy.ca>`__) using ``nbgitpuller``. 

To reset a single notebook, simply delete it and click the relevant link again. 

To reset your whole lecture set, run ``rm -rf quantecon-notebooks-julia`` in the Terminal (after ``cd``-ing to where they're downloaded, which is usually the root) and click any lecture's link again. 

Reporting an Issue
===================

One way to give feedback is to raise an issue through our `issue tracker 
<https://github.com/QuantEcon/lecture-source-py/issues>`__.

Please be as specific as possible.  Tell us where the problem is and as much
detail about your local set up as you can provide.

Another feedback option is to use our `discourse forum <https://discourse.quantecon.org/>`__.

Finally, you can provide direct feedback to contact@quantecon.org

