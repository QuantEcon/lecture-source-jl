.. _version_control:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Introduction to Git and Version Control
******************************************

An essential part of modern software engineering is using version control

    The ridiculous things people name their documents to do versioning, like "proposal v2 good revised NEW 11-15-06.doc", continue to crack me up. ---- Drew Houston, Founder of Dropbox

In this lecture, we'll discuss how it works on the GitHub platform 

.. contents:: :depth: 2

Overview
============

Topics:

* Installing Git and a GitHub Client
* Structure of a Git Repo
* How to Collaborate 
* How to Fight Fires 


Installing Git and a GitHub Client
===================================

First, make sure you install and setup `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_

Note that git is pre-installed if you're using the `docker image <https://hub.docker.com/r/quantecon/base/>`_

You might also consider installing a client like `GitHub desktop <https://desktop.github.com>`_ 

Lastly, make sure that you register for an account on `GitHub.com <https://www.github.com>`_ 

Structure of a Git Repo
=================================

The central object in GitHub is a *repository* (or repo). For an example of one, see the `QuantEcon Expectations Package <https://github.com/QuantEcon/Expectations.jl>`_ 

Every repository comes with a few standard files: 

* A `README.md` file, which is what GitHub displays for page visitors. This contains information about the project, how to contribute, etc. 

* A `LICENSE.md` file, which describes the terms under which the code is made available. 

* A `.gitignore` file, which tells GitHub not to move certain files (like `.aux` files from LaTeX) to and from the server

