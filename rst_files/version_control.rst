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
* Collaborating with GitHub  
* Firefighting and Debugging 


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

* A `LICENSE.md` file, which describes the terms under which the code is made available

* A `.gitignore` file, which tells GitHub not to move certain files (like `.aux` files from LaTeX) to and from the server

The distinguishing feature of a git repo is that it has history:

Git repos are represented as a sequence of *commits*, each corresponding to a *diff* from the previous state 

For example, see `this commit <https://github.com/QuantEcon/Expectations.jl/commit/22d1e3c11f012dbcd4878d03f66cb3b39529d781>`_, which makes some changes to the documentation in the `Expectations.jl`
 
The process of uploading commits to the server (and thus adding to history) is called *pushing*. Likewise, updating your local machine to match what's on the server (which acts as the "ground truth" for git) is called *pulling*

Each commit has a unique identifier, like `22d1e3c11f012dbcd4878d03f66cb3b39529d781`, which is an SHA-1 hash. It also has some METADATA, such as the time it was created, the identity of the committer, and a message left by the committer 

Git repositories also support parallel development, with objects called *branches*: 

* The main branch is called *master*, and cannot be deleted

* People can "check out" branches from a given state of *master* (or some other branch), in much the same way that two rivers can diverge downstream after sharing the same history. They can eventually be merged using an object called a *pull request*, covered below  

Lastly, it is possible to "tag" see of a git repo (see "releases" of the `QuantEcon/Expectations.jl`), which is akin to publishing archival copies for future reference or cleanup 

Collaborating with GitHub 
==============================

Instantianting a Repository Locally 
--------------------------------------

First, you'll need to "clone" a copy of the repository locally. You can do this in GitHub Desktop, or via the command line 

.. code-block:: none 

    git clone https://github.com/QuantEcon/Expectations.jl 

This will create a folder with the repository name (`Expectations.jl`), and all its contents, along with a `.git` repository that contains the commit sequence 

If you don't have write access to the repository, you can create a "fork" on your account, which tells GitHub to copy the current state and all history 

Check Out a Branch 
-------------------

Before starting work, it's best to check out a new branch, to avoid contaminating the `master` of your repo 

You can do this either using the intuitive interface in GitHub Desktop, or by running 

.. code-block:: none 

    git checkout -b myBranchName 

Your branch name should be something descriptive, like `add-tests` or `plot-fixes`

Make Your Changes 
-----------------------

You're now ready to edit your files as you would normally (see our "tools and editors" lecture). When working on adding a feature to code, a submission isn't complete unless it has all of:

* The actual feature you're implementing 

* Documentation

* Tests 

Commit Your Changes 
----------------------

As you work, you should add to the sequence of commits (as opposed to making one monolithic commit with all your work)

As before, you can do this either in GitHub Desktop, or by running

.. code-block:: none 

    git add * # "Stages" files to GitHub 
    git commit -m "My commit message." # Commits with a message 

The goal is that each commit should represent an atomic change in code which can be parseable by a human reviewer 

Push Your Commits
--------------------

Once you have a sequence of commits you're happy with, it's time to upload them to the server 

In GitHub Desktop, you can do this by clicking "publish branch." In the command line, it's 

.. code-block:: none 

    git push -u origin myBranchName 


Merging your Work 
-----------------------

Now, it's time to incorporate your changes into `master` (either in your fork, or in a central repo) 

To do this, we open what's called a "pull request." It's best to do this via that GitHub website, e.g. `here <https://github.com/quantecon/expectations.jl/pulls>`_ 

The pull request will have one of two outcomes: 

* Automatically mergeable, if the history you created and the history created in `master` since then don't conflict (i.e., you don't try to change the same text in two ways)

* The opposite. In this case, you will need to use the GitHub website's conflict resolution tool to manually indicate which set of diffs (those you applied, or those in `master`) should be applied to a file

Syncing with the Server 
--------------------------

An additional part, not part of the above workflow, is syncing your command with the server. This is useful so that your local directory doesn't get out of date 

The GitHub Desktop button for this is fairly simple, so we'll give the command line instructions below

.. code-block:: none 

    git fetch origin # Updates git's "history book" of commits, but doesn't apply the new ones to your branch(es)
    git pull # Applies the history gained 

Firefighting and Debugging 
==============================

Sometimes, if things were misconfigured somewhere, `git` will complain about operations you ask it to perform. This section attempts to categorize and solve such errors

    I want to reset my local branch against what's on the server 

To do this, run `git reset --hard origin/master`. This will reset each file `git` knows about to its state in the `master` on the server. But, it won't delete untracked files

    I want to delete untracked files 

Run `git clean -nf` to see what files would be deleted, and `git clean -f` to carry out the operation. 

    I ran `git pull` and it's asking me to "merge"?

This is normal, and happens when `git` wants to apply upstream changes to your local copy. Just enter a commit message (see our lecture on vim) and hit enter 

    Git can't merge because of a merge conflict 

This is because the conflict described above with PRs (two conflicting diffs attempted on the same piece of text) is happening locally. To resolve it, you can do the following using Visual Studio Code: 

[PUT IMAGES HERE]

For more information, and debugging tips for nearly any conceivable scenario, see this `excellent git repo <https://github.com/k88hudson/git-flight-rules>`_ 