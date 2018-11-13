.. _version_control:

.. include:: /_static/includes/lecture_howto_jl_full.raw

******************************************
Introduction to Git and Version Control
******************************************

Co-authored with Arnav Sood

An essential part of modern software engineering is using version control

We use version control because 

* Not all iterations on a file are perfect, and you may want to revert changes
* We want to be able to see who has changed what and how 
* We want a uniform version scheme to do this between people and machines
* Concurrent editing on code is necessary for collaboration
* Version control is an essential part of creating reproducible research


In this lecture, we'll discuss how to use Git and GitHub

.. contents:: :depth: 2

Setup 
==================

First, make sure you create an account on `GitHub.com <http://github.com/>`_

If you are a student, be sure to use the GitHub `Student Developer Pack <https://education.github.com/pack/>`_

Otherwise, see if you qualify for a free `Non-Profit/Academic Plan <https://help.github.com/articles/about-github-education-for-educators-and-researchers/>`_

These come with things like unlimited private repositories, testing support, etc.

Next, install ``git`` and the GitHub Desktop application 

1. Install `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/>`_

2. Install the `GitHub Desktop <https://desktop.github.com/>`_

Optionally (but strongly recommended):  On Windows, change the default line-ending by

* Opening a Windows/Powershell console, or the "Git Bash" installed in the previous step

* Run the following

.. code-block:: none

    git config --global core.eol lf
    git config --global core.autocrlf false

Git vs. GitHub vs. GitHub Desktop
-----------------------------------

To understand the relationship

* Git is an infrastructure for versioning and merging files, and is neither specific to GitHub and does not even require an online server to function
* GitHub provides a server to coordinate distributing the Git versions, and adds some additional features for managing projects
* GitHub Desktop is just one of many GUI-based clients to make Git and GitHub easier to use

Later, you may find yourself using alternatives

* GitHub is the market leader for open-source projects and Julia, but there are other options, e.g. `GitLab <https://about.gitlab.com/>`_ and `Bitbucket <https://bitbucket.org>`_
* Instead of the GitHub Desktop, you may directly use the Git `command-line <version_control_commandline>`_ , `GitKraken <https://www.gitkraken.com/>`_, or use the Git functionality built into editors such as `Atom <https://atom.io/>`_ or `VS Code <https://code.visualstudio.com/>`_

Since these lecture notes are intended to provide a minimal path to using the technologies, here we will conflate the workflow of these distinct projects


Basic Objects 
========================

Repositories
------------------------

The fundamental object in GitHub is a *repository* (or "repo.") This is the master directory for a project 

One example of a repo is the QuantEcon `Expectations.jl <https://github.com/quantecon/expectations.jl/>`_ package 

On the machine, a repo is a normal directory, along with a subdirectory called ``.git`` which contains the history of changes 

Commits
----------------------

GitHub stores history as a sequence of changes to text, called *commits* 

Here is an example of a commit, which revises the style guide in a QuantEcon repo (`link <https://github.com/QuantEcon/lecture-source-jl/commit/ba59c3ea9a0dec10def3f4f3928af5e2827f3b92/>`_)

In particular, commits have the following features

* An ID (formally, an "SHA-1 hash")
* Content (i.e., a before and after state)
* Metadata (author, timestamp, commit message, etc.)

It is crucial to remember that commits represent differences in text, as opposed to repository states 

The ideal commit is small enough to be scanned by a human being in this window 

Common Files 
-----------------------------------------

In addition, each GitHub repository typically comes with a few standard text files 

* A ``.gitignore`` file, which lists files/extensions/directories that GitHub shouldn't try to track (e.g., LaTeX compilation byproducts)
* A ``README.md`` file, which is a Markdown file which GitHub puts on the repository website 
* A ``LICENSE.txt`` file, which describes the terms under which the repository's contents are made available 

For an example of all three, see the `Expectations.jl <https://github.com/quantecon/expectations.jl/>`_ repo linked above 

Of these, the ``README.md`` is the most important, as GitHub will display it as `Markdown <https://guides.github.com/features/mastering-markdown/>`_ when accessing the repository online

.. _new_repo_workflow:

Individual Workflow
====================================

In this section, we'll describe how to use GitHub to version your own projects 

Much of this will carry over to the collaborative section 

Creating a Repository 
---------------------------------

In general, we will always want to make new repos using the following dropdown 

.. figure:: /_static/figures/git-makerepo.png
    :scale: 60%

We can then configure repository options as such 

.. figure:: /_static/figures/git-makerepo-full.png
    :scale: 60%

In this case, we're making a public repo ``github.com/quantecon_user/example_repository``, which will come with a ``README.md``, is licensed under the MIT License, and will ignore Julia compilation byproducts 

Cloning a Repository 
---------------------------------------

The next step is to get this to our local machine 

.. figure:: /_static/figures/git-clone.png
    :scale: 60%

This dropdown gives us a few options 

* "Open in Desktop" will call to the GitHub Desktop application that we've installed 
* "Download Zip" will download the directory *without the .git* subdirectory
* The copy/paste button next to the link lets us use the command line, i.e. ``git clone https://github.com/quanteconuser/example_repository.git``

Making and Managing Changes 
-------------------------------------------

Now that we have the repository, we can start working with it 

For example, let's say that we've amended the ``README.md`` (using our editor of choice), and also added a new file ``economics.jl`` which we're still working on 

Returning to GitHub Desktop, we should see something like 

.. figure:: /_static/figures/git-desktop-commit.png
    :scale: 60%

To select individual files for commit, we can use the check boxes to the left of each file 

Let's say you select only the README to commit. Going to the history tab should show you our change 

.. figure:: /_static/figures/git-desktop-commit2.png
    :scale: 60%

The Julia file is unchanged 

Pushing to the Server 
--------------------------------

As of now, this commit lives only on our local machine. To upload it to the server, you can simply click the "Push Origin" button atop the screen

The small "1^" to the right of the text indicates we have one commit to upload 

Reading and Reverting History 
-----------------------------------------

As mentioned, one of the key features of GitHub is the ability to scan through history 

By clicking the "commits" tab on the repo front page, for example, we see `this page <https://github.com/quanteconuser/example_repository/commits/master>`_

Clicking an individual commit gives us the granular view, (e.g., `example commit <https://github.com/quanteconuser/example_repository/commit/d0b17f5ce0f8742e88da9b604bfed418d6a16884/>`_)

Sometimes, however, we want to not only inspect what happened before, but go back to it 

* If you haven't made the commit yet, just right-click the file and hit "discard changes" to reset the file to the last known commit 
* If you have made the commit but haven't pushed to the server yet, go to the "history" tab as above, right click the commit and click "revert this commit." This will create the inverse commit, as above 

.. figure:: /_static/figures/git-revert-commit.png
    :scale: 60%

Working across Machines
--------------------------------------

Oftentimes, you will want to work on the same project across multiple machines (e.g., a home laptop and a lab workstation)

The key is to push changes from one machine, and then to pull changes from the other machine 

Pushing can be done as above. To pull, simply click pull under the "repository" dropdown at the top of the screen 

.. figure:: /_static/figures/git-pull.png
    :scale: 60%

Collaborative Work
==================================

Adding Collaborators
----------------------------

First, let's add a collaborator to the ``quanteconuser/example_repository`` lecture we created earlier 

We can do this by clicking "settings => collaborators," as follows

.. figure:: /_static/figures/git-collab.png
    :scale: 60%

Project Management 
--------------------------------

GitHub's website also comes with project management tools to coordinate work between people 

The main one is an *issue*, which we can create from the issues tab. You should see something like this

.. figure:: /_static/figures/git-issue.png
    :scale: 60%

Let's unpack the different components 

* The *assignees* dropdown lets you select people tasked to work on the issue 

* The *labels* dropdown lets you tag the issue with labels visible from the issues page, such as "high priority" or "feature request" 

* It's possible to tag other issues and collaborators (including in different repos) by linking to them in the comments. This is part of what's called *GitHub-Flavored Markdown* 

For an example of an issue, see `example issue <https://github.com/quanteconuser/example_repository/issues/1>`_ 

The checkbox idiom is a common one to manage projects in GitHub 

You can see open issues at a glance from the general issues tab 

.. figure:: /_static/figures/git-issue-tab.png
    :scale: 60%

Reviewing Code 
------------------------------

There are a few different ways to review people's code in GitHub 

* Whenever people push to a project you're working on, you'll receive an email notification
* You can also review individual line-items or commits by opening commits in the granular view as `above <https://github.com/quanteconuser/example_repository/commit/d0b17f5ce0f8742e88da9b604bfed418d6a16884/>`_)

.. figure:: /_static/figures/git-review.png
    :scale: 60%

.. merge_conflict:

Merge Conflicts
----------------------------

Any project management tool needs to figure out how to reconcile conflicting changes between people

In GitHub, this event is called a "merge conflict," and occurs whenever people make conflicting changes to the same *line* of code 

Note that this means that two people touching the same file is OK, so long as the differences are compatible 

A common use case is when we try to push changes to the server, but someone else has pushed conflicting changes. GitHub will give us the following window 

.. figure:: /_static/figures/git-merge-conflict.png
    :scale: 60%

* The warning symbol next to the file indicates the existence of a merge conflict 
* The viewer tries to show us the discrepancy (I changed the word repository to repo, but someone else tried to change it to "repo" with quotes)

To fix the conflict, we can go into a text editor (such as Atom or VS Code). Here's an image of what we see in Atom 

.. figure:: /_static/figures/atom-merge-conflict.png
    :scale: 60%

Let's say we click the first "use me" (to indicate that my changes should win out), and then save the file. Returning to GitHub Desktop gives us a pre-formed commit to accept 

.. figure:: /_static/figures/git-merge-commit.png
    :scale: 60%

Collaboration via Pull Request
=====================================

One of the defining features of GitHub is that it is the dominant platform for *open-source* code, which (generally) anyone has rights to modify or work with 

You can use GitHub to work on such projects 

The key object is a *pull request* ("PR"), which is a request for a project maintainer to merge ("pull") changes you've worked on into their repository 

There are a few different workflows for creating and handling PRs, which we'll walk through below 

.. _web_interface:

Quick Fixes 
---------------------

GitHub's website provides an online editor for quick-and-dirty changes, such as fixing typos in documentation 

To use it, open a file in GitHub and click the small pencil to the upper right 

.. figure:: /_static/figures/git-quick-pr.png
    :scale: 60%

Here, we're trying to add the QuantEcon link to the Julia project's README

After making our changes, we can then describe them and propose them for review by maintainers 

But what if we want to make more in-depth changes? 

.. _fork_workflow:

No-Access Case 
-----------------------

A common problem is when we don't have write access to the repo in question

To work around this, we can click the "Fork" button that lives in the top-right of every repo's main page 

.. figure:: /_static/figures/git-fork-button.png
    :scale: 60%

This will create a repo under account with the same name, contents, and history as the original. For example, `this repo <https://github.com/ubcecon/example_repository>`_ is a fork of our original `git setup <https://github.com/quanteconuser/example_repository/>`_ 

We can clone this fork and work with it in exactly the same way as we would a repo we own (because a fork *is* a repo we own)

.. figure:: /_static/figures/git-edit-fork.png
    :scale: 60%

Here, for example, we've committed and pushed some changes to the fork that we want to upstream into the main repo 

We should make sure these changes are on the server 

.. figure:: /_static/figures/git-fork-history.png
    :scale: 60%

Next, we go to the pull requests menu and click "New Pull Request." You'll see something like 

.. figure:: /_static/figures/git-create-pr.png
    :scale: 60%

This gives us a quick overview of the commits we want to merge in, as well as the end-to-end differences

Let's fill out the form and then hit create. This opens a form like this on the main repo

.. figure:: /_static/figures/git-create-pr-2.png
    :scale: 60%

The key pieces are 

* A list of the commits we're proposing 
* A list of reviewers, who can approve or modify our changes 
* Labels, Markdown space, assignees, and the ability to tag other git issues and PRs, just as with issues 

For an example of a PR, see `example pull request <https://github.com/quanteconuser/example_repository/pull/3>`_

To edit a PR, simply push changes to the fork that you opened the PR from. That is, a pull request is not like bundling up your changes and delivering them, but rather like opening an *ongoing connection* between two repositories, that is only severed when the PR is closed or merged 

Write Access Case 
----------------------

In this case, we don't need to create a fork, but will rather work with a *git branch* 

Branches in git represent parallel development streams (i.e., sequences of commits) that the PR is trying to merge 

First, load the repo in GitHub Desktop and use the branch dropdown 

.. figure:: /_static/figures/git-pr-branch.png
    :scale: 60%

Click "New Branch" and choose an instructive name (make sure there are no spaces or special characters)

This will "check out" a new branch with the same history as the old one (but new commits will be added only to this branch)

We can see the active branch in the top dropdown 

.. figure:: /_static/figures/git-branch.png
    :scale: 60%

For example, let's say we add some stuff to the Julia code file and commit it 

.. figure:: /_static/figures/git-pr-edits.png
    :scale: 60%

To put this branch (with changes) on the server, we simply need to click "Publish Branch"

Navigating to the repo page, we will see a suggestion about a new branch

.. figure:: /_static/figures/git-new-branch.png
    :scale: 60%

At which point the workflow is identical to the previous case 

Julia Package Case 
-----------------------

One special case is when the repo in question is actually a Julia project or package 

We cover that (along with package workflow in general) in the ``testing`` lecture 

Additional Resources and Troubleshooting
================================================

You may want to go beyond the scope of this tutorial when working with GitHub. For example, perhaps you run into a bug, or you're working with a setup that doesn't have GitHub Desktop installed 

Here are some resources to help 

* Kate Hudson's excellent `git flight rules <https://github.com/k88hudson/git-flight-rules/>`_, which is a near-exhaustive list of situations you could encounter, and command-line fixes 
* The GitHub `Learning Lab <https://lab.github.com/>`_, an interactive sandbox environment for git 


.. _version_control_commandline:

Command-Line Basics
----------------------------------------

Git also comes with a set of command-line tools. They're optional, but many people like using them

Furthermore, in some environments (e.g. JupyterHub installations) you may only have access to the commandline

* On Windows, downloading ``git`` will have installed a program called ``git bash``, which installs these tools along with a general Linux-style shell

* On Linux/macOS, these tools are integrated into your usual terminal

To open the terminal in a directory, either right click and hit "open git bash" (in Windows), or use Linux commands like ``cd`` and ``ls`` to navigate 

See `here <https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/command-line-101>`_ for a short introduction to the command line

As above, you can clone by grabbing the repo URL (say, GitHub's `site-policy repo <https://github.com/github/site-policy/>`_) and running ``git clone https://github.com/github/site-policy.git``

This won't be connected to your GitHub Desktop, so you'd need to use it manually (``File => Add Local Repository``) or drag-and-drop from the file explorer onto the GitHub Desktop

.. figure:: /_static/figures/git-add-local.png
    :scale: 60%

From here, you can get the latest files on the server by ``cd``-ing into the directory and running ``git pull`` 

When you ``pull`` from the server, it will never overwrite your modified files, so it is impossible to lose local changes

Instead, to do a hard reset of all files and overwrite any of your local changes, you can run ``git reset --hard origin/master``

.. Removed this, since I have never done it!
.. To remove files that aren't tracked by git (e.g., compilation byproducts and output directories), run ``git clean -fd``


Exercises
============

Exercise 1a
---------------

Follow the instructions to create a `new repository <new_repo_workflow>`_ for one of your github accounts
In this repository
* Take the code from one of your previous assignments, such as 
`Newton's method <jbe_ex8a>`_ in `Julia by example <julia_by_example>`_ (either as a ``.jl`` file or a Jupyter notebook)
* Put in a ``README.jl`` with some text
* Put in a ``.gitignore`` file, ignoring the Jupyter files ``.ipynb_checkpoints`` and the project files, ``.projects``

Exercise 1b
----------------

Pair-up with another student who has done Exercise 1a and find out their github ID, and each do the following
* Add the github ID as a collaborators on your repository
* Clone the repositories to your local desktop
* Assign each other an issue
* Submit a commit from GitHub desktop which references the issue by number
* Comment on the commits
* Ensure you can run their code without any modifications

Exercise 1c
--------------

Pair-wise with the results of Exercise 1b examine a merge-conflict by editing the ``README.md`` file for your repository that you have both setup as collaborators

Start by ensuring there are multiple lines in the file so that some changes may have conflicts, and some may not

* Clone the repository to your local desktops
* Modify **different** lines of code in the file and both commit and push to the server (prior to pulling from each other)--and see how it merges things "automatically"
* Modify **the same** line of code in the file, and deal with the `merge conflict <merge_conflict>`_

Exercise 4a
----------------

Just using the GitHub's `web interface <web_interface>`_, submit a Pull Request for a simple change of documentation to a public repository

The easiest may be to submit a PR for a typo in the source repository for these notes, i.e. ``https://github.com/QuantEcon/lecture-source-jl`` 

Note: The source for that repository is in ``.rst`` files, but you should be able to find spelling mistakes/etc. without much effort

Exercise 4b
-------------------------

Following the `instructions <fork_workflow>`_ for forking and cloning a public repository to your local desktop, submit a Pull Request to a public repository

Again, you could submit it for a typo in the source repository for these notes, i.e. ``https://github.com/QuantEcon/lecture-source-jl``, but you are also encouraged to instead look for a small change that could help the documentation in another repository.

If you are ambitious, then go to the Exercise Solutions for one of the Exercises in these lecture notes and submit a PR for your own modified version (if you think it is an improvement!)
