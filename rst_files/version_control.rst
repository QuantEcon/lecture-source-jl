.. _version_control:

.. include:: /_static/includes/lecture_howto_jl.raw

******************************************
Introduction to Git and Version Control
******************************************

Co-authored with Arnav Sood

An essential part of modern software engineering is using version control

We use version control because 

* Not all iterations on a file are perfect 
* We want to be able to see who has changed what and how 
* We want a uniform version scheme to do this between people and machines 

In this lecture, we'll discuss how it works on the GitHub platform

.. contents:: :depth: 2

Setup 
==================

First, make sure you create an account on `GitHub.com <http://github.com/>`_

If you are a student, be sure to use the GitHub `Student Developer Pack <https://education.github.com/pack/>`_

Otherwise, see if you qualify for a free `Non-Profit/Academic Plan <https://help.github.com/articles/about-github-education-for-educators-and-researchers/>`_

These come with things like unlimited private repositories, testing support, etc.

Next, install ``git`` and the GitHub Desktop application 

1. Follow the instructions for installing `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git/>`_

2. Follow the instructions for installing `GitHub Desktop <https://desktop.github.com/>`

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

Of these, the README is the most important

Individual Workflow
====================================

In this section, we'll describe how to use GitHub to version your own projects 

Much of this will carry over to the collaborative section 

Creating a Repository 
---------------------------------

In general, we will always want to make new repos using the following dropdown 

.. figure:: /_static/figures/git-makerepo.png
    :scale: 120%

We can then configure repository options as such 

.. figure:: /_static/figures/git-makerepo-full.png
    :scale: 120%

In this case, we're making a public repo ``github.com/arnavs/git-setup``, which will come with a ``README.md``, is licensed under the MIT License, and will ignore Julia compilation byproducts 

Cloning a Repository 
---------------------------------------

The next step is to get this to our local machine 

.. figure:: /_static/figures/git-clone.png
    :scale: 120%

This dropdown gives us a few options 

* "Open in Desktop" will call to the GitHub Desktop application that we've installed 
* "Download Zip" will download the directory *without the .git* subdirectory. 
* The copy/paste button next to the link lets us use the command line, i.e. ``git clone https://github.com/arnavs/git-setup.git``. 

Making and Managing Changes 
-------------------------------------------

Now that we have the repository, we can start working with it 

For example, let's say that we've amended the ``README.md`` (using our editor of choice), and also added a new file ``economics.jl`` which we're still working on 

Returning to GitHub Desktop, we should see something like 

.. figure:: /_static/figures/git-desktop-commit.png
    :scale: 120%

To select individual files for commit, we can use the check boxes to the left of each file 

Let's say you select only the README to commit. Going to the history tab should show you our change 

.. figure:: /_static/figures/git-desktop-commit2.png
    :scale: 120%

The Julia file is unchanged 

Pushing to the Server 
--------------------------------

As of now, this commit lives only on our local machine. To upload it to the server, simply click the "Push Origin" button atop the screen

The small "1^" to the right of the text indicates we have one commit to upload 

Reading and Reverting History 
-----------------------------------------

As mentioned, one of the key features of GitHub is the ability to scan through history 

By clicking the "commits" tab on the repo front page, for example, we see `this page <https://github.com/arnavs/git-setup/commits/master/>`_

Clicking an individual commit gives us the granular view, (e.g., `here <https://github.com/arnavs/git-setup/commit/5ed516c7949dee5c60ec12be10d26e1bdee23ca5/>`_)

Sometimes, however, we want to not only inspect what happened before, but go back to it 

* If you haven't made the commit yet, just right-click the file and hit "discard changes" to reset the file to the last known commit 
* If you have made the commit but haven't pushed to the server yet, go to the "history" tab as above, right click the commit and click "revert this commit." This will create the inverse commit, as above 

.. figure:: /_static/figures/git-revert-commit.png
    :scale: 120%

Working across Machines
--------------------------------------

Oftentimes, you will want to work on the same project across multiple machines (e.g., a home laptop and a lab workstation)

The key is to push changes from one machine, and then to pull changes from the other machine 

Pushing can be done as above. To pull, simply click pull under the "repository" dropdown at the top of the screen 

.. figure:: /_static/figures/git-pull.png
    :scale: 120%

Collaborative Work
==================================

Adding Collaborators
----------------------------

First, let's add a collaborator to the ``arnavs/git-setup`` lecture we created earlier 

We can do this by clicking "settings => collaborators," as follows

.. figure:: /_static/figures/git-collab.png
    :scale: 120%

Project Management 
--------------------------------

GitHub's website also comes with project management tools to coordinate work between people 

The main one is an *issue*, which we can create from the issues tab. You should see something like this

.. figure:: /_static/figures/git-issue.png
    :scale: 120%

Let's unpack the different components 

* The *assignees* dropdown lets you select people tasked to work on the issue 

* The *labels* dropdown lets you tag the issue with labels visible from the issues page, such as "high priority" or "feature request" 

* It's possible to tag other issues and collaborators (including in different repos) by linking to them in the comments. This is part of what's called *GitHub-Flavored Markdown* 

For an example of an issue, see `here <https://github.com/arnavs/git-setup/issues/1/>`_ 

The checkbox idiom is a common one to manage projects in GitHub 

Reviewing Code 
------------------------------

There are a few different ways to review people's code in GitHub 

* Whenever people push to a project you're working on, you'll receive an email notification
* You can also review individual line-items or commits by opening commits in the granular view as `above <https://github.com/arnavs/git-setup/commit/5ed516c7949dee5c60ec12be10d26e1bdee23ca5/>`_

.. figure:: /_static/figures/git-review.png
    :scale: 120%

Merge Conflicts
----------------------------

Any project management tool needs to figure out how to reconcile conflicting changes between people

In GitHub, this event is called a "merge conflict," and occurs whenever people make conflicting changes to the same *line* of code 

Note that this means that two people touching the same file is OK, so long as the differences are compatible 

A common use case is when we try to push changes to the server, but someone else has pushed conflicting changes. GitHub will give us the following window 

.. figure:: /_static/figures/git-merge-conflict.png
    :scale: 120%

* The warning symbol next to the file indicates the existence of a merge conflict 
* The viewer tries to show us the discrepancy (I changed the word repository to repo, but someone else tried to change it to "repo" with quotes)

To fix the conflict, we can go into a text editor (such as Atom or VS Code). Here's an image of what we see in Atom 

.. figure:: /_static/figures/atom-merge-conflict.png
    :scale: 120%

Let's say we click the first "use me" (to indicate that my changes should win out), and then save the file. Returning to GitHub Desktop gives us a pre-formed commit to accept 

.. figure:: /_static/figures/git-merge-commit.png
    :scale: 120%

Open-source Projects 
======================================

One of the defining features of GitHub is that it is the dominant platform for *open-source* code, which (generally) anyone has rights to modify or work with 

You can use GitHub to work on such projects 

Quick Fixes
--------------------

GitHub's website provides an online editor for quick-and-dirty changes, such as fixing typos 

To use it, open a file in GitHub and click the small pencil to the upper right 

.. figure:: /_static/figures/git-quick-pr.png
    :scale: 120%

Here, we're trying to add the QuantEcon link to the Julia project's README

After making our changes, we can then describe them and propose them for review by maintainers 

But what if we want to make more in-depth changes? 

Forking and Pull Requests 
-------------------------------------

The first problem to solve is that we don't have write access (usually) to open-source repos 

To work around this, we can click the "Fork" button that lives in the top-right of every repo's main page 

This will create a repo under account with the same name, contents, and history as the original. For example, `this repo <https://github.com/ubcecon/git-setup/>`_ is a fork of our original `git setup <https://github.com/arnavs/git-setup/>`_ 

Making Changes 
----------------------------

We can clone this fork and work with it in exactly the same way as we would a repo we own (because a fork *is* a repo we own)

In particular, we can follow the same process of: 

* Updating the fork via sequences of commits, which we push and pull using GitHub Desktop 

* Collaborating with other people using issues 

* Looking at history using the GitHub website

The Pull Request 
---------------------------

Eventually, you will want to upstream your changes into the main repository 

The first thing you want to do is go to the pull requests menu and click "New Pull Request." You'll see something like 

.. figure:: /_static/figures/git-create-pr.png
    :scale: 120%

This gives us a quick overview of the commits we want to merge in, as well as the end-to-end differences

Clicking through gives us a window like 

.. figure:: /_static/figures/git-create-pr-2.png
    :scale: 120%

The key pieces are 

* A list of the commits we're proposing 
* A list of reviewers, who we can ask to approve or modify our changes 
* Labels, Markdown space, assignees, and the ability to tag other git issues and PRs, just as with issues 

For an example of a PR, see `here <https://github.com/arnavs/git-setup/pull/2#pullrequestreview-170918768/>`_

To edit a PR, simply push changes to the fork that you opened the PR from. That is, a pull request is not like bundling up your changes and delivering them, but rather like opening an *ongoing connection* between two repositories, that is only severed when the PR is closed or merged 

Additional Resources and Troubleshooting
================================================

You may want to go beyond the scope of this tutorial when working with GitHub. For example, perhaps you run into a bug, or you're working with a setup (like the QuantEcon Docker image) that doesn't have GitHub Desktop installed 

Here are some resources to help 

* Kate Hudson's excellent `git flight rules <https://github.com/k88hudson/git-flight-rules/>`_, which is a near-exhaustive list of situations you could encounter, and command-line fixes 
* The GitHub `Learning Lab <https://lab.github.com/>`_, an interactive sandbox environment for git 

Command-Line Basics
----------------------------------------

Git also comes with a set of command-line tools. They're optional, but many people like using them

* On Windows, downloading ``git`` will have installed a program called ``git bash``, which installs these tools along with a general Linux-style shell

* On Linux/macOS, these tools are integrated into your usual terminal

To open the terminal in a directory, either right click and hit "open git bash" (in Windows), or use Linux commands like ``cd`` and ``ls`` to navigate 

See `here <https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/command-line-101>`_ for a short introduction to the command line

As above, you can clone by grabbing the repo URL (say, GitHub's `site-policy repo <https://github.com/github/site-policy/>`_) and running ``git clone https://github.com/github/site-policy.git``

This won't be connected to your GitHub Desktop, so you'd need to use it manually (``File => Add Local Repository``)

.. figure:: /_static/figures/git-add-local.png
    :scale: 120%

From here, you can pull by ``cd``-ing into the directory and running ``git pull`` 

To do a hard reset of all tracked files, you can run ``git reset --hard origin/master``

To remove files that aren't tracked by git (e.g., compilation byproducts and output directories), run ``git clean -fd``