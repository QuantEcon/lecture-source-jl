.. _data_statistical_packages:

.. include:: /_static/includes/header.raw


*****************************************
Data and Statistics Packages
*****************************************

.. contents:: :depth: 2

Overview
============

.. code-block:: julia 

    using Pkg 
    ctx = Pkg.Types.Context();
    @show ctx.env.project_file 
    @show ctx.env.manifest_file 
    @show ctx.env.project.name 
    @show ctx.env.project.version 
    @show pwd()

.. code-block:: julia 

    pkg"dev InstantiateFromURL"
    using InstantiateFromURL
    github_project("QuantEcon/QuantEconLectureAllPackages")

.. code-block:: julia 

    ctx = Pkg.Types.Context();
    @show ctx.env.project_file 
    @show ctx.env.manifest_file 
    @show ctx.env.project.name 
    @show ctx.env.project.version 
    @show pwd()
