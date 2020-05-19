
## Website

1. Run a preview build using `make julia` using `make julia-preview` in the AWS Admin repo. If it looks good, go live on the HTML with `make julia-live`.  Arnav: anything else you can add here or does this require logging into the AWS server?  Wasn't there some way to do it in slack or something like that?

How to initiate a full coverage test, etc? 

## Pushing Notebooks

1. Download the executed notebooks using `scp -r ubuntu@build.quantecon.org:~/repos/lecture-source-jl/_build/website/jupyter/executed ~/some/local/path`.
2. Move them into the `quantecon-notebooks-julia` repo, and push.
3. Run the colab python script?  Describe it 
4. Move them into the `quantecon-notebooks-julia-colab` repo, and push.

## Full Package Manifest Update

Hints on how to Update and Add Packages
- The basic process to updating packages.  Just `]up`, run coverage (locally?) and then update?  Or do you always run on the build server?

Then
- Pick version number to tag it at?
- Modify source/_static/includes/deps_generic.jl to reflect the version number
- Modify source/rst/getting_started_julia/getting_started.rst  as well.
- Do we need to do anything on the build server side or is that stuff automatic?
- Issue a new release of the `quantecon-notebooks-julia` repo with the version number used above. 

## Upgrading to a new Julia Version
- Where do we change the kernelspec?
- If we have colab, point out where to modify the colab installation code
- Do we need to do anything on the build server side or is that stuff automatic?
- Put out some sort of release announcement on the QuantEcon website and Discourse.
