## Preliminary 

Get Matt to add you to the `https://github.com/mmcky/QuantEcon.aws.admin` repo. That's the one which executes all the build commands on the ANU server (which you should also get ssh access to; it's `build.quantecon.org` and the username is `ubuntu`.)

## Steps for a Full Release 

1. Merge all RST changes into the source repo. 

2. To test locally, run `make coverage` (which will also execute the notebooks) or `make test` (which will generate the notebooks with tests, but no execution). Get tests passing. 

3. To test on the AWS server, go to the AWS admin repo, and run `make julia`. (If you need to test changes to non-RST files, like TOML or themes, run `make julia-clean`, but this will be sequential.) Coverage happens automatically with both of these. 

4. If package changes are required, note the following steps: 

    A. A good starting point is to run `] up`. The TOML lives in the `source/rst` folder under the source repo. This happens locally, and then gets pushed. 

    B. Once you get a set of packages that pass tests locally, try running a clean build on the server as above. 

    C. Make sure you bump the version number in the TOML to whatever you intend the version number of the next release to be. (This will be the version of the tagged set of notebooks in the `QuantEcon/quantecon-notebooks-julia` and `.../quantecon-notebooks-julia-colab` repos.)

    D. But, note that those repos don't need to be populated; the coverage etc. will run against the TOML in the `source/rst`, which will get carried over. 

    E. Also, good practice to do a global find/replace for the old version number, and update as appropriate (but don't do this blindly, because it may pop up in innocuous places; e.g. a package numbered v0.7.0.) Mainly, though, the version numbers are used in the `source/_static/includes/deps_generic.jl` and the getting started lecture. 

5. If you're also bumping Julia versions, make sure you change the `conf.py` kernelspec to use the new one. Otherwise, each coverage will trivially fail. And colab will fail. **If you're updating the Julia version, make sure you also bump it in the Colab install script under source/_static/includes**. 

    A. Don't forget to install the new Julia version on the ANU server. And you'll need to run something like `sudo ln -fs new_julia /usr/bin/julia`, so that running `julia` on the machine points to the new version. (TODO: Check syntax.) Note that you should install `InstantiateFromURL` and `IJulia` on the new version, and ideally grab the manifest version you intend to use with a `precompile = true`. 

6. Once you get this stuff in a place that looks good (e.g., coverage on the website is 100%, etc.), you can go live with `make julia-live` from the AWS repo. 

7. To update the notebooks, first download them using

    > scp -r ubuntu@build.quantecon.org:~/repos/lecture-source-jl/_build/website/jupyter/executed ~/some/local/path

8. The executed notebooks you just downloaded can be dragged/dropped as-is into the `../quantecon-notebooks-julia` repository. Then push them, tag a release with the appropriate version number. 

9. For colab, drag these notebooks into the `../quantecon-notebooks-julia-colab` repo. And then run the colab build script that lives inside that repo before pushing. Tag a release (ideally with the same version number as the other repo; no need to keep these separate.)

10. Hit the binder button in the `../quantecon-notebooks-julia` repo to get it cached. 

## Other Notes

**Theme Updates**: TODO. (Need to get latest instructions from Matt on where this lives.) And note that the default launch option is not the first in the list of launch options (the default is given separately.)

**Re-Triggering Builds**: If your build on the ANU server doesn't go through, `ssh` into it, delete the lockfile (you'll see something like `lecture-source-jl.lock` in the home directory), and kill all Julia processes (identified using `ps aux | grep julia`) and then `sudo kill -9` or `kill -9`. 

**Package Changes**: Note that simply running `] up` might not actually update everything, since the resolver is trying to solve a joint optimization problem. So, it's good practice to check the results of `] up` against what the latest versions of packages actually are. If they're behind, try running things like `] add package@latest_version` to force it to give you the latest of important packages. 

**PDFs**: TODO. Fill in with stuff from Aakash and Matt. Don't know much about this. 

**Build results**: The `#notifications` channel in the QE slack will alert you to builds when completed. The `make julia` ones (only RST changes) are relatively quick. A full clean build takes about 6 hours. 