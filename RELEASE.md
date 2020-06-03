## Preliminary

Get Matt to add you to the `https://github.com/mmcky/QuantEcon.aws.admin` repo. That's the one which executes all the build commands on the ANU server (which you should also get ssh access to; it's `build.quantecon.org` and the username is `ubuntu`.)

**Note** That whenever it says to run something in the AWS repo, it means navigate to that directory **on your local machine**, and then run the command. The repo comes with the `ssh` key to run things on the `build.quantecon.org` server.

## Steps for a Full Release

1. Merge all RST changes into the source repo.

2. To test locally, run `make coverage` (which will also execute the notebooks) or `make test` (which will generate the notebooks with tests, but no execution). Get tests passing.

    A. Note that to do this in parallel you can do something like `make coverage -e parallel=4`, depending on how many cores you have available. But if you're doing parallel, it's good practice to have the packages all precompiled first.

3. To test on the AWS server, go to the AWS admin repo, and run `make julia`. (If you need to test changes to non-RST files, like TOML or themes, run `make julia-clean`, but this will be sequential.) Coverage happens automatically with both of these.

4. If package changes are required, note the following steps:

    A. A good starting point is to run `] up`. The TOML lives in the `source/rst` folder under the source repo. This happens locally, and then gets pushed.

    B. Once you get a set of packages that pass tests locally, try running a clean build on the server as above.

    C. Make sure you bump the version number in the TOML to whatever you intend the version number of the next release to be. (This will be the version of the tagged set of notebooks in the `QuantEcon/quantecon-notebooks-julia` and `.../quantecon-notebooks-julia-colab` repos.)

    D. But, note that those repos don't need to be populated; the coverage etc. will run against the TOML in the `source/rst`, which will get carried over.

    E. Also, good practice to do a global find/replace for the old version number, and update as appropriate (but don't do this blindly, because it may pop up in innocuous places; e.g. a package numbered v0.8.0.) Mainly, though, the version numbers are used in the `source/_static/includes/deps_generic.jl` and the getting started lecture.

5. If you're also bumping Julia versions, make sure you change the `conf.py` kernelspec to use the new one. Otherwise, each coverage will trivially fail. And colab will fail. **If you're updating the Julia version, make sure you also bump it in the Colab install script under source/_static/includes**.

    0. Run `] add IJulia InstantiateFromURL` locally after install to update to the latest IJulia (for minor updates, can simply do `] build`.)

    1. Don't forget to install the new Julia version on the ANU server. Canonically these live in `~/applications`. Navigate there and run (e.g.) `wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.2-linux-x86_64.tar.gz | tar -xzv`.

    2. To make this the default, go to the home directory on that machine, run `vim .bashrc`, and then edit the path to Julia (should be top line in the file) to the new version. Then, either log out and ssh again, or run `source .bashrc` to pull the changes.

    3. As above, install `InstantiateFromURL` and `IJulia` on the new version, and ideally grab the manifest version you intend to use with a `precompile = true`. For minor updates, can simply run `] build`.
    
    4. When bumping Julia versions, you may have to **adjust tolerances on coverage** or **run `make-julia` twice to get coverage back up.** 

6. Once you get the source in a state that looks good (e.g., coverage on the website is 100%, etc.), you can go live with `make julia-live` from the AWS repo.

7. To update the notebooks, first download them using

    > scp -r ubuntu@build.quantecon.org:~/repos/lecture-source-jl/_build/website/jupyter/executed ~/some/local/path

8. The executed notebooks you just downloaded can be dragged/dropped as-is into the `../quantecon-notebooks-julia` repository. Then push them, tag a release with the appropriate version number.

   To tag a release, navigate to `https://github.com/QuantEcon/quantecon-notebooks-julia/releases`, click "Draft a New Release," and fill in the data. **The version number should be something like v0.8.0, with the v**. 

9. For colab, drag these notebooks into the `../quantecon-notebooks-julia-colab` repo. And then run the colab build script that lives inside that repo before pushing. Tag a release (ideally with the same version number as the other repo; no need to keep these separate.)

10. Hit the binder button in the `../quantecon-notebooks-julia` repo to get it cached.

11. If appropriate, submit a PR to the `QuantEcon/website` repo with a news item (see https://github.com/QuantEcon/website/pull/7 for an example of how to format the PR.)

## Other Notes

**Theme Updates**: TODO. (Need to get latest instructions from Matt on where this lives.) And note that the default launch option is not the first in the list of launch options (the default is given separately.)

**Re-Triggering Builds**: If your build on the ANU server doesn't go through, `ssh` into it, delete the lockfile (you'll see something like `lecture-source-jl.lock` in the home directory), and kill all Julia processes (identified using `ps aux | grep julia`) and then `sudo kill -9` or `kill -9`.

**Package Changes**: Note that simply running `] up` might not actually update everything, since the resolver is trying to solve a joint optimization problem. So, it's good practice to check the results of `] up` against what the latest versions of packages actually are. If they're behind, try running things like `] add package@latest_version` to force it to give you the latest of important packages.

**PDFs**: To trigger a PDF build, run `make julia-pdf` from the AWS build repo. The PDF is a bit fragile (pre-Jupyterbook), e.g. doesn't include any `svg` plots (such as those from `VegaLite`), so check it over carefully for errors.

**Build results**: The `#notifications` channel in the QE slack will alert you to builds when completed. The `make julia` ones (only RST changes) are relatively quick. A full clean build takes about 6 hours.
