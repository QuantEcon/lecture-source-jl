Here are the steps required to issue a new version of the lectures (e.g., to upgrade to a new Julia version.)

This is assuming that you're not changing the `InstantiateFromURL` package. Which is something you should not do lightly, as you invalidate every user's setup.

1. Bump references in the code (the literalinclude and the `getting_started` lecture RST) using the old version number, to the anticipated new version number.

2. Bump the version number of the `source/rst/Project.TOML` to the new version number.

3. Run a preview build using `make julia` using `make julia-preview` in the AWS Admin repo. If it looks good, go live on the HTML with `make julia-live`.

4. Download the executed notebooks using `scp -r ubuntu@build.quantecon.org:~/repos/lecture-source-jl/_build/website/jupyter/executed ~/some/local/path`.

5. Move them into the `quantecon-notebooks-julia` repo, and push.

6. Issue a new release of the `quantecon-notebooks-julia` repo with the version number used above. 

7. Update the `quantecon.syzygy.ca` (`ssh ptty2u@quantecon.syzygy.ca`, contact Ian Allison or Arnav Sood if you need access) by bumping the Dockerfile to use the new version in its InstantiateFromURL call. 

7a. On the syzygy, we do this simply by rebuilding the Docker image. Pull the `quantecon-syzygy` repo down and then run `sudo docker build . -t "quantecon/syzygy` once you're inside.)

8. Update the `quantecon.syzygy.ca` to use a new Julia version number if necessary (involves updating the checksum.)

9. Put out some sort of release announcement on the QuantEcon website and Discourse.
