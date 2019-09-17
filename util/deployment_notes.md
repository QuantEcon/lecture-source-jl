## Major deployment Instructions

If we are making changes to the HTML that are only cosmetic, then no need to deploy the notebooks or full setup.  But if the notebooks change significantly (and especially if there is any TOML changes!)  Then we need a full deployment

1. Make sure the `Project.toml` is tagged (that is, bump the version number in the actual `Project.toml`, and then update the `deps_generic.jl` file to use the new version number.)

2. Run `make lectures-julia` in the AWS admin repo to queue up a preview build.

3. Make sure things look good.

4. From the AWS admin repo, run `scp -i keys/quantecon-build.pem ubuntu@build.quantecon.org:~/repos/lecture-source-jl/_build/website/jupyter/_downloads ~/Desktop` (or wherever you want to copy it to.)

5. Check that those executed notebooks (in executed) look good, and manually copy them over into your clone of quantecon-notebooks-julia. Push this up and then tag a new release of that repo.

6. Update the version numbers for the hubs and rebuild, so that both jupyerhubs will be synced to the correct package versions for the deployed notebooks. (i.e., `sudo docker build . -t "ubcecon/vse-jupyterhub:latest"`, etc.)

7. Push your changes to github and dockerhub.

8. Run `make lectures-julia-live` in the AWS admin repo when you're ready to go live. Run `reset-servers.sh` in the vse-jupyterhub repo on the syzygy when you're ready to make the switch. Also run `sudo docker system prune` to nuke old (now untagged images.)

