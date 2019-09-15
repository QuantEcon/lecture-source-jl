## Major deployment Instructions
If we are making changes to the HTML that are only cosmetic, then no need to deploy the notebooks or full setup.  But if the notebooks change significantly (and especially if there is any project.toml changes!)  Then we need a full deployment

1. Make sure the Project.toml is tagged (not sure what is required?) with a version we can update in the output
2. Update the tagged version in the literal include scripts
3. Initiate a build of the HTML with intention that it would go live with the correct versions
4. Build the notebooks for deployment on the notebook site.
5. Update the version numbers for the hubs (or is that even needed?) so that both jupyerhubs will be synced to the correct package versions for the deployed notebooks.
6.  Wait until (a) the jupyterhub images have been built, (b) the notebook build; and (c) the website build are all complete
7. Deploy the notebooks to the notebook server, deploy the html, and deploy the jupyterhubs
8. Run the script to stop all servers for the jupyterhubs and delete all .julia for all users.
