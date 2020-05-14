## Colab Instructions

1. Load the notebook using the launcher at the bottom of the page, ignoring any errors about a missing kernel. 

2. Run the following cell in the notebook

```
!curl -sSL "https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.1-linux-x86_64.tar.gz" -o julia.tar.gz
!tar -xzf julia.tar.gz -C /usr --strip-components 1
!rm -rf julia.tar.gz*
!julia -e 'using Pkg; pkg" up; add IJulia ;  precompile"'
!echo "Julia Installed!"
```

3. In the top menu, click "Edit," then "Notebook Settings," and then choose "Julia 1.4" as your runtime. Hit "Save."

4. To test your setup, run a Julia command in the window (something like `versioninfo()`.) If it doesn't work, try refreshing your browser window in between steps (2) and (3).

