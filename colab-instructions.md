## Colab Instructions

1. Load the notebook using the launcher at the bottom of the page, ignoring any errors about a missing kernel. 

2. Run the following cell in the notebook

```
# Installation cell
%%shell
if ! command -v julia 3>&1 > /dev/null
then
    wget 'https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.1-linux-x86_64.tar.gz' \
        -O /tmp/julia.tar.gz
    tar -x -f /tmp/julia.tar.gz -C /usr/local --strip-components 1
    rm /tmp/julia.tar.gz
fi
julia -e 'using Pkg; pkg"add IJulia; precompile;"'
echo 'Done'
```

3. After it says "done," in the top menu, click "Edit," then "Notebook Settings," and then choose "Julia 1.4" as your runtime. Hit "Save."

4. To test your setup, run a Julia command in the window (something like `versioninfo()`.) If it doesn't work, try refreshing your browser window in between steps (2) and (3).

