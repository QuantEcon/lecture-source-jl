## Colab Instructions
Colab does not have direct Julia support, and it must be installed each time you are working on a notebook.  Even after installation, there will be more precompiling latency since we do not compile every package used in the notes in the setup step. 

Instructions:

1. Ignore any errors about a missing kernel when loading the notebook.  Choose `Run Anyway` when it prompts that the notebook wasn't authored by google.

2. Run the cell below this with `Shift+Enter`.
    -  It will **3 to 8 minutes** the first time you run it for Julia and key packages to install
    - Afterwards, the colab container associated with the notebook will be activate for some time, but will likely be recycled after 60-90 minutes after closing the notebook, at which point you would need to install again.
    - After julia has been installed, you do not need to run the cell again, and errors may be safely ignored
3. Refresh your browser, and then execute any code as normal.
  - Even after installation, there will be more precompiling latency since we do not compile every package used in the notes in the setup step. 
