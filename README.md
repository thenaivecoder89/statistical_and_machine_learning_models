# machine_learning_algorithms
This repo is a consolidation of multiple ML and NN algorithms that have been implemented.
The NN models have been developed using CUDA enabled Tensorflow.
# requirements to enable tensorflow with CUDA
1. Python version 3.12  (check: run "python --version" in shell).
2. CUDA version 12.3 (check: run "nvcc --version" in shell).
3. cuDNN version 8.9.7 (not needed but can be installed as a safeguard).
# process to enable tensorflow with CUDA
1. Enable Windows Subsystem for Linux (WSL) - one-time setup:
    - About: WSL is a compatibility layer built into Windows that allows developers to run a genuine Linus environment directly inside Windows.
    - Need: WSL is needed since "tensorflow[and-cuda]" is not supported on native Windows.
    - How to Enable: run "wsl --install -d Ubuntu" in shell.
    - Check Version: run "wsl -l -v" in shell - this should show version 2. If not, run "wsl --set-version Ubuntu -2" in shell.
2. Enable WSL environment in VSCode (can also be done in PyCharm but needs professional license to access the environment):
    - From the extensions marketplace, search and install WSL.
    - From the extensions marketplace, search and install Python and Jupyter.
    - Once installed, from the WSL terminal run "python3 -V" to ensure the version reads "3.12".
3. 