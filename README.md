# machine_learning_algorithms_and_statistical_models
 - This repo is a consolidation of multiple ML and Statistical models that have been implemented as part of a reasearch exercise.
 - Data sourcing for stock analysis model:
   - Data for the stock analysis model has been sourced from the Zerodha API (KiteConnect).
   - The API uses OAuth which requires the login url (generated using login_url() function) to be called each time the data is to be collected.
   - In order to eliminate manual intervention each time, a Uvicorn server has been developed which writes the request_token into a file.
   - Steps to activate the server:
        1. Use the link "http://127.0.0.1:8000/login_url" to access the Kite Login page.
        2. Enter username, password and token number (for 2FA) when prompted.
        3. Once entered, the page will redirect and a message: {'status': 'ok'} should be displayed.
        4. Once the above steps are completed, open a new VSCode Terminal and enable virtual environment.
        5. After enabling virtual environment, navigate to the directory containing "server.py" file and run the uvicorn command to activate the same (with reload enabled).
   - Once the above steps are completed, run the "Zerodha_data_sourcing.py" program to source the necessary data.

 - The NN models have been developed using CUDA enabled Tensorflow.
# requirements to enable tensorflow with CUDA
1. Python version 3.12  (check: run "python --version" in shell).
2. CUDA version 12.3 (check: run "nvcc --version" in shell).
3. cuDNN version 8.9.7 (not needed but can be installed as a safeguard).
# process to enable tensorflow with CUDA
1. Enable Windows Subsystem for Linux (WSL) - one-time setup:
    - About: WSL is a compatibility layer built into Windows that allows developers to run a genuine Linux environment directly inside Windows.
    - Need: WSL is needed since "tensorflow[and-cuda]" is not supported on native Windows.
    - How to Enable: run "wsl --install -d Ubuntu" in shell.
    - Check Version: run "wsl -l -v" in shell - this should show version 2. If not, run "wsl --set-version Ubuntu -2" in shell.
    - Enter username and password when prompted (VVIP: Safely store this password as this will be needed everytime a new login is made).
2. Enable WSL environment in VSCode (can also be done in PyCharm but needs professional license to access the environment):
    - From the extensions marketplace, search and install WSL.
    - From the extensions marketplace, search and install Python and Jupyter.
    - Once installed, from the WSL terminal run "python3 -V" to ensure the version reads "3.12".
    - Install "pip" in the WSL environment by running the following:
        
        a. sudo apt update
        
        b. sudo apt install -y python3-pip python3-venv
    
    - Verify the installations by running the following:
        
        a. python3 -m pip --version
        
        b. pip3 --version
        
        c. which pip3 # this should show /usr/bin/pip3

3. Enable virtual environment in WSL (and "pip" in the virtual environment):
    - Run the following commands in the WSL terminal:
        
        a. python3 -m venv .<name_of_virtual_environment>
        
        b. source .<name_of_virtual_environment>/bin/activate #this needs to be run everytime the virtual environment is to be activated for pip installations.
        
        c. python -m pip install --upgrade pip setuptools wheel

4. Rename/ deactivate virtual environment:
    - In case renaming of the virtual environment is needed, recommended practice is to deactivate, remove the current environment and create a new environment by running:
        
        a. deactivate
        
        b. rm -rf .<name_of_virtual_environment>
        
        c. pythom3 -m venv .<name_of_new_virtual_environment>
    
    - Once a new envornment is created, re-run the pip upgrade command as instructed in item 3.c.
5. Installing tensorflow (with CUDA):
    - In shell, run "wsl -l -v" to make sure WSL is up and running.
    - In shell, run "nvidia-smi" to make sure CUDA is available.
    - In WSL virtual environment, run "pip install "tensorflow[and-cuda]"
    - Run the following script to check tensorflow is running with CUDA and has Keras:
        
        a. import tensorflow as tf
        
        b. print("TensorFlow version:", tf.__version__)
        
        c. print("Built with CUDA:", tf.test.is_built_with_cuda())
        
        d. print("Physical GPUs:", tf.config.list_physical_devices('GPU'))
        
        e. print("Default GPU device name:", tf.test.gpu_device_name()) # this should show the GPU device name exactly as per actual GPU name.
        
        f. print(f'tf.keras exists: {hasattr(tf, 'keras')}')

# optional process to connect git repo with WSL (without SSH):
1. From github website, copy the SSH link. If repo is public no need for additional steps but if private, personal access token (PAT) will need to be created.
2. In WSL environment (not virtual) run the following commands (without the <> brackets):
    - git config --global user.name "<your_git_username>"
    - git config --global user.email "<your_git_email>"
    - git congig --global credential.helper store
3. Creating a personal access token:
    - GitHub → Settings → Developer settings → Personal access tokens.
    - Create a fine-grained (or classic) token.
    - Minimum permissions/scopes for push:
        
        a. Fine-grained: select your repo, grant Contents: Read and write.
        
        b. Classic: scope repo.
    
    - Copy the token (you’ll paste it once; with a helper it’s remembered).
    - If your org uses SSO, authorize the token for the org when prompted.
4. Cloning repo in WSL:
    - To clone the repo in WSL, run the following commands:
        
        a. mkdir -p ~/projects
        
        b. cd ~/projects
        
        c. git clone https://github.com/<paste_SSH_link_copied_in_step_1>
        
        d. enter PAT when prompted

5. Test the repo:
    - run "ls -la" to see the folder that was created - this should have the same name as the git repo.
    - run "cd <folder_name>"
    - run "pwd" to make sure you're in the same directory as the repo.
    - run "git status" # this should say "On branch ..."
    - Make a change to the README.md file.
    - run "git status" to see the files ready for commit.
    - run "git add "README.md"" to stage a file; "README.md" can be replaced with any other filename later. Run "git restore --staged "<file_name>" to unstage a file.
    - run "git commit -m "<some_message>"
    - run "git push"
# requirements to enable data frames with CUDA - RAPIDS cuDF + connectorx (leads to ~70% drop - on average - in processing times {tested on a dataset of 20Mn records})
1. Core requirement: Python 3.12 and CUDA 13 in a WSL environment.
2. Key note: Since we already have tensorflow enbaled with CUDA, we chose to setup RAPIDS in a different virtual environment (to avoid conflicts). Furthermore, for ease of use, the environment was exposed on a separate Jupyter kernel that allows us to switch between tensorflow and rapids as needed.
3. Setup steps:
    - Pre-checks: run "nvidia-smi" and "python3 --version" in the WSL environment to verify CUDA and Python versions.
    - Step 1: create a clean virtual environment for rapids.

        a. run "python3 -m venv ~/<name_of_your_environment>" to setup your v_env and activate the same using "source ~/<name_of_your_environment>/bin/activate".

        b. run "python -m pip install -U pip setuptools wheel" as good hygneie.

    - Step 2: install CUDA runtime pieces (Python Wheels).

        a. run "python -m pip install -U nvidia-cuda-runtime nvidia-cublas"

    - Step 3: install RAPIDS (cuDF) + friends (dask, pyarrow, fastparquet) for CUDA 13.

        a. run "python -m pip install -U cudf-cu13 dask-cudf-cu13 rmm-cu13" to install core RAPIDS data frame libraries.

        b. run "python -m pip install -U cupy-cuda13x" to enable GPU numpy.

        c. run "python -m pip install -U pandas pyarrow fastparquet" to enable parquet/csv helpers.

    - Step 4: register Jupyter kernel for the virtual environment - for ease of access to the environment.

        a. run "python -m pip install -U ipykernel" and "python -m ipykernel install --user --name=rapids-cuDF --display-name=<display_name_of_your_environment>"

    - Step 5: setup "connectorx" for parallelization and direct data read during data calls from a database - further reduces data fetch time.

        a. run "python -m pip install -U connectorx"

    - Step 6: install CUDA APT repo and expose library path - to be executed inside the WSL terminal (system level).

        a. run "source /etc/os-release" and "echo "$VERSION_ID"" to check Ubuntu version (22.04: jammy/ 24.04: noble)

        *Important Note: The following steps are only for 24.04: noble.*

        b. run "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" to download the nvidia apt repo.

        c. run "sudo dpkg -i cuda-keyring_1.1-1_all.deb" and "sudo apt-get update" to add the nvidia cuda repo keyring.

        d. run "sudo apt-get install -y cuda-toolkit-13-0" to install the toolkit. Includes "libnvrtc.so.13".

        e. run "echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc" and "source ~/.bashrc" to expose library path.