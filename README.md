# ICRA 2020 Learning Haptics project

## Environment Setup
### Install Mujoco
We recommend installing mujoco in a python virtual environment. Create and activate a virtual environment using:
```
$ virtualenv -p /path/to/python/python.3.6 /path/to/virtualenv/my_venv_name
$ source /path/to/virtualenv/my_venv_name/bin/activate
```

To install mujoco, go to [mujoco's website](https://www.roboti.us/) and download the latest version of mujoco for your operating system. We are using mjpro200.  

Unzip mjpro200_linux.zip file:
```
$ unzip mjpro200_linux.zip $HOME./mujoco/mujoco200
```

Set the following environment variables either in your ~/.bashrc or ~/.profile
```
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nicolai/.mujoco/mujoco200/bin
```

Get a key from the mujoco website and store it in $ HOME/.mujoco/mjkey.txt.

### Clone this repository
```
$ git clone https://github.umn.edu/RSN-Robots/corl2019_learningHaptics.git
$ cd corl2019_learningHaptics
```

### Install mujoco-py
Install the prerequisites
```
$ sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Next, install all the required packages with

```
$ pip install -r requirements.txt
```

## Training
We provide a training script. Refer to the following code snippet for usage.

```
usage: train.py [-h] (--mdp | --pomdp) --model_path MODEL_PATH
                [--device DEVICE] [--n_states N_STATES]
                [--n_actions N_ACTIONS] [-b BATCH_SIZE] [-j GAMMA]
                [--epochs N] [--lr LR] [--momentum M]
                [--output_dir OUTPUT_DIR] [--save_freq SAVE_FREQ] [--sim]

PyTorch Tactile Training

optional arguments:
  -h, --help            show this help message and exit
  --mdp                 train model using MDP
  --pomdp               train model using POMDP
  --model_path MODEL_PATH
                        XML model to load
  --device DEVICE       device
  --n_states N_STATES   state space size
  --n_actions N_ACTIONS
                        action space size
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -j GAMMA, --gamma GAMMA
                        future reward decay
  --epochs N            number of total epochs to run
  --lr LR               initial learning rate
  --momentum M          momentum
  --output_dir OUTPUT_DIR
                        path where to save
  --save_freq SAVE_FREQ
                        frequency to save checkpoints
  --sim                 whether to run in simulation mode or on a real robot

```
