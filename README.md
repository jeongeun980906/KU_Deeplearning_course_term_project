# KU-DEEP-LEARNING Term Project

### This code is on-going

## Install
```
pip install -r requrements.txt
```

## Files

### 0.teleop.ipynb
Contains keyboard teleoperation demo.

Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of rthe otations.

SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.


### 1.Visualize.ipynb

It contains downloading dataset from huggingface and visualizing it.

First, download the dataset
```
python download_data.py
```


```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
root = './dataset/demo_data'
dataset = LeRobotDataset('Jeongeun/deep_learning_2025',root = root )
```
Running this code will download the dataset independatly.

### 2.transform.ipynb
Define the action and observation space for the environment. 
```python
action_type = 'delta_joint'  # Options: 'joint','delta_joint, 'delta_eef_pose', 'eef_pose'
proprio_type = 'eef_pose' # Options: 'joint', 'eef_pose'
observation_type = 'image' # options: 'image', 'object_pose'
image_aug_num = 2  # Number of augmented images to generate per original image
transformed_dataset_path = './dataset/transformed_data'
```

Based on this configuration, it will transform the actions into the action_type and create new dataset for training. 

- action_type: representation of the actions. Options: 'joint','delta_joint','eef_pose','delta_eef_pose'
- proprio_type: representations of propriocotative informations. Options: eef_pose, joint_pos
- observation_type: whether to use image of a object position informations. Options: 'image','objet_pose'
- image_aug_num: the number of augmented trajectories to make when you are using image features

You can just use the python script to do this as well. 

```
python transform.py --action_type delta_eef_pose --proprio_type eef_pose --observation_type image --image_aug_num 2
```

### 3.train.ipynb
Train simple MLP models with dataset.

First, set up the configurations
```python
@PreTrainedConfig.register_subclass("omy_baseline")
@dataclass
class BaselineConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 5
    n_action_steps: int = 5

    # Architecture.
    backbone: str = 'mlp' # 'mlp' or 'transformer'
    # Vision encoder
    vision_backbone: str ="facebook/dinov3-vitb16-pretrain-lvd1689m" #"facebook/dinov2-base"
    projection_dim : int = 128
    freeze_backbone:  bool = True


    # Num hidden layers
    n_hidden_layers: int = 5
    hidden_dim: int = 512   

    ## For transformer-based architectures
    n_heads: int = 4
    dim_feedforward: int = 2048
    feedforward_activation: str = "gelu"
    dropout: float = 0.1
    pre_norm: bool = True
    n_encoder_layers: int = 6

    # Training preset
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-6

    # Learning rate scheduler parameters 
    lr_warmup_steps: int = 1000
    total_training_steps: int = 500000

# Policy Config
cfg = BaselineConfig(
    chunk_size=10,
    n_action_steps=10,
    backbone='mlp',
    optimizer_lr= 5e-4,
    n_hidden_layers=10,
    hidden_dim=512,
    # If you are using image features, uncomment the following line
    vision_backbone='facebook/dinov3-vitb16-pretrain-lvd1689m',#"facebook/dinov2-base", **You need access to use this model** Use dinov2 if you don't have access
    projection_dim=128,
    freeze_backbone=True,

)
```
Then you can train the baseline models!

You can run this with the scripts as follows
```
python train.py
  --dataset_path DATASET_PATH
  --batch_size BATCH_SIZE
  --num_epochs NUM_EPOCHS
  --ckpt_path CKPT_PATH
  --chunk_size CHUNK_SIZE
  --n_action_steps N_ACTION_STEPS
  --learning_rate LEARNING_RATE
  --backbone BACKBONE
  --n_hidden_layers N_HIDDEN_LAYERS
  --hidden_dim HIDDEN_DIM
  --vision_backbone {facebook/dinov3-vitb16-pretrain-lvd1689m,facebook/dinov2-base}
  --projection_dim PROJECTION_DIM
  --freeze_backbone FREEZE_BACKBONE
```

### 4.eval.ipynb

This file contains evaluation of the trained models.

<img src="./media/baseline.gif" width="480" height="360" controls></img>


Action Representation: Target Joint Position, State Representation: Current Joint Position

<table> Success rate
    <tr>
    <th>  MLP with GT Object Pose </th>
    <th>  MLP with Image (DINOv3 feature)</th>
    <th> SmolVLA with Image</th>
    </tr>
    <tr>
    <th>  65%</th>
    <th> 60% </th>
    <th>65%</th>
    </tr>
</table>

## Try with your own policy
Look at [src/policies/how_to.md](./src/policies/how_to.md) for instructions. 


For training change [3.train.ipynb](3.train.ipynb)
```python 
from src.policies.baseline.configuration import BaselineConfig
from src.policies.baseline.modeling import BaselinePolicy
```
to your own path in  in **first** cell and

```python
cfg = BaselineConfig(
    chunk_size=10,
    n_action_steps=10,

)
```
change configuration class in **second** cell.
```python
'''`
Instantiate Policy
'''
policy = BaselinePolicy(**kwargs)
```
Finally, change policy class in **fourth** cell.


For evaluation [4.eval.ipynb](4.eval.ipynb) , change 
```python
from src.policies.baseline.modeling import BaselinePolicy
```
to your own path in **first** cell and
```python
policy = BaselinePolicy.from_pretrained(CKPT, **kwargs)
```
change policy class in the **third** cell. 


## Others

### Data collection with leader arm
First, launch the ros2 package from ROBOTIS to turn on the leader. This requires ROS2. 
```
ros2 launch open_manipulator_bringup hardware_y_leader.launch.py
```
Then, with the other terminal run 
```
python leader.py
```

Finally, on the third terminal, run
```
python collect_data.py
```
to collect the data!