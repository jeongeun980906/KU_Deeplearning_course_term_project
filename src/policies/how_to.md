# Try your own policy!
## File Structure
```
src
  |- policies
    |- baseline
       |- configuration.py
       |- modeling.py
    |[Your policy]
       |- configuration.py
       |- modeling.py
```

## Configuration File
Configuration file contains


```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import AdamConfig, AdamWConfig, SGDConfig
from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, VQBeTSchedulerConfig, DiffuserSchedulerConfig

@PreTrainedConfig.register_subclass("[YOUR POLICY]")
@dataclass
class YourConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 5
    n_action_steps: int = 5

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ENVIRONMENT_STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    # And any hyperparameters that you need for architecture

    # Then, add the hyperparemeters for optimizer, and schedular
    # Example:
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-6
    def get_optimizer_preset(self):
        return OPTIMIZERCONFIG(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )
    def get_scheduler_preset(self):
        return SCHEDULARCONFIG()
```


Lerobot supports Adam, AdamW, SGD optimizer. If you want to make your own optimizer or schedular config, make new python file and register like this: 

```python
from lerobot.common.optim.optimizers import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
'''
Register Schedular
'''
@LRSchedulerConfig.register_subclass("[YOUR POLICY]")
@dataclass
class MyOwnSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_training_steps: int


'''
Register Optimizer
'''
@OptimizerConfig.register_subclass("[YOUR POLICY]")
@dataclass
class MyOwnConfig(OptimizerConfig):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def build(self, params: dict) -> torch.optim.Optimizer:
        kwargs = asdict(self)
        # Any torch optimizer you want
        return torch.optim.YOUROPTIMIZER(params, **kwargs)
```

## 2. Modeling 
This file contains model of the policy. 

```python
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from .configuration import YourConfig
'''
Make your Policy
'''
class YourPolicy(PreTrainedPolicy):
    config_class = YourConfig
    name = "[YOUR POLICY]"

    def __init__(
        self,
        config: YourConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    )
        super().__init__(config)
        config.validate_features()
        self.config = config
        # Normalization of input and outputs
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        # Your code Starts Here
        self.model = YOURPOLICYMODEL()
        # Ends here

    def get_optim_params(self) -> dict:
        # return parameters for optimizer
        # You can change if needed
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.requires_grad
                ]
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

     @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.
        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        batch = self.normalize_inputs(batch)
        if len(self._action_queue) == 0:
            # Inference
            actions = self.model(batch)[0][: self.config.n_action_steps, :]
            actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
            self._action_queue.extend(actions)
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
            batch = self.normalize_inputs(batch)
            """Run the batch through the model and compute the loss for training or validation."""
            batch = self.normalize_targets(batch)
            # Your Code Starts here
            actions = self.model(batch)
            loss = [YOUR LOSS FUNCTION]
            loss_dict = {"anything": loss}
            # Ends here
            return loss, loss_dict


class YOURPOLICYMODEL(nn.Module):
    def __init__(self, config: YourConfig):
        super().__init__()
        '''
        Build Model than returns action
        '''
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        '''
        Input: batch dict with keys OBS_IMAGES, OBS_STATE, OBS_ENV_STATE
        Output: actions [B, chunk_size, action_dim]
        '''
```