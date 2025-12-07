
from email.mime import image
import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from .configuration import BaselineConfig
from transformers import AutoImageProcessor, AutoModel
from lerobot.common.policies.normalize import Normalize, Unnormalize

class BaselinePolicy(PreTrainedPolicy):
    """
    Baseline policy using either MLP or Transformer backbone to map from observations to action chunks.
    The model can process image inputs through a visual encoder and concatenate them with robot state
    and environment object state inputs. The model outputs action chunks which can be used to interact
    with the environment.
    """

    config_class = BaselineConfig
    name = "baseline"

    def __init__(
        self,
        config: BaselineConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,

    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        if config.backbone == 'mlp':
            self.model = BaselineModel(config)
        elif config.backbone == 'transformer':
            self.model = TransformerModel(config)
        else:
            raise ValueError(f"Unknown backbone type: {config.backbone}")
        self.reset()

    def get_optim_params(self) -> dict:
        # return parameters for optimizer
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
        if self.config.image_features:
            batch = dict(batch)  
        if len(self._action_queue) == 0:
            actions = self.model(batch)[0][: self.config.n_action_steps, :]
            actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions)
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self.normalize_inputs(batch)
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
        
        batch = self.normalize_targets(batch)
        actions = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss
        return loss, loss_dict
    
class TransformerModel(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()

        self.config = config
        input_dim = 0
        if self.config.image_features:
            self.visual_encoder = VisualEncoder(config)
            input_dim += config.projection_dim
        if self.config.robot_state_feature:
            input_dim += self.config.robot_state_feature.shape[0]
        if self.config.env_state_feature:
            input_dim += self.config.env_state_feature.shape[0]
        
        # input_dim should be dividable by n_heads
        self.upsample_layer = nn.Linear(input_dim, self.config.n_heads * 64)

        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=self.config.n_heads * 64,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            activation=self.config.feedforward_activation,
            dropout=self.config.dropout,
            batch_first=True,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_layer,
            num_layers=self.config.n_encoder_layers,
        )
        self.output_layer = nn.Linear(self.config.n_heads * 64, 
                                      self.config.chunk_size * self.config.action_feature.shape[0])
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        '''
        Input: batch dict with keys OBS_IMAGES, OBS_STATE, OBS_ENV_STATE
        Output: actions [B, chunk_size, action_dim]
        '''
        features = []
        if self.config.image_features:
            images = batch[OBS_IMAGE]   # list of [B, C, H, W]
            visual_features = self.visual_encoder(images)  # [B, projection_dim]
            features.append(visual_features)
        if self.config.robot_state_feature:
            state = batch[OBS_STATE]   # [B, state_dim]
            features.append(state)
        if self.config.env_state_feature:
            env_state = batch[OBS_ENV_STATE]   # [B, env_state_dim]
            features.append(env_state)

        x = torch.cat(features, dim=-1)   # [B, total_input_dim]
        x = self.upsample_layer(x)   # [B, n_heads * 64]
        x = x.unsqueeze(1)   # [B, 1, n_heads * 64] , add sequence dimension
        x = self.transformer_decoder(x, x)    # [B, 1, n_heads * 64]
        x = x.squeeze(1)    # [B, n_heads * 64]
        x = self.output_layer(x)    # [B, chunk_size * action_dim]
        B = x.shape[0]
        action_dim = self.config.action_feature.shape[0]
        x = x.view(B, self.config.chunk_size, action_dim)   # [B, chunk_size, action_dim]
        return x

class BaselineModel(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        """
              Action Chunks
            |-----------------|
            |Output Projection|        
            |-----------------|
                    |        
        |--------------------------|
        | MLP or Transformer layers|
        |--------------------------|
            |  Input Features   |      
        |----------|     |      |
        |Projection|     |      |
        |----------|     |      |
            | (Optional) |      | (Optional) 
        |--------|     State  Env_Object_State
        | Visual |    
        |Encoder |        
        |--------|    
            |
          Images      
        """
        self.config = config
        input_dim = 0
        if self.config.image_features:
            self.visual_encoder = VisualEncoder(config)
            input_dim += self.visual_encoder.output_dim
        if self.config.robot_state_feature:
            input_dim += self.config.robot_state_feature.shape[0]
        if self.config.env_state_feature:
            input_dim += self.config.env_state_feature.shape[0]

        
        # simple MLP to process
        hidden_layers = []
        last_dim = input_dim
        for _ in range(self.config.n_hidden_layers):
            hidden_layers.append(nn.Linear(last_dim, self.config.hidden_dim))
            hidden_layers.append(nn.ReLU())
            last_dim = self.config.hidden_dim
        self.mlp = nn.Sequential(
            *hidden_layers,
            nn.Linear(self.config.hidden_dim, self.config.chunk_size * self.config.action_feature.shape[0]),
        )
        # initialize mlp weights
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    # zero initialization for biases
                    nn.init.zeros_(m.bias)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        '''
        Input: batch dict with keys OBS_IMAGES, OBS_STATE, OBS_ENV_STATE
        Output: actions [B, chunk_size, action_dim]
        '''
        features = []
        if self.config.image_features:
            images = batch[OBS_IMAGE]   # list of [B, C, H, W]
            visual_features = self.visual_encoder(images)  # [B, projection_dim]
            features.append(visual_features)
        if self.config.robot_state_feature:
            state = batch[OBS_STATE]   # [B, state_dim]
            features.append(state)
        if self.config.env_state_feature:
            env_state = batch[OBS_ENV_STATE]   # [B, env_state_dim]
            features.append(env_state)

        x = torch.cat(features, dim=-1)   # [B, total_input_dim]
        x = self.mlp(x)    # [B, chunk_size * action_dim]
        B = x.shape[0]
        action_dim = self.config.action_feature.shape[0]
        x = x.view(B, self.config.chunk_size, action_dim)   # [B, chunk_size, action_dim]
        return x

    


class VisualEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg.vision_backbone
        pretrained_model_name = cfg.vision_backbone
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto", 
        )
        self.num_cams = len(cfg.image_features)
        if cfg.projection_dim > 0:
            self.feature_dim = self.model.config.hidden_size
            self.projection = nn.Linear(self.feature_dim, cfg.projection_dim)
            self.output_dim = cfg.projection_dim
        else:
            self.projection = nn.Identity()
            print(f"Using concatenated features without projection: {self.model.config.hidden_size * self.num_cams}")
            self.output_dim = self.model.config.hidden_size
        if cfg.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, image: list[Tensor]) -> Tensor:
        '''
        Input: [B, C, H, W] 
        Output: [B, D]
        '''
        inputs = self.processor(images=image, return_tensors="pt",do_rescale=False).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        pooled_output = outputs.pooler_output # [B, D] <- [B,D] stack num_cams times
        projected_features = self.projection(pooled_output)           # [B, projection_dim]
        return projected_features