import argparse
from lerobot.common.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.datasets.utils import cycle
from lerobot.configs.types import FeatureType
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
import torch
from src.policies.baseline.configuration import BaselineConfig
from src.policies.baseline.modeling import BaselinePolicy

def main(args):
    dataset_cfg = DatasetConfig("transformed_data")
    dataset_cfg.root = args.dataset_path
    pipeline_cfg = TrainPipelineConfig(dataset_cfg)


    cfg = BaselineConfig(
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        optimizer_lr= args.learning_rate,
        backbone=args.backbone,
        n_hidden_layers=args.n_hidden_layers,
        hidden_dim=args.hidden_dim,
        # If you are using image features, uncomment the following line
        vision_backbone=args.vision_backbone,#"facebook/dinov2-base", **You need access to use this model** Use dinov2 if you don't have access
        projection_dim=args.projection_dim,
        freeze_backbone=args.freeze_backbone,
    )
    kwargs = {}
    pipeline_cfg.policy = cfg
    pipeline_cfg.optimizer = cfg.get_optimizer_preset()
    pipeline_cfg.scheduler = cfg.get_scheduler_preset()
    dataset = make_dataset(pipeline_cfg)
    ds_meta = dataset.meta
    features = dataset_to_policy_features(ds_meta.features)
    kwargs["dataset_stats"] = ds_meta.stats
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    policy = BaselinePolicy(**kwargs)
    policy.to(pipeline_cfg.policy.device)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(pipeline_cfg, policy)


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size= args.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=4,
    )

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    total_params = list(policy.parameters())
    print(f"Total number of parameters: {format_big_number(sum(p.numel() for p in total_params))}")
    print(f"Number of trainable parameters: {format_big_number(sum(p.numel() for p in trainable_params))}")

    device = get_safe_torch_device(pipeline_cfg.policy.device, log=True)
    step = 0
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}/{args.num_epochs}")
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            policy.train()
            loss, output_dict = policy.forward(batch)
            # grad_norm = torch.nn.utils.clip_grad_norm_(
            #     policy.parameters(),
            #     pipeline_cfg.optimizer.grad_clip_norm,
            #     error_if_nonfinite=False,
            # )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step through pytorch scheduler at every batch instead of epoch
            if lr_scheduler is not None:
                lr_scheduler.step()
            step += 1
            if step % 100 == 0:
                print(f"Step: {step}, Loss: {loss.item():.4f}, learning rate: {optimizer.param_groups[0]['lr']:.6f}")


    policy.save_pretrained(args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/transformed_data2',)
    parser.add_argument('--batch_size', type=int, default=64,)
    parser.add_argument('--num_epochs', type=int, default=100,)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/baseline_model_image',)
    parser.add_argument('--chunk_size', type=int, default=10,)
    parser.add_argument('--n_action_steps', type=int, default=10,)  
    parser.add_argument('--learning_rate', type=float, default=5e-4,)   
    parser.add_argument('--backbone', type=str, default='mlp',)
    parser.add_argument('--n_hidden_layers', type=int, default=10,)
    parser.add_argument('--hidden_dim', type=int, default=512,)
    parser.add_argument('--vision_backbone', type=str, default='facebook/dinov3-vitb16-pretrain-lvd1689m',choices=["facebook/dinov3-vitb16-pretrain-lvd1689m","facebook/dinov2-base"],)
    parser.add_argument('--projection_dim', type=int, default=0,)
    parser.add_argument('--freeze_backbone', type=bool, default=True,)
    args = parser.parse_args()
    main(args)