import torch

from transformers import ViTConfig
from utils.func import print_msg, select_out_features

from .bridge import FineGrainedPromptTuning, FusionModule
from .side_vit import ViTForImageClassification as SideViT
from .frozen_vit import ViTForImageClassification as FrozenViT


def generate_model(cfg):
    model = build_model(cfg)
    if cfg.train.checkpoint:
        load_weights(model, cfg.train.checkpoint)
    model = model.to(cfg.base.device)

    if cfg.base.preload:
        frozen_encoder = None
    else:
        frozen_encoder = build_frozen_encoder(cfg).to(cfg.base.device)

        num_learnable_params = 0
        total_params = 0
        for _, param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                num_learnable_params += param.numel()
        if frozen_encoder is not None:
            for _, param in frozen_encoder.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    num_learnable_params += param.numel()

        print('Total params: {}'.format(total_params))
        print('Learnable params: {}'.format(num_learnable_params))
        print('Learnable params ratio: {:.4f}%'.format(num_learnable_params / total_params * 100))

    return frozen_encoder, model


def load_weights(model, checkpoint):
    weights = torch.load(checkpoint)
    model.load_state_dict(weights, strict=True)
    print_msg('Load weights form {}'.format(checkpoint))    


def build_model(cfg):
    out_features = select_out_features(
        cfg.data.num_classes,
        cfg.train.criterion
    )

    vit_config = ViTConfig.from_pretrained(cfg.base.pretrained_path)
    fusion_module = FusionModule(
        num_layer=vit_config.num_hidden_layers,
        in_dim=vit_config.hidden_size,
        out_dim=cfg.train.side_dimension,
        num_heads=vit_config.num_attention_heads,
        num_prompts=cfg.train.num_prompts,
        prompts_dim=cfg.train.side_dimension
    )

    side_config = ViTConfig.from_pretrained(
        cfg.base.pretrained_path,
        hidden_size=cfg.train.side_dimension,
        intermediate_size=cfg.train.side_dimension * 4,
        image_size=cfg.data.coarse_input_size,
        num_labels=out_features,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0
    )
    side_encoder = SideViT(side_config)

    model = FineGrainedPromptTuning(side_encoder, fusion_module)
    return model


def build_frozen_encoder(cfg):
    frozen_config = ViTConfig.from_pretrained(cfg.base.pretrained_path)
    frozen_config.token_imp = cfg.train.token_imp
    frozen_config.token_ratio = cfg.train.token_ratio
    if cfg.train.pretrained:
        frozen_encoder = FrozenViT.from_pretrained(
            cfg.base.pretrained_path,
            config=frozen_config
        )
    else:
        frozen_encoder = FrozenViT(frozen_config)

    frozen_encoder.eval()
    for p in frozen_encoder.parameters():
        p.requires_grad = False

    return frozen_encoder
