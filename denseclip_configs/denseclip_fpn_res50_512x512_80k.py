_base_ = [
    "_base_/models/denseclip_r50.py",
    "_base_/datasets/ade20k_clip.py",
    "_base_/default_runtime.py",
    "_base_/schedules/schedule_80k.py",
]

model = {
    "type": "DenseCLIP",
    "pretrained": "pretrained/RN50.pt",
    "context_length": 5,
    "text_head": False,
    "backbone": {
        "type": "CLIPResNetWithAttention",
        "layers": [3, 4, 6, 3],
        "output_dim": 1024,
        "input_resolution": 512,
        "style": "pytorch",
    },
    "text_encoder": {
        "type": "CLIPTextContextEncoder",
        "context_length": 13,
        "embed_dim": 1024,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
        "style": "pytorch",
    },
    "context_decoder": {
        "type": "ContextDecoder",
        "transformer_width": 256,
        "transformer_heads": 4,
        "transformer_layers": 3,
        "visual_dim": 1024,
        "dropout": 0.1,
        "outdim": 1024,
        "style": "pytorch",
    },
    "neck": {
        "type": "FPN",
        "in_channels": [256, 512, 1024, 2048 + 150],
        "out_channels": 256,
        "num_outs": 4,
    },
    "decode_head": {
        "type": "FPNHead",
        "num_classes": 150,
        "loss_decode": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 1.0,
        },
    },
}

lr_config = {
    "policy": "poly",
    "power": 0.9,
    "min_lr": 1e-6,
    "by_epoch": False,
    "warmup": "linear",
    "warmup_iters": 1500,
    "warmup_ratio": 1e-6,
}


optimizer = {
    "type": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "paramwise_cfg": {
        "custom_keys": {
            "backbone": {"lr_mult": 0.1},
            "text_encoder": {"lr_mult": 0.0},
            "norm": {"decay_mult": 0.0},
        }
    },
}

data = {"samples_per_gpu": 4}
