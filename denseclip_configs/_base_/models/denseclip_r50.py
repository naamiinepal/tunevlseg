# model settings
norm_cfg = {"type": "SyncBN", "requires_grad": True}

model = {
    "type": "DenseCLIP",
    "pretrained": "pretrained/RN50.pt",
    "context_length": 5,
    "backbone": {
        "type": "CLIPResNetWithAttention",
        "layers": [3, 4, 6, 3],
        "style": "pytorch",
    },
    "text_encoder": {
        "type": "CLIPTextContextEncoder",
        "context_length": 21,
        "style": "pytorch",
    },
    "context_decoder": {
        "type": "ContextDecoder",
        "context_length": 16,
        "transformer_width": 256,
        "transformer_heads": 4,
        "transformer_layers": 6,
        "visual_dim": 1024,
        "dropout": 0.1,
        "outdim": 512,
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
        "in_channels": [256, 256, 256, 256],
        "in_index": [0, 1, 2, 3],
        "feature_strides": [4, 8, 16, 32],
        "channels": 256,
        "dropout_ratio": 0.1,
        "num_classes": 19,
        "norm_cfg": norm_cfg,
        "align_corners": False,
        "loss_decode": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 1.0,
        },
    },
    "identity_head": {
        "type": "IdentityHead",
        "in_channels": 1,
        "channels": 1,
        "num_classes": 1,
        "dropout_ratio": 0.1,
        "align_corners": False,
        "loss_decode": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 0.4,
        },
    },
    # model training and testing settings
    "train_cfg": {},
    "test_cfg": {"mode": "slide", "crop_size": (512, 512), "stride": (341, 341)},
}
