# optimizer
optimizer = {"type": "SGD", "lr": 0.01, "weight_decay": 0.0005}
optimizer_config = {}
# learning policy
lr_config = {"policy": "poly", "power": 0.9, "min_lr": 1e-6, "by_epoch": False}
# runtime settings
runner = {"type": "IterBasedRunner", "max_iters": 80000}
checkpoint_config = {"by_epoch": False, "interval": 8000}
evaluation = {"interval": 8000, "metric": "mIoU"}
