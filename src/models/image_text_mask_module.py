# pyright: reportIncompatibleMethodOverride=false
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import wandb
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics import Dice, JaccardIndex
from torchvision.transforms import functional as TF
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from pytorch_lightning.loggers import WandbLogger

    MappingStr2Any = Mapping[str, Any]
    DictStr2Any = dict[str, Any]


class ImageTextMaskModule(LightningModule):
    """Example of a `LightningModule` for Segmentation using Image and Text."""

    plot_columns = ["Image", "Caption", "Label"]

    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: type[optim.optimizer.Optimizer],
        scheduler: type[optim.lr_scheduler.LRScheduler] | None,
        tokenizer_name_or_path: str | Path,
        compile: bool,
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        weight_decay: float = 0.0,
        log_image_num: int = 8,
        lr_scheduler_config: MappingStr2Any | None = None,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] | None = torch.sigmoid,
        cache_outputs: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `ImageTextModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net", "loss_fn", "optimizer", "scheduler"])

        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name_or_path,
        )

        # Dice Loggers
        dice_kwargs = {"threshold": threshold, "average": "samples"}
        self.train_dice = Dice(**dice_kwargs)
        self.val_dice = Dice(**dice_kwargs)
        self.test_dice = Dice(**dice_kwargs)

        # IoU Loggers
        iou_kwargs = {"task": task, "threshold": threshold}
        self.train_iou = JaccardIndex(**iou_kwargs)
        self.val_iou = JaccardIndex(**iou_kwargs)
        self.test_iou = JaccardIndex(**iou_kwargs)

        self.activation_fn = nn.Identity() if activation_fn is None else activation_fn

        self.logger: WandbLogger | None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`."""
        return self.net(*args, **kwargs)

    def model_step(
        self,
        batch: MappingStr2Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        logits = self.get_logits(batch)

        mask = batch["mask"]
        loss = self.loss_fn(logits, mask)

        preds = self.activation_fn(logits)

        return loss, preds, mask.long()

    def training_step(self, batch: MappingStr2Any, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        self.train_dice(preds, targets)
        self.train_iou(preds, targets)

        self.log_dict(
            {
                "train_dice_step": self.train_dice,
                "train_iou_step": self.train_iou,
            },
            prog_bar=True,
            batch_size=len(preds),
        )

        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(preds),
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        # log epoch metric
        self.log_dict(
            {"train_dice_epoch": self.train_dice, "train_iou_epoch": self.train_iou},
        )

    def validation_step(self, batch: MappingStr2Any, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_dice(preds, targets)
        self.val_iou(preds, targets)

        self.log_dict(
            {"val_dice": self.val_dice, "val_iou": self.val_iou, "val_loss": loss},
            prog_bar=True,
            batch_size=len(preds),
        )

        # Only log images on the first validation step of first epoch
        if self.logger is not None and self.global_step == 0:
            img = batch["image"]

            selected_images = img[: self.hparams.log_image_num]  # type: ignore

            plot_images = self.get_plot_images(self.normalize_img(selected_images))

            input_ids = batch["input_ids"]

            plot_input_ids = self.decode_input_ids(
                input_ids[: self.hparams.log_image_num],  # type: ignore
            )

            plot_label = self.get_plot_images(targets)

            data = [
                [img, inp_id, label]
                for img, inp_id, label in zip(
                    plot_images,
                    plot_input_ids,
                    plot_label,
                    strict=True,
                )
            ]

            self.logger.log_table(
                "val_caption_label",
                columns=self.plot_columns,
                data=data,
            )

        # Log images only for the first batch of each epoch
        if self.logger is not None and batch_idx == 0:
            plot_preds = self.get_plot_images(preds)

            self.logger.log_image("val_pred", list(plot_preds))

    def get_plot_images(self, images: torch.Tensor) -> map[wandb.Image]:
        return map(
            wandb.Image,
            map(TF.to_pil_image, images[: self.hparams.log_image_num].float()),  # type: ignore
        )

    def decode_input_ids(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> list[str]:
        tokenizer = tokenizer or self.tokenizer
        return tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    @staticmethod
    def normalize_img(img: torch.Tensor) -> torch.Tensor:
        kwargs = {"dim": (2, 3), "keepdim": True}

        img_min = img.amin(**kwargs)
        img_max = img.amax(**kwargs)

        return (img - img_min) / (img_max - img_min)

    def test_step(self, batch: MappingStr2Any) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_dice(preds, targets)
        self.test_iou(preds, targets)

        self.log_dict(
            {"test_dice": self.test_dice, "test_iou": self.test_iou, "test_loss": loss},
            prog_bar=True,
            batch_size=len(preds),
        )

    def predict_step(self, batch: MappingStr2Any) -> DictStr2Any:
        logits = self.get_logits(batch)

        preds = self.activation_fn(logits)

        # Mask name and shape is needed to save the predictions
        # in their original size
        return {
            "preds": preds,
            "mask_name": batch["mask_name"],
            "mask_shape": batch["mask_shape"],
        }

    def get_logits(self, batch: MappingStr2Any) -> torch.Tensor:
        text_input = {k: batch[k] for k in ("input_ids", "attention_mask")}

        if getattr(self.hparams, "cache_outputs", False):
            text_input["cache_name"] = batch["cache_name"]

        img = batch["image"]

        return self(image_input=img, text_input=text_input)

    def setup(self, stage: str | None) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and (stage is None or stage == "fit"):  # type: ignore
            self.net = torch.compile(self.net)

    def get_optim_groups(self):
        if self.hparams.weight_decay <= 0:  # type: ignore
            return self.parameters()

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv1d)
        blacklist_weight_modules = (
            nn.Embedding,
            nn.LayerNorm,
            nn.BatchNorm2d,
        )
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                if pn.endswith("proj_weight"):
                    # Add project weights to decay set
                    decay.add(fpn)
                elif pn.endswith("weight"):
                    # random note: because named_modules and named_parameters are recursive
                    # we will see the same tensors p many many times. but doing it this way
                    # allows us to know which parent module any tensor p belongs to...
                    if isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                else:
                    # all paramters except weight will not be decayed
                    no_decay.add(fpn)

        inter_params = decay & no_decay
        if len(inter_params) == 0:
            msg = f"parameters {inter_params} made it into both decay/no_decay sets!"
            raise ValueError(msg)

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())

        extra_params = param_dict.keys() - (decay | no_decay)
        if len(extra_params) == 0:
            msg = f"parameters {extra_params} were not separated into either decay/no_decay set!"
            raise ValueError(msg)

        # create the pytorch optimizer parameters
        return [
            {
                "params": [param_dict[pn] for pn in decay],
                "weight_decay": self.hparams.weight_decay,  # type: ignore
            },
            {
                "params": [param_dict[pn] for pn in no_decay],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self) -> DictStr2Any:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        Examples
        --------
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.

        """
        optim_groups = self.get_optim_groups()
        optimizer = self.optimizer(optim_groups)  # type: ignore

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    **(self.hparams.lr_scheduler_config or {}),  # type: ignore
                },
            }

        return {"optimizer": optimizer}
