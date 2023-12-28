# pyright: reportIncompatibleMethodOverride=false
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Union

import torch
import wandb
from lightning import LightningModule
from torch import nn, optim
from torchmetrics import Dice, JaccardIndex
from transformers import AutoTokenizer, PreTrainedTokenizerBase

BatchType = Mapping[str, Any]


class ImageTextModule(LightningModule):
    """Example of a `LightningModule` for Segmentation using Image and Text."""

    plot_columns = ["Image", "Caption", "Label"]

    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        optimizer: type[optim.optimizer.Optimizer],
        scheduler: type[optim.lr_scheduler.LRScheduler],
        tokenizer_name_or_path: Union[str, Path],
        compile: bool,
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        weight_decay: float = 0,
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name_or_path
        )

        # Dice Loggers
        self.train_dice = Dice(threshold=threshold)
        self.val_dice = Dice(threshold=threshold)
        self.test_dice = Dice(threshold=threshold)

        # IoU Loggers
        self.train_iou = JaccardIndex(task=task, threshold=threshold)
        self.val_iou = JaccardIndex(task=task, threshold=threshold)
        self.test_iou = JaccardIndex(task=task, threshold=threshold)

        # This is true until image, caption, and mask are logged for the first time
        self.log_image_caption_mask = True

    def forward(self, *args, **kwargs):
        """Perform a forward pass through the model `self.net`."""
        return self.net(*args, **kwargs)

    def model_step(
        self, batch: BatchType
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # batch = {
        #     "text": text,
        #     "img": img,
        #     "mask": mask,
        #     "img_name": img_name,
        #     "mask_name": mask_name,
        #     "img_shape": img.shape,
        #     "mask_shape": mask.shape,
        # }
        text = batch["text"]
        img = batch["img"]

        logits = self.forward(image_input=img, text_input=text)

        mask = batch["mask"]
        loss = self.loss_fn(logits, mask)

        preds = torch.sigmoid(logits)

        return loss, preds, mask

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
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
        )

        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        # log epoch metric
        self.log_dict(
            {"train_dice_epoch": self.train_dice, "train_iou_epoch": self.train_iou}
        )

    def validation_step(self, batch: BatchType, batch_idx: int) -> None:
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
        )

        # Log images only for the first batch
        if self.logger is not None and batch_idx == 0:
            plot_preds = list(map(wandb.Image, preds[: self.hparams.log_image_num]))

            self.logger.log_image("val_pred", plot_preds)

            if self.log_image_caption_mask:
                img = batch["img"]

                selected_images = img[: self.hparams.log_image_num]

                plot_images = map(wandb.Image, self.normalize_img(selected_images))

                text = batch["text"]

                plot_input_ids = self.decode_input_ids(
                    text.input_ids[: self.hparams.log_image_num]
                )

                plot_label = map(wandb.Image, targets[: self.hparams.log_image_num])

                data = list(zip(plot_images, plot_input_ids, plot_label))

                self.logger.log_table(
                    "val_caption_label", columns=self.plot_columns, data=data
                )

                # Stop logging now
                self.log_image_caption_mask = False

    def decode_input_ids(
        self,
        input_ids: torch.Tensor,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        tokenizer = tokenizer or self.tokenizer
        return tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    @staticmethod
    def normalize_img(img: torch.Tensor):
        kwargs = dict(dim=(2, 3), keepdim=True)

        img_min = img.amin(**kwargs)
        img_max = img.amax(**kwargs)

        return (img - img_min) / (img_max - img_min)

    def test_step(self, batch: BatchType) -> None:
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
        )

    def setup(self, stage: Optional[str]) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and (stage is None or stage == "fit"):
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
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
        if self.hparams.weight_decay > 0:
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
                msg = (
                    f"parameters {inter_params} made it into both decay/no_decay sets!"
                )
                raise ValueError(msg)

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}

            extra_params = param_dict.keys() - (decay | no_decay)
            if len(extra_params) == 0:
                msg = f"parameters {extra_params} were not separated into either decay/no_decay set!"
                raise ValueError(msg)

            # create the pytorch optimizer object
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in decay],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [param_dict[pn] for pn in no_decay],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optim_groups = self.parameters()
        optimizer = self.optimizer(optim_groups)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
