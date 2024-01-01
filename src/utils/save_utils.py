from logging import LoggerAdapter
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

import torchvision.utils
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torchvision.transforms import functional as TF

if TYPE_CHECKING:
    import torch


def save_predictions(
    cfg: DictConfig,
    log: LoggerAdapter,
    trainer: Trainer,
    model: Optional[LightningModule],
    dataloaders: Optional[LightningDataModule],
    ckpt_path: Optional[str],
) -> None:
    # Writes output masks to files
    output_masks_dir = cfg.get("output_masks_dir")

    if output_masks_dir is None:
        output_masks_dir = "outputs"
        log.warning(f"Output directory not found. Reverting to {output_masks_dir}")

    output_masks_dir = Path(output_masks_dir)

    log.info("Generating prediction masks of test dataset")
    pred_outputs: Iterable[dict] = trainer.predict(
        model=model, dataloaders=dataloaders, ckpt_path=ckpt_path,
    )  # type: ignore

    log.info(f"Saving the generated masks in directory {output_masks_dir}")

    # The interpolation to use to save the images
    output_interpolation = TF.InterpolationMode(
        cfg.get("output_interpolation", "bicubic"),
    )

    for p in pred_outputs:
        preds: Iterable[torch.Tensor] = p["preds"]
        mask_names: Iterable[str] = p["mask_name"]
        mask_shapes: Iterable[List[int]] = p["mask_shape"]
        for pred, mask_name, mask_shape in zip(preds, mask_names, mask_shapes):
            file_path: Path = output_masks_dir / mask_name

            # `mask_name` may contain directories, so making sure they exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            torchvision.utils.save_image(
                TF.resize(
                    pred.float(),
                    size=mask_shape,
                    interpolation=output_interpolation,
                ),
                file_path,
            )

    total_images_saved = sum(len(p["preds"]) for p in pred_outputs)

    log.info(
        f"Logged {total_images_saved} masks to {output_masks_dir} using {output_interpolation} interpolation.",
    )