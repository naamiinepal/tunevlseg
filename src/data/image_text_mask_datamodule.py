from __future__ import annotations

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ImageTextDatamodule(LightningDataModule):
    def __init__(
        self,
        train_ds: Dataset | None = None,
        val_ds: Dataset | None = None,
        test_ds: Dataset | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if train_ds is None and val_ds is None and test_ds is None:
            msg = "Either train, validation, or test dataset should be divided."
            raise ValueError(msg)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        self.batch_size_per_device: int = self.hparams.batch_size  # type: ignore

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:  # type:ignore
                msg = f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."  # type:ignore
                raise ValueError(
                    msg,
                )
            # Divide batch size by the number of devices.
            self.batch_size_per_device //= self.trainer.world_size  # type:ignore

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.hparams.train_ds,  # type:ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,  # type:ignore
            pin_memory=self.hparams.pin_memory,  # type:ignore
            drop_last=self.hparams.drop_last,  # type:ignore
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.hparams.val_ds,  # type:ignore
            batch_size=self.batch_size_per_device,  # type:ignore
            num_workers=self.hparams.num_workers,  # type:ignore
            pin_memory=self.hparams.pin_memory,  # type:ignore
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.hparams.test_ds,  # type:ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,  # type:ignore
            pin_memory=self.hparams.pin_memory,  # type:ignore
            shuffle=False,
        )
