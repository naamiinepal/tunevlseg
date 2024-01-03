# pyright: reportGeneralTypeIssues=false
from typing import Any, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ImageTextDatamodule(LightningDataModule):
    def __init__(
        self,
        train_ds: Optional[Dataset] = None,
        val_ds: Optional[Dataset] = None,
        test_ds: Optional[Dataset] = None,
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

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        self.batch_size_per_device = self.hparams.batch_size

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:  # pyright: ignore [reportOptionalOperand]
                msg = f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                raise ValueError(
                    msg,
                )
            # Divide batch size by the number of devices.
            self.batch_size_per_device //= self.trainer.world_size  # pyright: ignore [reportOptionalOperand]

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.hparams.train_ds,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.hparams.val_ds,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.hparams.test_ds,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
