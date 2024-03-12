from collections.abc import Iterable
from typing import Any

from torch.utils.data import default_collate
from transformers import DataCollatorWithPadding


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __init__(self, padding_keys: Iterable[str], *args, **kwargs):
        self.padding_keys = set(padding_keys)
        if not self.padding_keys:
            msg = "`padding_keys` should not be empty."
            raise ValueError(msg)

        super().__init__(*args, **kwargs)

    def __call__(self, features: list[dict[str, Any]]):
        features_to_pad = [
            {key: value for key, value in example.items() if key in self.padding_keys}
            for example in features
        ]

        # Pad the selected items
        padded_features = super().__call__(features_to_pad)

        # Use the default pytorch collate function for the leftover items
        leftover_features = [
            {key: value for key, value in example.items() if key not in padded_features}
            for example in features
        ]

        collate_features = default_collate(leftover_features)

        return {**collate_features, **padded_features}
