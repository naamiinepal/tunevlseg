#!/usr/bin/env python

from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass
