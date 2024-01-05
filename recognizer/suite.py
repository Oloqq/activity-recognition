from .samples import Sample
import numpy as np
from models.utils import ACTIVITIES

class Suite:
    def __init__(self, name: str, samples: list[Sample]):
        self.samples = samples
        self.name = name
        self.specs = f"length {len(samples)}"

    def __repr__(self):
        return f"{self.name}: {self.specs}"
    
    def synchronize(self):
        self.specs += f", synchronize"
        pass # TODO

    def resample(self, frequency_hz: float):
        self.specs += f", resample to {frequency_hz}Hz"
        for sample in self.samples:
            sample.resample(frequency_hz)

    def _count_classes(self):
        classes = []
        for sample in self.samples:
            classes.append(ACTIVITIES[sample.activity])

        count = len(set(classes))
        return count 
    
    # TO DO
    def _get_class_distribution(self):
        pass

    # TO DO
    # use class distribution to split
    def train_test_split(self, train_size: float):
        pass