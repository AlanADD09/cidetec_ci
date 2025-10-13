from sklearn.neighbors import NearestCentroid
from .base import ClassifierStrategy

class NearestCentroidStrategy(ClassifierStrategy):
    def build(self):
        return NearestCentroid(**self.params)
