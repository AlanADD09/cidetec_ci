from sklearn.neighbors import KNeighborsClassifier
from .base import ClassifierStrategy

class KNNStrategy(ClassifierStrategy):
    def build(self):
        return KNeighborsClassifier(**self.params)
