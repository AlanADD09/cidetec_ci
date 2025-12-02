from sklearn.svm import SVC
from .base import ClassifierStrategy


class SVMStrategy(ClassifierStrategy):
    def build(self):
        return SVC(**self.params)
