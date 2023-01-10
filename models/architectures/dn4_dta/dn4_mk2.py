import pandas as pd

from dataset.datasets_csv import CSVLoader
from models.architectures.classifier import ClassifierTemplate


class DN4_DTA(ClassifierTemplate):

    def __init__(self, model_cfg, num_class, dataset: CSVLoader):
        super().__init__(model_cfg, num_class, dataset)
        self.build()
        self.data.X = pd.DataFrame(None, columns=self.cfg.CLASS_NAMES)
        self.data.y = pd.DataFrame(None)

    def forward(self):
        self.backbone_2d(self.data)
        self.knn(self.data)
        self.data.X.append()

    def post_fit_learners(self):
        """
        Fit lazy learning head(s) after training process
        :return:
        :rtype:
        """
        self.dt.fit(self.data)

