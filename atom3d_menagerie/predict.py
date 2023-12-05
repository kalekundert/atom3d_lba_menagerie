import lightning.pytorch as pl

from torch.nn import MSELoss
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import (
        MeanAbsoluteError, PearsonCorrCoef, SpearmanCorrCoef,
)
from dataclasses import dataclass, fields

class RegressionModule(pl.LightningModule):

    def __init__(self, model, opt_factory):
        super().__init__()
        self.model = model
        self.loss = MSELoss()
        self.mae = MeanAbsoluteError()
        self.pearson_r = PearsonCorrCoef()
        self.spearman_r = SpearmanCorrCoef()
        self.optimizer = opt_factory(model.parameters())

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)

        return Forward(
                loss=self.loss(y_hat, y),
                mae=self.mae(y_hat, y),
                pearson_r=self.pearson_r(y_hat.flatten(), y.flatten()),
                spearman_r=self.spearman_r(y_hat.flatten(), y.flatten()),
        )

    def training_step(self, batch, _):
        fwd = self.forward(batch)
        self._log_forward('train', fwd)
        return fwd.loss


    def validation_step(self, batch, _):
        fwd = self.forward(batch)
        self._log_forward('val', fwd)
        return fwd.loss

    def test_step(self, batch, _):
        fwd = self.forward(batch)
        self._log_forward('test', fwd)
        return fwd.loss

    def _log_forward(self, step, fwd):
        for field in fields(fwd):
            metric = field.name
            value = getattr(fwd, metric)
            self.log(f'{step}/{metric}', value, on_epoch=True)

@dataclass
class Forward:
    loss: float
    mae: float
    pearson_r: float
    spearman_r: float


def get_trainer(out_dir, **kwargs):
    return pl.Trainer(
            logger=TensorBoardLogger(
                save_dir=out_dir.parent,
                name=out_dir.name,
                default_hp_metric=False,
            ),
            **kwargs,
    )

