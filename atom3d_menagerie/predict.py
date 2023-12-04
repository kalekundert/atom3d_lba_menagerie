import lightning.pytorch as pl

from torch.nn import MSELoss
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef
from dataclasses import dataclass

class RegressionModule(pl.LightningModule):

    def __init__(self, model, opt_factory):
        super().__init__()
        self.model = model
        self.loss = MSELoss()
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
                pearson_r=self.pearson_r(y_hat.flatten(), y.flatten()),
                spearman_r=self.spearman_r(y_hat.flatten(), y.flatten()),
        )

    def training_step(self, batch, _):
        fwd = self.forward(batch)
        self.log('train/loss', fwd.loss, on_epoch=True)
        self.log('train/pearson_r', fwd.pearson_r, on_epoch=True)
        self.log('train/spearman_r', fwd.spearman_r, on_epoch=True)
        return fwd.loss

    def validation_step(self, batch, _):
        fwd = self.forward(batch)
        self.log('val/loss', fwd.loss, on_epoch=True)
        self.log('val/pearson_r', fwd.pearson_r, on_epoch=True)
        self.log('val/spearman_r', fwd.spearman_r, on_epoch=True)
        return fwd.loss

    def test_step(self, batch, _):
        fwd = self.forward(batch)
        self.log('test/loss', fwd.loss, on_epoch=True)
        self.log('test/pearson_r', fwd.pearson_r, on_epoch=True)
        self.log('test/spearman_r', fwd.spearman_r, on_epoch=True)
        return fwd.loss

@dataclass
class Forward:
    loss: float
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
