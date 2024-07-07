import lightning.pytorch as pl
import torch

from torch.nn import Module, MSELoss, BCELoss
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.regression import (
        MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
)
from torcheval.metrics import (
        BinaryAccuracy, BinaryPrecision, BinaryRecall,
        BinaryAUROC, BinaryAUPRC,
)
from dataclasses import dataclass, fields

class RegressionModule(pl.LightningModule):

    @dataclass
    class Forward:
        loss: float
        mae: float
        rmse: float
        pearson_r: float

    def __init__(self, model, opt_factory):
        super().__init__()
        self.model = model
        self.loss = MSELoss()
        self.mae = MeanAbsoluteError()
        self.rmse = MeanSquaredError(squared=False)
        self.pearson_r = PearsonCorrCoef()
        self.optimizer = opt_factory(model.parameters())

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)

        return self.Forward(
                loss=self.loss(y_hat, y),
                mae=self.mae(y_hat, y),
                rmse=self.rmse(y_hat, y),
                pearson_r=self.pearson_r(y_hat.flatten(), y.flatten()),
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

class BinaryClassificationModule(pl.LightningModule):

    def __init__(self, model, opt_factory):
        super().__init__()
        self.model = model
        self.loss = BCELoss()

        # Decided to use `torcheval` instead of `torchmetrics` for a few 
        # reasons:
        # - `torchmetrics` doesn't have an AUPRC metric.
        # - The `torcheval` API is more explicit, so I'm more confident that 
        #   it's doing what I think it should be.
        # - I'm still mad at `torchmetrics` for covertly installing 
        #   `pretty_errors`.

        self.metrics = {
                k: dict(
                    accuracy=BinaryAccuracy(),
                    precision=BinaryPrecision(),
                    recall=BinaryRecall(),
                    auprc=BinaryAUPRC(),
                    auroc=BinaryAUROC(),
                )
                for k in ['train', 'val', 'test']
        }
        self.metric_preproc = dict(
                recall=(lambda x: x, lambda x: x.int()),
        )
        self.optimizer = opt_factory(model.parameters())

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, step, batch):
        x, y = batch
        y_hat = torch.flatten(self.model(x))

        for key, metric in self.metrics[step].items():
            f_hat, f = self.metric_preproc.get(key, (lambda x: x, lambda x: x))
            metric.update(f_hat(y_hat), f(y))

        loss = self.loss(y_hat, y)
        self.log(f'{step}/loss', loss, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        return self.forward('train', batch)

    def validation_step(self, batch, _):
        return self.forward('val', batch)

    def test_step(self, batch, _):
        return self.forward('test', batch)

    def on_train_epoch_end(self):
        self._on_epoch_end('train')

    def on_validation_epoch_end(self):
        self._on_epoch_end('val')

    def on_test_epoch_end(self):
        self._on_epoch_end('test')

    def _on_epoch_end(self, step):
        for key, metric in self.metrics[step].items():
            self.log(f'{step}/{key}', metric.compute())
            metric.reset()

class PairEncoder(Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        y0 = self.encoder(x[:,0])
        y1 = self.encoder(x[:,1])
        return torch.cat([y0, y1], dim=1)


def get_trainer(out_dir, **kwargs):
    return pl.Trainer(
            logger=TensorBoardLogger(
                save_dir=out_dir.parent,
                name=out_dir.name,
                default_hp_metric=False,
            ),
            **kwargs,
    )

