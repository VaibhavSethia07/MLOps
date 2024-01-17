import pytorch_lightning as pl
import torch
from data import DataModule
from model import ColaModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main():
    cola_data = DataModule()
    cola_model = ColaModel()

    checkpoint_callback = ModelCheckpoint(dirpath='./models', monitor='val_loss', mode='min')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')

    trainer = pl.Trainer(default_root_dir='logs',max_epochs=5, fast_dev_run=False,
                         logger=pl.loggers.TensorBoardLogger(save_dir='logs/', name='cola', version=1),
                         callbacks=[checkpoint_callback, early_stopping_callback])
    trainer.fit(model=cola_model, datamodule=cola_data)


if __name__ == '__main__':
    main()