import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from data_module import DataModule
from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="name of the segmentation model to use",
        dest="model",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="default learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        help="batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="directory containing the data for training",
        dest="data_dir",
    )
    parser.add_argument(
        "-n",
        "--epochs_num",
        type=int,
        help="number of epochs to train",
        dest="epochs_num",
    )
    parser.add_argument(
        "-r",
        "--run_name",
        type=str,
        metavar="R",
        help="name of the run for display in W&B",
        dest="run_name",
    )
    args = parser.parse_args()
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    model = Model(args.model, args.learning_rate)
    data_module = DataModule(
        train_dir, val_dir, "google/t5-efficient-tiny", args.batch_size
    )
    log_dir = Path("training") / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    summary_callback = ModelSummary(max_depth=2)
    wandb_logger = WandbLogger(
        project="grammar-check",
        name=args.run_name,
        log_model="all",
        save_dir=str(log_dir),
        job_type="train",
    )
    wandb_logger.watch(model)
    filename_format = "epoch={epoch}-validation.loss={validation/loss:.5f}"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        filename=filename_format,
        monitor="validation/loss",
        mode="min",
        auto_insert_metric_name=False,
        dirpath=log_dir,
        every_n_epochs=1,
    )
    es_callback = EarlyStopping(monitor="validation/loss", mode="min", patience=3)
    trainer = pl.Trainer(
        max_epochs=args.epochs_num,
        accelerator="gpu",
        devices=-1,
        callbacks=[
            summary_callback,
            checkpoint_callback,
            es_callback,
        ],
        logger=wandb_logger,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    params = {
        "batch_size": data_module.batch_size,
        "epochs_num": args.epochs_num,
        "model": args.model,
        "learning_rate": args.learning_rate,
    }
    wandb_logger.log_hyperparams(params)
    trainer.fit(model=model, datamodule=data_module)
