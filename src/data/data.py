import pytorch_lightning as pl
from functools import partial

from torch.utils.data import DataLoader
from .utils import collate
from .dataset import MetaDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.subset_train = MetaDataset(self.args.data_file, args,
                splits_to_load=["train"], sample_size=args.sample_size)
        self.subset_val = MetaDataset(self.args.data_file, args,
                splits_to_load=["val"], sample_size=args.sample_size)

    def train_dataloader(self):
        train_loader = DataLoader(self.subset_train,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=(not self.args.debug),
                                  collate_fn=partial(collate, self.args))
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.subset_val,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            persistent_workers=(not self.args.debug),
                            collate_fn=partial(collate, self.args))
        return val_loader

class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.subset_predict = MetaDataset(args.data_file, args,
                splits_to_load=["test"], sample_size=args.sample_size)

    def predict_dataloader(self):
        predict_loader = DataLoader(self.subset_predict,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=partial(collate, self.args))
        return predict_loader