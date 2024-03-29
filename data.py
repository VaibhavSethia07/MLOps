import datasets
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class DataModule(pl.LightningDataModule):
    def __init__(self, model_name='google/bert_uncased_L-2_H-128_A-2', batch_size=64, max_length=128, num_workers=3):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_workers = num_workers

    def prepare_data(self):
        cola_dataset = load_dataset('glue', 'cola')
        print(f'cola_dataset={cola_dataset}')
        self.train_data = cola_dataset['train']
        self.val_data = cola_dataset['validation']

    def tokenize_data(self, example):
        return self.tokenizer(example['sentence'], truncation=True, padding='max_length', max_length=self.max_length)

    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            print(f'Before tokenization train_data[0]={self.train_data[0]}')
            self.train_data = self.train_data.map(self.tokenize_data, batched=True)
            print(f'After tokenization train_data[0]={self.train_data[0]}')

            print(f'Before formatting train_data[0]={self.train_data[0]}')
            self.train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
            print(f'After formatting train_data[0]={self.train_data[0]}')

            print(f'Before tokenization val_data[0]={self.val_data[0]}')
            self.val_data = self.val_data.map(self.tokenize_data, batched=True)
            print(f'After tokenization val_data[0]={self.val_data[0]}')
            self.val_data.set_format(type='torch', columns=['sentence', 'input_ids', 'attention_mask', 'label'])
            print(f'After formatting val_data[0]={self.val_data[0]}')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)


if __name__ == '__main__':
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))['input_ids'].shape)
