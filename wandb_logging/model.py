import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification


class ColaModel(pl.LightningModule):
    def __init__(self, model_name='google/bert_uncased_L-2_H-128_A-2', lr=3e-5):
        super(ColaModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.num_classes = 2
        self.validation_step_outputs = []

        self.train_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(task='multiclass', average='macro',
                                                             num_classes=self.num_classes)
        self.recall_macro_metric = torchmetrics.Recall(task='multiclass', average='macro',
                                                       num_classes=self.num_classes)
        self.precision_micro_metric = torchmetrics.Precision(task='multiclass', average='micro',
                                                             num_classes=self.num_classes)
        self.recall_micro_metric = torchmetrics.Recall(task='multiclass', average='micro',
                                                       num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch['input_ids'], batch['attention_mask'], labels=batch['label'])
        # loss = F.cross_entropy(logits, batch['label'])
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch['label'])
        self.log('train/loss', outputs.loss, prog_bar=True, on_epoch=True)
        self.log('train/acc', train_acc, prog_bar=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch['label']
        outputs = self.forward(batch['input_ids'], batch['attention_mask'],
                               labels=batch['label'])
        preds = torch.argmax(outputs.logits, 1)

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log('valid/loss', outputs.loss, prog_bar=True, on_step=True)
        self.log('valid/acc', valid_acc, prog_bar=True, on_epoch=True)
        self.log('valid/precision_macro', precision_macro, prog_bar=True, on_epoch=True)
        self.log('valid/recall_macro', recall_macro, prog_bar=True, on_epoch=True)
        self.log('valid/precision_micro', precision_micro, prog_bar=True, on_epoch=True)
        self.log('valid/recall_micro', recall_micro, prog_bar=True, on_epoch=True)
        self.log('valid/f1', f1, prog_bar=True, on_epoch=True)

        result = {'labels': labels, 'logits': outputs.logits}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        labels = torch.cat([x['labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        preds = torch.argmax(logits, 1)

        # There are multiple ways to track the metrics
        # 1. Confusion matrix plotting using inbuilt W&B method
        self.logger.experiment.log(
            {
                'conf': wandb.plot.confusion_matrix(
                    probs=logits.numpy(), y_true=labels.numpy()
                )
            }
        )

        # 2. Confusion Matrix plotting using scikit-learn method
        wandb.log({'cm': wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds)})

        # 3. Confusion Matric plotting using Seaborn
        data = confusion_matrix(labels.numpy(), preds.numpy())
        df_cm = pd.DataFrame(data, columns=np.unique(labels), index=np.unique(labels))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        plt.figure(figsize=(7, 4))
        plot = sns.heatmap(
            df_cm, cmap='Blues', annot=True, annot_kws={'size': 16}
        )  # font size
        self.logger.experiment.log({'Confusion Matrix': wandb.Image(plot)})

        self.logger.experiment.log(
            {'roc': wandb.plot.roc_curve(labels.numpy(), logits.numpy())}
        )
        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
