import torch
from data import DataModule
from model import ColaModel


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        # loading the trained model
        self.model = ColaModel.load_from_checkpoint(model_path)
        # keep the model in eval mode
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ['unacceptable', 'acceptable']

    def predict(self, text):
        """Get the predictions
            Args:
            text (str): The input sentence to classify as either acceptable or unacceptable.
            Returns: 
            A dictionary containing the predicted label and its probability score.
        """
        # text - run time input
        inference_sample = {'sentence': text}
        # tokenizing the input
        processed = self.processor.tokenize_data(inference_sample)
        # predictions
        logits = self.model(torch.tensor([processed['input_ids']]),
                            torch.tensor([processed['attention_mask']]))

        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({'label': label, 'score': score})

        return predictions


if __name__ == '__main__':
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor('./models/best-checkpoint.ckpt-v1.ckpt')
    print(predictor.predict(text=sentence))