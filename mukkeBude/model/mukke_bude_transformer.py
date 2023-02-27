from transformers import TFTransfoXLModel, AdamWeightDecay, AutoTokenizer
from datasets import Dataset, load_dataset
import numpy as np
import json
from pathlib import Path
import tensorflow as tf

class MukkeBudeTransformer():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = TFTransfoXLModel.from_pretrained("transfo-xl-wt103")

        optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer, metrics=["accuracy"])
        
    def train(self, data, batch_size: int = 16):
        
        dataset = self.getDataset(data, batch_size)
        self.model.fit(x=dataset, epochs=1, batch_size=batch_size)
        
    
    def generate_sequence(self, input):
        pass

    def _loadDataset(self, data: list):

        # dataset = Dataset.from_list(data)
        dataset = load_dataset("json", data_files="./data.json")
        return dataset

    def _tokenize(self, example):
        return self.tokenizer(example["input"], example["labels"], truncation=True, padding="max_length", max_length=64)
        # for input, labels in zip(example["input"], example["labels"]):
        #     example["input"] = self.tokenizer(input, padding="max_length", truncation=True, max_length=64)
        #     example["labels"] = self.tokenizer(labels, padding="max_length", truncation=True, max_length=64)

        # return example
        # return self.tokenizer(example["input"][0], example["labels"][0], truncation=True)

    def getDataset(self, data: list, batch_size: int = 16):
        dataset = self._loadDataset(data)
        dataset = dataset.map(self._tokenize, batched=True)
        tensorflow_dataset = dataset["train"].to_tf_dataset(columns=["input_ids","labels"], shuffle=True, batch_size=batch_size)
        return tensorflow_dataset
