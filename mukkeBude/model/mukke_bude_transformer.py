from transformers import TFTransfoXLLMHeadModel, TFTransfoXLModel, TransfoXLConfig, AdamWeightDecay, TransfoXLTokenizer, TFTrainer, TFTrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_dataset
import numpy as np
import json
from pathlib import Path
import tensorflow as tf

class MukkeBudeTransformer():
    def __init__(self):
        self.tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initializing a Transformer XL configuration
        configuration = TransfoXLConfig()
        self.model = TFTransfoXLModel.from_pretrained("transfo-xl-wt103")

        self.optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        # self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # self.model.compile(optimizer=optimizer, metrics=["accuracy"])
        
    def train(self, data, batch_size: int = 16):
        
        train_dataset = self.getDataset(data, batch_size)
        self.model.compile(
            optimizer=self.optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
        )
        # self.model.build(input_shape=(None, 1020, 64))
        self.model.summary()
        # self.model.fit(x=train_dataset, epochs=10)
        training_args = TFTrainingArguments(
            output_dir="./",
            num_train_epochs=3,
            auto_find_batch_size=True,
            do_train=True
        )
        
        # with training_args.strategy.scope():
        #     model = TFTransfoXLModel.from_pretrained("transfo-xl-wt103")
            
        trainer = TFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )
        
        trainer.train()
        
    
    def generate_sequence(self, input):
        pass

    def _loadDataset(self, data: list):
        dataset = load_dataset("json", data_files="./data.json")
        return dataset

    def _preprocess_dataset(self, dataset):            
        inputs = self.tokenizer(dataset["input"],  truncation=True, padding="max_length",  max_length=64)
        labels = self.tokenizer(dataset["labels"], truncation=True, padding="max_length",  max_length=1, return_tensors="tf")
        # model_inputs = {
        #     "inputs": inputs,
        #     "labels": labels
        # }
        inputs["label"] = labels.input_ids
        return inputs

    def getDataset(self, data: list, batch_size: int = 16):
        #Get raw data from JSON
        raw_dataset = self._loadDataset(data)
        #Tokenzize data
        tokenized_dataset = raw_dataset.map(self._preprocess_dataset)
        print(tokenized_dataset)
        print(tokenized_dataset["train"]["input"][0])
        print(tokenized_dataset["train"]["label"][0])
        print(tokenized_dataset["train"]["input_ids"][0])
        print(tokenized_dataset["train"]["input"][1])
        print(tokenized_dataset["train"]["label"][1])
        print(tokenized_dataset["train"]["input_ids"][1])
        temp = tokenized_dataset["train"]["label"][0]
        #init DataCollator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        
        train_dataset = tokenized_dataset["train"].to_tf_dataset(
            columns=["input_ids"],
            label_cols=["label"],
            shuffle=True,
            batch_size=32,
            collate_fn=data_collator
        )

        return train_dataset
