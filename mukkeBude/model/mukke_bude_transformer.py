from transformers import TFTransfoXLModel, AdamWeightDecay, AutoTokenizer
from datasets import load_dataset

class MukkeBudeTransformer():
    def __init__(self):    
        self.tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")   
        self.model = TFTransfoXLModel.from_pretrained("transfo-xl-wt103")
        
    def train(self, train_arr):
        optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
        self.model.compile(optimizer=optimizer, metrics=["accuracy"])
        history = self.model.fit(x=train_arr, epochs=5) #geht nicht weil der die daten in [x,y] haben will. also y=target. Also so umwandeln wie bei LSTM
        return history
        
        
    
    def generate_sequence(self, input):
        pass