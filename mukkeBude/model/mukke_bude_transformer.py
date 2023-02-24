import tensorflow_models as tfm
import inspect

class MukkeBudeTransformer(tfm.nlp.layers.TransformerXL):
    def __init__(self, *args, encode_position=True, mask_steps=1, **kwargs):       
        sig = inspect.signature(tfm.nlp.layers.TransformerXL)
        arg_params = { k:kwargs[k] for k in sig.parameters if k in kwargs }
        super().__init__(*args, **arg_params)
        self.encode_position = encode_position            
        self.mask_steps=mask_steps