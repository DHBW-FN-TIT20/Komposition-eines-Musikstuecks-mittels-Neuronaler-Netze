# Disable tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

from mukkeBude.model import MukkeBudeTransformer
from mukkeBude.mapping import MusicMapping
import mukkeBude.utils as utils
import music21 as m21
import tensorflow as tf
import keras

# Check if GPU is found
print(tf.config.list_physical_devices('GPU'))

# Create mappings
mapping = MusicMapping.create()

# optional save the mapping
# mapping.save("mapping.txt")



# Train model
model = MukkeBudeTransformer(mapping)
# print(model)

logdir = "logs/bach_transformer"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model.train("raw_train_ds.txt", min_training_seq_len=32, epochs=5, tensorboard_callback=tensorboard_callback)

print(model)

import keras
keras.utils.plot_model(model.model, "my_first_model.png")

model.save("Bach_polyphonie_transformer")

# Generate music

# Load model
lmodel = MukkeBudeTransformer.load(mapping=mapping, name="Bach_polyphonie_transformer")
lmodel.generate("xxbos n67 d4 n62 d4 n58 d4 n43 d4 xxsep d4 n67 d4 n62 d4 n58 d4 n55 d4 xxsep d4 n69 d4 n62 d4 n57 d4 n54 d4 xxsep", max_length=128)