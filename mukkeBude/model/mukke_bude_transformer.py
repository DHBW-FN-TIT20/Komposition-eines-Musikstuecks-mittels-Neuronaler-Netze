import os
import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Data
BATCH_SIZE = 64
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 450

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000  # Limits parameters in model.

# Training
EPOCHS = 6

# Inference
NUM_TOKENS_TO_GENERATE = 80

class MukkeBudeTransformer():
    def __init__(self, mapping):
        self.mapping = mapping
        
        inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
        # Embedding.
        embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=len(mapping),
            sequence_length=SEQ_LEN,
            embedding_dim=EMBED_DIM,
            mask_zero=True,
        )
        x = embedding_layer(inputs)
        # Transformer decoders.
        for _ in range(NUM_LAYERS):
            decoder_layer = keras_nlp.layers.TransformerDecoder(
                num_heads=NUM_HEADS,
                intermediate_dim=FEED_FORWARD_DIM,
            )
            x = decoder_layer(x)  # Giving one argument only skips cross-attention.
        # Output.
        outputs = keras.layers.Dense(len(mapping))(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
        self.model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])
        self.model.summary()

    def train(self):
        self.model.fit(self.train_ds, validation_data=self.val_ds, verbose=2, epochs=EPOCHS)

    def generate(self, input):
        # Unpadded bos token.
        prompt_tokens = tf.convert_to_tensor([self.tokenizer.token_to_id("[BOS]")])
        output_tokens = keras_nlp.utils.greedy_search(
        self.token_logits_fn,
        prompt_tokens,
        max_length=NUM_TOKENS_TO_GENERATE)
        txt = self.tokenizer.detokenize(output_tokens)
        print(f"Greedy search generated text: \n{txt}\n")

    
    def token_logits_fn(self, inputs):
        cur_len = inputs.shape[1]
        output = self.model(inputs)
        return output[:, cur_len - 1, :]  # return next token logits


    def loadDataset(self):
        keras.utils.get_file(
        origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
        extract=True,
        )
        dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

        # Load simplebooks-92 train set and filter out short lines.
        self.raw_train_ds = (
            tf.data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
            .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
            .batch(BATCH_SIZE)
            .shuffle(buffer_size=256)
        )

        # Load simplebooks-92 validation set and filter out short lines.
        self.raw_val_ds = (
            tf.data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
            .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
            .batch(BATCH_SIZE)
        )
        
        # Train tokenizer vocabulary
        self.vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        self.raw_train_ds,
        vocabulary_size=len(self.mapping),
        lowercase=True,
        reserved_tokens=["[PAD]", "[UNK]", "[BOS]"])
        
        self.tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=self.vocab,
        sequence_length=SEQ_LEN,
        lowercase=True)
        
        # packer adds a start token
        self.start_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=SEQ_LEN,
            start_value=self.tokenizer.token_to_id("[BOS]"),
        )

    def preprocess(self, inputs):
        outputs = self.tokenizer(inputs)
        features = self.start_packer(outputs)
        labels = outputs
        return features, labels

    def getDataset(self):
        # Tokenize and split into train and label sequences.
        self.train_ds = self.raw_train_ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE
        )
        self.val_ds = self.raw_val_ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE
        )
