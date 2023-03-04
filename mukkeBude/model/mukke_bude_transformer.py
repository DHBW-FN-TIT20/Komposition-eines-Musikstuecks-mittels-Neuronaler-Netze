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


class MukkeBudeTransformer:
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

    def train(
        self,
        path: os.PathLike,
        vocabulary_size: int,
        min_training_seq_len: int = 64,
        batch_size: int = 64,
        buffer_size: int = 256,
    ) -> tf.keras.callbacks.History:
        self.__loadDataset(path, vocabulary_size, min_training_seq_len, batch_size, buffer_size)

        self.train_ds = self.raw_train_ds.map(self.__preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE,
        )
        return self.model.fit(self.train_ds, verbose=2, epochs=EPOCHS)

    def generate(self, input: str, max_length: int = 128) -> str:
        # Unpadded bos token.
        prompt_tokens = tf.convert_to_tensor([self.tokenizer.token_to_id(input)])

        print(f"prompt_tokens: {prompt_tokens}")

        output_tokens = keras_nlp.utils.greedy_search(
            self.token_logits_fn,
            prompt_tokens,
            max_length=max_length,
        )
        txt = self.tokenizer.detokenize(output_tokens)
        clean_txt = self.__removeTokens(txt)
        return clean_txt

    def token_logits_fn(self, inputs):
        cur_len = inputs.shape[1]
        output = self.model(inputs)
        return output[:, cur_len - 1, :]  # return next token logits

    def __loadDataset(
        self,
        path: os.PathLike,
        vocabulary_size: int,
        min_training_seq_len: int = 64,
        batch_size: int = 64,
        buffer_size: int = 256,
    ):
        # Load train set and filter out short lines.
        self.raw_train_ds = (
            tf.data.TextLineDataset(path)
            .filter(lambda x: tf.strings.length(x) > min_training_seq_len)
            .batch(batch_size)
            .shuffle(buffer_size=buffer_size)
        )

        # Train tokenizer vocabulary
        vocabulary_size += 3  # Add 3 for [PAD], [UNK], [BOS].
        self.vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
            self.raw_train_ds,
            vocabulary_size=vocabulary_size,
            lowercase=True,
            reserved_tokens=[
                "[PAD]",  # padding token used to pad sequences to the same length /
                "[UNK]",  # out-of-vocabulary (OOV) sub-words / unknown words are replaced with this token
                "[BOS]",  # stands for beginning of sentence, but here technically it is a token representing the beginning of each line of training data
            ],
        )

        # Load tokenizer
        # WordPieceTokenizer is an efficient implementation of the WordPiece algorithm used by BERT and other models.
        # It will strip, lower-case and do other irreversible preprocessing operations
        self.tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=self.vocab,
            sequence_length=SEQ_LEN,
            lowercase=True,
        )

        # packer adds a start token
        self.start_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=SEQ_LEN,
            start_value=self.tokenizer.token_to_id("[BOS]"),
        )

    def __preprocess(self, inputs):
        outputs = self.tokenizer(inputs)
        features = self.start_packer(outputs)
        labels = outputs
        return features, labels

    def __removeTokens(self, inputs):
        return self.__removeStartToken(self.__removePadding(inputs))

    def __removePadding(self, inputs):
        return tf.strings.regex_replace(inputs, "\s{0,2}\[PAD\]\s{0,2}", " ")

    def __removeStartToken(self, inputs):
        return tf.strings.regex_replace(inputs, "\s{0,2}\[BOS\]\s{0,2}", " ")

    def __str__(self) -> str:
        text = []
        self.model.summary(print_fn=lambda x: text.append(x))
        return "\n".join(text)
