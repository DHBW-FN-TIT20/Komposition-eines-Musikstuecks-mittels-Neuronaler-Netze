import os

import keras_nlp
import tensorflow as tf
from tensorflow import keras

# Data
BATCH_SIZE = 64

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 4
VOCAB_SIZE = 5000  # Limits parameters in model.

# Inference
NUM_TOKENS_TO_GENERATE = 80

#TODO add load and save trained model, add output of diagrams during training (for documentation)
class MukkeBudeTransformer:
    def __init__(self, mapping):
        self.mapping = mapping
        # self.vocabulary_size = len(mapping) + 1 # +1 for [UNK] token
        self.vocabulary_size = 1000

        inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)

        # Embedding.
        # Token and position embeddings are ways of representing words and their order in a sentence
        embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
            vocabulary_size=self.vocabulary_size,
            sequence_length=2048,  # TODO make this dynamic
            embedding_dim=EMBED_DIM,  # The output dimension of the embedding layer
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
        outputs = keras.layers.Dense(self.vocabulary_size)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)

        self.model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

    def train(
        self,
        path: os.PathLike,
        min_training_seq_len: int = 64,
        seq_len: int = 128,
        batch_size: int = 64,
        buffer_size: int = 256,
        epochs: int = 12,
        tensorboard_callback: keras.callbacks.TensorBoard = None,
    ) -> tf.keras.callbacks.History:
        self.__loadDataset(
            path=path,
            min_training_seq_len=min_training_seq_len,
            seq_len=seq_len,
            batch_size=batch_size,
            buffer_size=buffer_size,
        )

        self.train_ds = self.raw_train_ds.map(self.__preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE,
        )

        # Only allow TenosrBoard callback
        if tensorboard_callback is not None and not isinstance(tensorboard_callback, keras.callbacks.TensorBoard):
            raise TypeError("Only TensorBoard callback allowed")
        
        args = {}
        if tensorboard_callback is not None:
            args["callbacks"] = [tensorboard_callback]

        return self.model.fit(self.train_ds, verbose=2, epochs=epochs, **args)

    def generate(self, input: str, max_length: int = 128, probability=0.5) -> str:
        # Unpadded bos token.
        # TODO consider using a sequence of input (split by ' ' and tokenize every element)
        prompts = input.split(' ')
        prompt_ids = []
        for prompt in prompts:
            prompt_ids.append(self.tokenizer.token_to_id(prompt))
        prompt_tokens = tf.convert_to_tensor(prompt_ids)
        # prompt_tokens = tf.convert_to_tensor([self.tokenizer.token_to_id(input)])
        print(f"prompt_tokens: {prompt_tokens}")

        output_tokens = keras_nlp.utils.top_p_search(
            self.token_logits_fn,
            prompt_tokens,
            max_length=max_length,
            p=probability,
            from_logits=True
        )
        txt = self.tokenizer.detokenize(output_tokens)
        return txt

    def token_logits_fn(self, inputs):
        cur_len = inputs.shape[1]
        output = self.model(inputs)
        return output[:, cur_len - 1, :]  # return next token logits
    
    def save(self, name: str) -> str:
        """Save the model with the given name. The model will be saved in the `model/preTrainedModels` folder.

        :param name: Name of the model
        :return: Path to the saved model
        """
        path = os.path.join(os.path.dirname(__file__), "preTrainedModels", name)
        self.model.save(path)
        return path

    @staticmethod
    def load(name: str):
        """Load the model with the given name. The model will be loaded from the `model/preTrainedModels` folder.

        :param name: Name of the model
        """
        path = os.path.join(os.path.dirname(__file__), "preTrainedModels", name)
        return tf.keras.models.load_model(path)

    def __loadDataset(
        self,
        path: os.PathLike,
        min_training_seq_len: int = 16,
        seq_len: int = 32,
        batch_size: int = 16,
        buffer_size: int = 256,
        special_tokens: list = ["xxpad", "[UNK]", "xxbos", "xxeos", "xxsep"],
    ):
        """load the dataset for the model. Each line in the dataset file is a training example.

        :param path: Path to the dataset
        :param min_training_seq_len: Each line in the file with less than `min_training_seq_len` will be removed, defaults to 64
        :param seq_len: Length that each line got splittet for the model, defaults to 128
        :param batch_size: How many lines to give the model (does not batch the line), defaults to 16
        :param buffer_size: Buffer size for shuffel each lines in the dataset, defaults to 256
        :param special_tokens: Tokens that need to be included in the vocabulary. First Token need to be the default padding token, defaults to ["xxpad", "[UNK]", "xxbos", "xxeos", "xxsep"]
        """
        # Load train set and filter out short lines.
        self.raw_train_ds = (
            tf.data.TextLineDataset(path)
            .filter(lambda x: tf.strings.length(x) > min_training_seq_len)
            .batch(batch_size)
            .shuffle(buffer_size=buffer_size)
        )

        # Train tokenizer vocabulary
        self.vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
            self.raw_train_ds,
            vocabulary_size=self.vocabulary_size,
            lowercase=True,
            reserved_tokens=special_tokens,
            # reserved_tokens=[
            #     "[PAD]",  # padding token used to pad sequences to the same length /
            #     "[UNK]",  # out-of-vocabulary (OOV) sub-words / unknown words are replaced with this token
            #     "[BOS]",  # stands for beginning of sentence, but here technically it is a token representing the beginning of each line of training data
            # ],
        )

        # Load tokenizer
        # WordPieceTokenizer is an efficient implementation of the WordPiece algorithm used by BERT and other models.
        # It will strip, lower-case and do other irreversible preprocessing operations
        self.tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary=self.vocab,
            sequence_length=seq_len,
            lowercase=True,
        )

        # packer adds a start token
        self.start_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=seq_len,
            start_value=self.tokenizer.token_to_id(special_tokens[3]),  # xxbos
        )

    def __preprocess(self, inputs):
        outputs = self.tokenizer(inputs)
        features = self.start_packer(outputs)
        labels = outputs
        return features, labels

    def __str__(self) -> str:
        text = []
        self.model.summary(print_fn=lambda x: text.append(x))
        return "\n".join(text)
