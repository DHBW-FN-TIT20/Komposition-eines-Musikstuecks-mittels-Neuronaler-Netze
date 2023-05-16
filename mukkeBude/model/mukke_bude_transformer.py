import os

import keras_nlp
import tensorflow as tf
from tensorflow import keras

from mukkeBude.mapping import MusicMapping

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


class OwnWordPieceTokenizer(keras_nlp.tokenizers.WordPieceTokenizer):
    def __init__(self, detokenize=False, **kwargs):
        super(OwnWordPieceTokenizer, self).__init__(**kwargs)
        self._detokenize = detokenize

    # def call(self, *args, mode="tokenize", training=None, **kwargs):
    #     if self._detokenize:
    #         return self._tokenize_without_call(*args, **kwargs)
    #     return self._detokenize_without_call(*args, **kwargs)
    
    # def call(self, *args, mode="tokenize", training=None, **kwargs):
    #     if mode == "tokenize":
    #         return self._tokenize_without_call(*args, **kwargs)
    #     elif mode == "detokenize":
    #         return self._detokenize_without_call(*args, **kwargs)
    #     else:
    #         raise ValueError(
    #             f"Unsupported tokenizer mode. Received: mode={mode}"
    #         )


# TODO add load and save trained model, add output of diagrams during training (for documentation)
class MukkeBudeTransformer:
    def __init__(
        self,
        mapping: MusicMapping,
        model: keras.Model = None,
        # tokenizer: keras_nlp.tokenizers.WordPieceTokenizer = None,
    ) -> None:
        """Transformer model for MukkeBude.

        :param mapping: Dictionary mapping unique symbols to integers
        :param model: Pretrained model, defaults to None.
        """
        self.mapping = mapping
        self.vocabulary_size = 1000

        if model is not None:
            self.model = model
            # self.tokenizer = tokenizer
            return

        # inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)

        # # Embedding.
        # # Token and position embeddings are ways of representing words and their order in a sentence
        # embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        #     vocabulary_size=self.vocabulary_size,
        #     sequence_length=2048,  # TODO make this dynamic
        #     embedding_dim=EMBED_DIM,  # The output dimension of the embedding layer
        #     mask_zero=True,
        # )
        # x = embedding_layer(inputs)

        # # Transformer decoders.
        # for _ in range(NUM_LAYERS):
        #     decoder_layer = keras_nlp.layers.TransformerDecoder(
        #         num_heads=NUM_HEADS,
        #         intermediate_dim=FEED_FORWARD_DIM,
        #     )
        #     x = decoder_layer(x)  # Giving one argument only skips cross-attention.

        # # Output.
        # outputs = keras.layers.Dense(self.vocabulary_size)(x)
        # self.model = keras.Model(inputs=inputs, outputs=outputs)
        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)

        # self.model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

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

        # Only allow TensorBoard callback
        if tensorboard_callback is not None and not isinstance(tensorboard_callback, keras.callbacks.TensorBoard):
            raise TypeError("Only TensorBoard callback allowed")

        args = {}
        if tensorboard_callback is not None:
            args["callbacks"] = [tensorboard_callback]

        self.model.fit(self.train_ds, verbose=2, epochs=epochs, **args)

        # Add our tokenization into our final model.
        inputs = keras.Input(shape=(), dtype=tf.string)
        tokens = self.tokenizer(inputs)
        outputs = self.model(tokens)
        self.model = keras.Model(inputs, outputs)

        self.model = keras.Model(inputs, outputs, name="mukkeBudeTransformer2")
        keras.utils.plot_model(self.model, "self_model.png")

    def generate(self, start_seed: str, max_length: int = 128, probability=0.5) -> str:
        prompts = start_seed.split(" ")
        # prompt_ids = []
        # for prompt in prompts:
        #     prompt_ids.append(self.tokenizer.token_to_id(prompt))
        # prompt_tokens = tf.convert_to_tensor(prompt_ids)
        prompt_tokens = tf.constant(prompts)

        output_tokens = keras_nlp.utils.top_p_search(
            self.token_probability_fn,
            prompt_tokens,
            max_length=max_length,
            p=probability,
            from_logits=True,
        )
        # txt = self.tokenizer.detokenize(output_tokens).numpy().decode("utf-8")
        return txt

    def token_probability_fn(self, inputs):
        cur_len = inputs.shape[1]
        inputs = tf.reshape(
            inputs,
            [
                cur_len,
            ],
        )
        output = self.model(inputs)
        return output[:, cur_len - 1, :]  # return next token logits

    def save(self, name: str) -> str:
        """Save the model with the given name. The model will be saved in the `model/preTrainedModels` folder.
        It will be saved as a `.h5` file. The extension will be added automatically.

        :param name: Name of the model
        :return: Path to the saved model
        """
        # save model
        model_path = os.path.join(os.path.dirname(__file__), "preTrainedModels", f"{name}.h5")
        self.model.save(model_path)

        # # save tokenizer
        # with open(os.path.join(os.path.dirname(__file__), "preTrainedModels", f"{name}_tokenizer.pickle"), "wb") as f:
        #     pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        return model_path

    @staticmethod
    def load(mapping: MusicMapping, name: str) -> "MukkeBudeTransformer":
        """Load the model with the given name from the `model/preTrainedModels` folder.

        :param mapping: Dictionary mapping unique symbols to integers
        :param name: Name of the model
        :return: Loaded model
        """
        # load model
        path = os.path.join(os.path.dirname(__file__), "preTrainedModels", f"{name}.h5")
        model = keras.models.load_model(path, compile=False)

        # load tokenizer
        # with open(os.path.join(os.path.dirname(__file__), "preTrainedModels", f"{name}_tokenizer.pickle"), "rb") as f:
        #     tokenizer = pickle.load(f)

        return MukkeBudeTransformer(mapping=mapping, model=model)

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

        detokenizer = OwnWordPieceTokenizer(
            vocabulary=self.vocab,
            sequence_length=seq_len,
            lowercase=True,
            detokenize=True,
        )

        # packer adds a start token
        self.start_packer = keras_nlp.layers.StartEndPacker(
            sequence_length=seq_len,
            start_value=self.tokenizer.token_to_id(special_tokens[3]),  # xxbos
        )

        # Add tokenizer to the model and compile it
        self.__compile(self.tokenizer, detokenizer)

    def __compile(self, input_tokenizer: keras.layers.Layer, output_tokenizer: keras.layers.Layer) ->None:
        """ Compile the model. The loss function is `SparseCategoricalCrossentropy` and the metric is `Perplexity`.

        :param inputs: Input Layer form `keras.layers.Input`
        :param outputs: Output Layer
        :return: compiled model
        """

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
        # outputs = output_tokenizer(x)
        # outputs = keras.layers.Dense(self.vocabulary_size)(outputs)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="mukkeBudeTransformer1")
        
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)

        self.model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

        keras.utils.plot_model(self.model, "main_model.png")


    def __preprocess(self, inputs):
        outputs = self.tokenizer(inputs)
        features = self.start_packer(outputs)
        labels = outputs
        return features, labels

    def __str__(self) -> str:
        text = []
        self.model.summary(print_fn=lambda x: text.append(x))
        return "\n".join(text)
