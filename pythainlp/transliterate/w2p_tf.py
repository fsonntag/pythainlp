from typing import Union, List

import numpy as np
import tensorflow as tf

from pythainlp.corpus import download, get_corpus_path

_GRAPHEMES = list(
    "พจใงต้ืฮแาฐฒฤๅูศฅถฺฎหคสุขเึดฟำฝยลอ็ม" + " ณิฑชฉซทรฏฬํัฃวก่ป์ผฆบี๊ธญฌษะไ๋นโภ?"
)
_PHONEMES = list(
    "-พจใงต้ืฮแาฐฒฤูศฅถฺฎหคสุขเึดฟำฝยลอ็ม" + " ณิฑชฉซทรํฬฏ–ัฃวก่ปผ์ฆบี๊ธฌญะไษ๋นโภ?"
)

_MODEL_NAME = "thai_w2p"


class _Hparams:
    batch_size = 256
    enc_maxlen = 30 * 2
    dec_maxlen = 40 * 2
    num_epochs = 50 * 2
    hidden_units = 64 * 8
    emb_units = 64 * 4
    graphemes = ["<unk>", "<pad>", "</s>"] + _GRAPHEMES
    phonemes = ["<unk>", "<pad>", "<s>", "</s>"] + _PHONEMES
    lr = 0.001


hp = _Hparams()


def _load_vocab():
    g2idx = {g: idx for idx, g in enumerate(hp.graphemes)}
    idx2g = {idx: g for idx, g in enumerate(hp.graphemes)}

    p2idx = {p: idx for idx, p in enumerate(hp.phonemes)}
    idx2p = {idx: p for idx, p in enumerate(hp.phonemes)}
    # note that g and p mean grapheme and phoneme, respectively.
    return g2idx, idx2g, p2idx, idx2p


class CustomGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, w_ih, w_hh, b_ih, b_hh):
        super(CustomGRUCell, self).__init__()
        self.units = units
        self.initial_w_ih = w_ih
        self.initial_w_hh = w_hh
        self.initial_b_ih = b_ih
        self.initial_b_hh = b_hh

    def build(self, input_shape):
        self.w_ih = tf.constant(
            value=self.initial_w_ih,
            shape=self.initial_w_ih.shape,
        )
        self.w_hh = tf.constant(
            value=self.initial_w_hh,
            shape=self.initial_w_hh.shape,
        )
        self.b_ih = tf.constant(
            value=self.initial_b_ih,
            shape=self.initial_b_ih.shape,
        )
        self.b_hh = tf.constant(
            value=self.initial_b_hh,
            shape=self.initial_b_hh.shape,
        )

    def _sigmoid(self, x):
        return 1 / (1 + tf.exp(-x))

    def call(self, inputs, states):
        x, h = inputs, states
        rzn_ih = tf.linalg.matmul(x, self.w_ih, transpose_b=True) + self.b_ih
        rzn_hh = tf.linalg.matmul(h, self.w_hh, transpose_b=True) + self.b_hh

        rz_ih, n_ih = (
            rzn_ih[:, : rzn_ih.shape[-1] * 2 // 3],
            rzn_ih[:, rzn_ih.shape[-1] * 2 // 3 :],
        )
        rz_hh, n_hh = (
            rzn_hh[:, : rzn_hh.shape[-1] * 2 // 3],
            rzn_hh[:, rzn_hh.shape[-1] * 2 // 3 :],
        )

        rz = self._sigmoid(rz_ih + rz_hh)
        r, z = tf.split(rz, 2, -1)

        n = tf.math.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h


class CustomGRU(tf.keras.layers.Layer):
    def __init__(self, units, w_ih, w_hh, b_ih, b_hh, return_sequences=True):
        super(CustomGRU, self).__init__()
        self.gru_cell = CustomGRUCell(units, w_ih, w_hh, b_ih, b_hh)
        self.return_sequences = return_sequences

    def call(self, x, initial_state=None):
        steps = tf.shape(x)[1]
        if initial_state is None:
            initial_state = [tf.zeros((x.shape[0], self.gru_cell.units))]

        h = initial_state
        if self.return_sequences:
            outputs = tf.TensorArray(tf.float32, size=steps, clear_after_read=False)
            t = tf.constant(0)

            def condition(t, outputs, h):
                return tf.less(t, steps)

            def body(t, outputs, h):
                h = self.gru_cell(x[:, t, :], h)
                outputs = outputs.write(t, h)
                return tf.add(t, 1), outputs, h

            _, outputs, _ = tf.while_loop(condition, body, loop_vars=[t, outputs, h])

            return tf.transpose(outputs.stack(), perm=[1, 0, 2])
        else:
            for t in tf.range(steps):
                h = self.gru_cell(x[:, t, :], h)

            return h


class ThaiW2PTFModel(tf.keras.Model):
    def __init__(self, graphemes: List[str], phonemes: List[str], weights):
        super().__init__()
        self.g2idx, self.idx2g, self.p2idx, self.idx2p = self._load_vocab(
            graphemes, phonemes
        )
        self._init_weights(weights)
        self.enc_gru = CustomGRU(
            units=128,
            w_ih=self.enc_w_ih,
            w_hh=self.enc_w_hh,
            b_ih=self.enc_b_ih,
            b_hh=self.enc_b_hh,
            return_sequences=True,
        )
        self.decGRUCell = CustomGRUCell(
            units=128,
            w_ih=self.dec_w_ih,
            w_hh=self.dec_w_hh,
            b_ih=self.dec_b_ih,
            b_hh=self.dec_b_hh,
        )

        self.enc_emb = tf.constant(self.enc_emb)
        self.dec_emb = tf.constant(self.dec_emb)

        self.fc_w = tf.constant(self.fc_w)
        self.fc_b = tf.constant(self.fc_b)

    def _init_weights(self, weights):
        # (29, 64). (len(graphemes), emb)
        self.enc_emb = weights.item().get("encoder.emb.weight")
        # (3*128, 64)
        self.enc_w_ih = weights.item().get("encoder.rnn.weight_ih_l0")
        # (3*128, 128)
        self.enc_w_hh = weights.item().get("encoder.rnn.weight_hh_l0")
        # (3*128,)
        self.enc_b_ih = weights.item().get("encoder.rnn.bias_ih_l0")
        # (3*128,)
        self.enc_b_hh = weights.item().get("encoder.rnn.bias_hh_l0")

        # (74, 64). (len(phonemes), emb)
        self.dec_emb = weights.item().get("decoder.emb.weight")
        # (3*128, 64)
        self.dec_w_ih = weights.item().get("decoder.rnn.weight_ih_l0")
        # (3*128, 128)
        self.dec_w_hh = weights.item().get("decoder.rnn.weight_hh_l0")
        # (3*128,)
        self.dec_b_ih = weights.item().get("decoder.rnn.bias_ih_l0")
        # (3*128,)
        self.dec_b_hh = weights.item().get("decoder.rnn.bias_hh_l0")
        # (74, 128)
        self.fc_w = weights.item().get("decoder.fc.weight")
        # (74,)
        self.fc_b = weights.item().get("decoder.fc.bias")

    def _load_vocab(self, graphemes: List[str], phonemes: List[str]):
        g2idx = tf.keras.layers.StringLookup(vocabulary=graphemes, oov_token="<unk>")
        idx2g = tf.keras.layers.StringLookup(
            vocabulary=graphemes, oov_token="<unk>", invert=True
        )

        p2idx = tf.keras.layers.StringLookup(vocabulary=phonemes, oov_token="<unk>")
        idx2p = tf.keras.layers.StringLookup(
            vocabulary=phonemes, oov_token="<unk>", invert=True
        )

        # note that g and p mean grapheme and phoneme, respectively.
        return g2idx, idx2g, p2idx, idx2p

    def _encode(self, word: tf.Tensor):
        chars = tf.strings.unicode_split(
            word, "UTF-8"
        )  # Splitting the word into characters
        chars = tf.concat(
            [chars, tf.constant(["</s>"], dtype=tf.string)], axis=0
        )  # Adding the end token
        x = self.g2idx(chars)  # Mapping each character using StringLookup
        x = tf.nn.embedding_lookup(
            self.enc_emb, tf.expand_dims(x, 0)
        )  # Getting the embeddings related to each index
        return x

    def _decode(self, preds):
        preds = self.idx2p(preds)
        return preds

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        enc = self._encode(inputs)
        enc_initial_state = tf.zeros((1, self.enc_w_hh.shape[-1]), tf.float32)
        enc_output = self.enc_gru(enc, enc_initial_state)
        last_hidden = enc_output[:, -1, :]

        # decoder
        dec = tf.gather(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = tf.TensorArray(
            tf.int64, size=0, dynamic_size=True, clear_after_read=False
        )
        t = tf.constant(0)
        reached_end = tf.constant(False)

        def condition(t, preds, h, dec, reached_end) -> tf.Tensor:
            return tf.logical_and(tf.less(t, 20), tf.logical_not(reached_end))

        def body(t, preds, h, dec, reached_end):
            h = self.decGRUCell(dec, h)
            logits = tf.linalg.matmul(h, self.fc_w, transpose_b=True) + self.fc_b
            pred = tf.argmax(logits, axis=-1)
            pred = tf.gather(pred, 0)
            dec = tf.gather(self.dec_emb, [pred], axis=0)
            return (
                t + 1,
                tf.cond(tf.equal(pred, 3), lambda: preds, lambda: preds.write(t, pred)),
                h,
                dec,
                tf.equal(pred, 3),
            )

        _, preds, _, _, _ = tf.while_loop(
            condition, body, loop_vars=[t, preds, h, dec, reached_end]
        )
        decoded_pres = self._decode(preds.stack())

        return decoded_pres


class ThaiW2PTFWithModel:
    def __init__(self):
        super().__init__()
        self.checkpoint = get_corpus_path(_MODEL_NAME, version="0.2")
        if self.checkpoint is None:
            download(_MODEL_NAME, version="0.2")
            self.checkpoint = get_corpus_path(_MODEL_NAME)
        self.graphemes = hp.graphemes
        self.phonemes = hp.phonemes
        self.weights = np.load(self.checkpoint, allow_pickle=True)
        # self.model = ThaiW2PTFModel(
        #     graphemes=self.graphemes, phonemes=self.phonemes, weights=self.weights
        # )
        self.model = tf.keras.models.load_model(
            "thai_w2p_tf_model"
        #     # , custom_objects={"ThaiW2PTFModel": ThaiW2PTFModel}
        )

    def _short_word(self, word: str) -> Union[str, None]:
        self.word = word
        if self.word.endswith("."):
            self.word = self.word.replace(".", "")
            self.word = "-".join([i + "อ" for i in list(self.word)])
            return self.word
        return None

    def save_model(self):
        word_tensor = tf.convert_to_tensor("โทรศัพท์")
        _ = self.model(word_tensor)
        self.model.save("thai_w2p_tf_model", save_format="tf")

    def __call__(self, word: str) -> str:
        if not any(letter in word for letter in self.graphemes):
            pron = [word]
        else:  # predict for oov
            word_tensor = tf.convert_to_tensor(word)
            pron = self.model(word_tensor)
        print(pron.numpy())
        return bytes.decode(b"".join(pron.numpy()))


class ThaiW2PTF:
    def __init__(self):
        super().__init__()
        self.graphemes = hp.graphemes
        self.phonemes = hp.phonemes
        self.g2idx, self.idx2g, self.p2idx, self.idx2p = _load_vocab()
        self.checkpoint = get_corpus_path(_MODEL_NAME, version="0.2")
        if self.checkpoint is None:
            download(_MODEL_NAME, version="0.2")
            self.checkpoint = get_corpus_path(_MODEL_NAME)
        self._load_variables()
        self.enc_gru = CustomGRU(
            units=128,
            w_ih=self.enc_w_ih,
            w_hh=self.enc_w_hh,
            b_ih=self.enc_b_ih,
            b_hh=self.enc_b_hh,
            return_sequences=True,
        )
        self.decGRUCell = CustomGRUCell(
            units=128,
            w_ih=self.dec_w_ih,
            w_hh=self.dec_w_hh,
            b_ih=self.dec_b_ih,
            b_hh=self.dec_b_hh,
        )

        self.enc_emb = tf.Variable(self.enc_emb, trainable=False)
        self.dec_emb = tf.Variable(self.dec_emb, trainable=False)

        self.fc_w = tf.Variable(self.fc_w, trainable=False)
        self.fc_b = tf.Variable(self.fc_b, trainable=False)

    def _load_variables(self):
        self.variables = np.load(self.checkpoint, allow_pickle=True)
        # (29, 64). (len(graphemes), emb)
        self.enc_emb = self.variables.item().get("encoder.emb.weight")
        # (3*128, 64)
        self.enc_w_ih = self.variables.item().get("encoder.rnn.weight_ih_l0")
        # (3*128, 128)
        self.enc_w_hh = self.variables.item().get("encoder.rnn.weight_hh_l0")
        # (3*128,)
        self.enc_b_ih = self.variables.item().get("encoder.rnn.bias_ih_l0")
        # (3*128,)
        self.enc_b_hh = self.variables.item().get("encoder.rnn.bias_hh_l0")

        # (74, 64). (len(phonemes), emb)
        self.dec_emb = self.variables.item().get("decoder.emb.weight")
        # (3*128, 64)
        self.dec_w_ih = self.variables.item().get("decoder.rnn.weight_ih_l0")
        # (3*128, 128)
        self.dec_w_hh = self.variables.item().get("decoder.rnn.weight_hh_l0")
        # (3*128,)
        self.dec_b_ih = self.variables.item().get("decoder.rnn.bias_ih_l0")
        # (3*128,)
        self.dec_b_hh = self.variables.item().get("decoder.rnn.bias_hh_l0")
        # (74, 128)
        self.fc_w = self.variables.item().get("decoder.fc.weight")
        # (74,)
        self.fc_b = self.variables.item().get("decoder.fc.bias")

    def _encode(self, word: str) -> np.ndarray:
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)
        return x

    def _short_word(self, word: str) -> Union[str, None]:
        self.word = word
        if self.word.endswith("."):
            self.word = self.word.replace(".", "")
            self.word = "-".join([i + "อ" for i in list(self.word)])
            return self.word
        return None

    def _predict(self, word: str) -> str:
        short_word = self._short_word(word)
        if short_word is not None:
            return short_word

        # encoder
        enc = self._encode(word)
        # enc_input = tf.constant(enc[np.newaxis, :, :], dtype=tf.float32)
        # enc_output = self.enc_gru(enc_input)
        enc_initial_state = tf.zeros((1, self.enc_w_hh.shape[-1]), tf.float32)
        enc_output = self.enc_gru(enc, enc_initial_state)
        last_hidden = enc_output[:, -1, :]

        # decoder
        dec = tf.gather(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for _ in range(20):
            h = self.decGRUCell(dec, h)
            # h = new_state[0]
            logits = tf.linalg.matmul(h, self.fc_w, transpose_b=True) + self.fc_b
            pred = tf.argmax(logits, axis=-1).numpy()[0]
            if pred == 3:
                break
            preds.append(pred)
            dec = tf.gather(self.dec_emb, [pred], axis=0)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]

        return preds

    def __call__(self, word: str) -> str:
        if not any(letter in word for letter in self.graphemes):
            pron = [word]
        else:  # predict for oov
            pron = self._predict(word)

        return "".join(pron)


_THAI_W2P = ThaiW2PTF()

_THAI_W2P_WITH_MODEL = ThaiW2PTFWithModel()


def _short_word(self, word: str) -> Union[str, None]:
    self.word = word
    if self.word.endswith("."):
        self.word = self.word.replace(".", "")
        self.word = "-".join([i + "อ" for i in list(self.word)])
        return self.word
    return None


def pronunciate(text: str) -> str:
    """
    Convert a Thai word to its pronunciation in Thai letters.

    Input should be one single word.

    :param str text: Thai text to be pronunciated

    :return: A string of Thai letters indicating
             how the input text should be pronounced.
    """
    use_model = True
    if use_model:
        global _THAI_W2P_WITH_MODEL
        return _THAI_W2P_WITH_MODEL(text)

    else:
        global _THAI_W2P
        return _THAI_W2P(text)


if __name__ == "__main__":
    print(pronunciate("โทรศัพท์"))
    # โท-ระ-สับ
    # โท-ระ-สับ
    # _THAI_W2P_WITH_MODEL.save_model()
