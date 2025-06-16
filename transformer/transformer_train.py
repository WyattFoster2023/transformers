import tensorflow as tf
from tokenizers import Tokenizer
from keras import layers, Model

seq_len     = 128
vocab_size  = 50000    # from your tokenizer
d_model     = 64
raw_seq_len = seq_len + 1   # one extra for the shift

### Tokenizer ###

# 1. Load your tokenizer
hf_tokenizer = Tokenizer.from_file("transformer/data/tokenizer-model.json")

# 2. A tf.py_function wrapper
def encode_fn(text):
    # text is a scalar tf.string
    toks = hf_tokenizer.encode(text.numpy().decode("utf-8")).ids
    return tf.constant(toks, dtype=tf.int32)

def tf_encode(text):
    toks = tf.py_function(encode_fn, inp=[text], Tout=tf.int32)
    toks.set_shape([None])
    # 1) truncate
    toks = toks[:raw_seq_len]
    # 2) pad
    pad_amount = raw_seq_len - tf.shape(toks)[0]
    toks = tf.cond(
      pad_amount > 0,
      lambda: tf.pad(toks, [[0, pad_amount]]),
      lambda: toks
    )
    toks.set_shape([raw_seq_len])
    return toks

### Dataset ###

raw_ds = tf.data.TextLineDataset("transformer/data/KJV.txt")  # or adapt from HuggingFace streaming

def make_lm(tokens):
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y

token_ds = (
    raw_ds
      .map(tf_encode)
      .padded_batch(32, padded_shapes=[seq_len])
      .map(make_lm)
)

### Transformer ###

def transformer_block(
    inputs,                 # (batch, seq_len, d_model)
    num_heads: int = 4,
    d_ff: int = seq_len,
    dropout_rate: float = 0.1
):
    # 1. Multi-head self-attention + residual + layer-norm
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=inputs.shape[-1]
    )(inputs, inputs)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # 2. Feed-forward network + residual + layer-norm
    ff = layers.Dense(d_ff, activation="relu")(out1)
    ff = layers.Dense(inputs.shape[-1])(ff)
    ff = layers.Dropout(dropout_rate)(ff)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ff)

    return out2

# Example: build a tiny model using that block

# 1. Inputs
token_ids = layers.Input(shape=(seq_len,), dtype=tf.int32)

# 2. Embedding + positional encoding
x = layers.Embedding(vocab_size, d_model)(token_ids)
positions = tf.range(start=0, limit=seq_len, delta=1)
pos_emb = layers.Embedding(seq_len, d_model)(positions)
x = x + pos_emb

# 3. One transformer block
x = transformer_block(x, num_heads=4, d_ff=256)
logits = layers.Dense(vocab_size)(x)   # per-token logits
model = Model(token_ids, logits)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.summary()


if __name__ == "__main__":
    history = model.fit(token_ds, epochs=1)
    print("Loss:", history.history["loss"][0])

    model.save("transformer/data/transformer-model.keras")
    print("Model saved to transformer/data/transformer-model.keras")

    # 4. Inference
    ids = hf_tokenizer.encode("Hello, world!").ids
    ids = ids + [0] * (seq_len - len(ids))
    pred = model.predict(tf.expand_dims(ids, 0))  # shape (1, seq_len, vocab_size)
    print(pred.shape)

    with open("transformer/data/test_inference.txt", "w") as f:
        f.write(str(pred.tolist()))