import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import LSTM, Embedding, Dense, Bidirectional, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def attentionLayer(input_layer):
    attention = Dense(1, activation='tanh')(input_layer)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(64)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    output_attention_mul = tf.keras.layers.Multiply()([input_layer, attention])
    output_attention_mul = LSTM(32)(output_attention_mul)

    return output_attention_mul


def build_model(max_tokens, num_words, embedding_dim, embedding_matrix):
    sequence_input = Input(shape=(max_tokens,), dtype='float32')
    embedded_inputs = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_tokens,
                                trainable=False)(sequence_input)
    lstm1 = Bidirectional(LSTM(32, return_sequences=True))(embedded_inputs)
    # lstm2 =LSTM(16)(lstm1) 
    attention = attentionLayer_new(lstm1)
    dense1 = tf.keras.layers.Dense(16)(attention)
    dense1 = tf.keras.layers.Dropout(0.5)(dense1)
    dense = Dense(1, activation='sigmoid')(dense1)
    return Model(sequence_input, dense)


def attentionLayer_new(input_layer):
    # input_dim = int(input_layer.shape[2])
    attention_vector = layers.TimeDistributed(layers.Dense(1))(input_layer)
    attention_vector = layers.Reshape((401,))(attention_vector)
    attention_vector = layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = layers.Dot(axes=1)([input_layer, attention_vector])
    return attention_output
