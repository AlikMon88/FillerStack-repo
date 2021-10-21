import tensorflow as tf
import random

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 221532  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = r"../input/english-to-german/deu.txt"

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
#'''
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    target_text = '[START] ' + target_text + ' [END]'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
#'''
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

temp = list(zip(input_texts, target_texts))
random.shuffle(temp)
input_texts, target_texts = zip(*temp)

input_texts = input_texts[217532:]
target_texts = target_texts[217532:]
print(len(input_texts), len(target_texts))

import numpy as np
token_encoder = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
token_encoder.fit_on_texts(input_texts)
inp_index = token_encoder.texts_to_sequences(input_texts)
token_decoder = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
token_decoder.fit_on_texts(target_texts)
out_index = token_decoder.texts_to_sequences(target_texts)

out_index = np.array(out_index)
inp_index = np.array(inp_index)

## taking after 3000 samples due to memory bottleneck
encoder_input = tf.keras.preprocessing.sequence.pad_sequences(inp_index, maxlen = max_encoder_seq_length, padding='post')
decoder_input = tf.keras.preprocessing.sequence.pad_sequences(out_index, maxlen = max_decoder_seq_length, padding='post')

input_token_index = token_encoder.word_index
target_token_index = token_decoder.word_index

print(encoder_input.shape, decoder_input.shape)
decoder_inp = decoder_input[:,:-1]
target_seq = decoder_input.reshape(decoder_input.shape[0], decoder_input.shape[1],1)[:,1:]
print(target_seq.shape)
 
## Training Mode 

latent_dim_2 = 500
units = 264
encoder = tf.keras.layers.LSTM(latent_dim_2, return_state=True, return_sequences=True, name='Encoder')
decoder = tf.keras.layers.LSTM(latent_dim_2, return_state=True, return_sequences = True, recurrent_dropout=0.2, name='Decoder')
decoder_dense = tf.keras.layers.Dense(len(list(target_token_index))+1, activation='softmax')
attention_layer = tf.keras.layers.Attention(causal = True)
conc_att = tf.keras.layers.Concatenate(axis=-1)

embed_enc = tf.keras.layers.Embedding(input_dim = len(list(input_token_index))+1,output_dim = 500)
embed_dec = tf.keras.layers.Embedding(input_dim = len(list(target_token_index))+1,output_dim = 500)

encoder_input_1 = tf.keras.layers.Input(shape=(encoder_input.shape[-1],), name='en_1')
enc_inp = embed_enc(encoder_input_1)
lstm_1 = tf.keras.layers.LSTM(latent_dim_2, return_sequences=True, return_state=True)(enc_inp)
lstm_2 = tf.keras.layers.LSTM(latent_dim_2, return_sequences=True, return_state=True)(lstm_1)
encoder_output, end_h, end_c = encoder(lstm_2)

encoder_states = [end_h, end_c]
encoder_all = [encoder_output, end_h, end_c]

decoder_input_1  = tf.keras.layers.Input(shape=(None,), name='dec_1')
dec_inp = embed_dec(decoder_input_1)
decoder_output,_ ,_ = decoder(dec_inp, initial_state=encoder_states)
decoder_att ,_ = attention_layer([decoder_output, encoder_output], return_attention_scores = True) ##decoder_output = query_tensor, encoder_output = value tensor

concat = conc_att([decoder_output, decoder_att])
dense_2 = decoder_dense(concat)

model_2 = tf.keras.models.Model(inputs = [encoder_input_1, decoder_input_1], outputs= dense_2)

model_2.compile(loss='sparse_categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
model_2.summary()
tf.keras.utils.plot_model(model_2, show_shapes=True)

stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',patience=5)

print(encoder_input.shape, decoder_inp.shape, target_seq.shape)
hist = model_2.fit([encoder_input, decoder_inp], target_seq, epochs=20, validation_split=0.2, batch_size=32, callbacks=[stop_early])

## Inference_Mode

from IPython.display import display
encoder_model = tf.keras.models.Model(encoder_input_1, encoder_all)

decoder_state_input_h = tf.keras.layers.Input(shape=(latent_dim_2,))
decoder_state_input_c = tf.keras.layers.Input(shape=(latent_dim_2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
encoder_ops = tf.keras.layers.Input(shape = (None,latent_dim_2), name = 'encoder_ops')
dec_emb = embed_dec(decoder_input_1)
decoder_outputs, state_h, state_c = decoder(dec_emb, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
attention_op ,_ = attention_layer([decoder_outputs, encoder_ops], return_attention_scores=True)
conc = conc_att([decoder_outputs, attention_op])
decoder_outputs = decoder_dense(conc)
decoder_model = tf.keras.models.Model(
    inputs = [decoder_input_1] + [encoder_ops, decoder_states_inputs],
    outputs = [decoder_outputs] + decoder_states)

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    enc_op, en_h, en_c = encoder_model.predict(input_seq)
    states_value = [en_h, en_c]                               
    print(len(states_value))
    
    target_seq = np.array([[target_token_index['[start]']]])
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [enc_op, states_value])

        sampled_token_index = np.argmax(np.ravel(output_tokens[0, -1, :]))
        sampled_char = reverse_target_char_index[sampled_token_index]
        print(sampled_char)
        if (sampled_char == '[end]' or len(decoded_sentence) > max_decoder_seq_length):
            break
        
        decoded_sentence += ' ' + sampled_char

        target_seq = np.array([[sampled_token_index]])
        
        states_value = [h, c]

    return decoded_sentence

k1 = tf.keras.utils.plot_model(encoder_model, show_shapes=True)
k2 = tf.keras.utils.plot_model(decoder_model, show_shapes=True)

print('Encoder_Model_inf')
display(k1)
print('--'*20)
print('Decoder_Model_inf')
display(k2)


## Prediction
inp_str = "Tom has visited Boston several times"
real_str = inp_str
inp_str = [inp_str]
inp_str_index = token_encoder.texts_to_sequences(inp_str)
encoder_str_input = tf.keras.preprocessing.sequence.pad_sequences(inp_str_index, maxlen = max_encoder_seq_length, padding='post')
decoded_sentence = decode_sequence(encoder_str_input)

print('-')
print('Input sentence:', real_str)
print('Decoded sentence:', decoded_sentence, ', Length: ',len(decoded_sentence.split(' ')))
print('-')

