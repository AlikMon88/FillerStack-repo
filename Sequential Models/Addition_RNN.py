import numpy as np
import random
import tensorflow as tf

max_enc_length = 11
max_dec_length = 10
train_sample = 500000



def gen():
    while 1:
        yield random.randint(3, max_enc_length)

def gen_data(train_samples):
    enc_input = []
    temp_enc = []
    ch = '0123456789'
    for epoch in range(train_samples):
        gen_size = next(gen())
        enc_one = np.zeros(shape=(gen_size), dtype='object')

        for ind, _ in enumerate(enc_one):
            fill = str(random.choice(ch)) 
            enc_one[ind] = fill
            
        plus = int(random.randint(1, gen_size-2))
        enc_one[plus] = '+' 
        
        temp_enc.append(enc_one)
        
        enc_one = ' '.join(enc_one.tolist())       
        enc_input.append(enc_one)
    temp_enc = np.array(temp_enc) 
    
    dec_input = []
    for i in temp_enc:
        lk = []
        i1,i2 = ''.join(i.tolist()).split('+')
        i1, i2 = ''.join(i1), ''.join(i2)
        dec = int(i1) + int(i2)
        lk[:0] = str(dec)
        dec = ' '.join(lk)
        dec_input.append(dec)
        
    
    return enc_input, dec_input

'''
Another form of Data Representation using ' ' + char
'''

enc_input, dec_input = gen_data(train_sample)

print('enc:', enc_input[0], 'dec:',dec_input[0])


inp_token = tf.keras.preprocessing.text.Tokenizer(num_words=max_enc_length,
    filters='!"#$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True)
inp_token.fit_on_texts(enc_input)
x_train = inp_token.texts_to_sequences(enc_input)
print(x_train[0])

dec_token = tf.keras.preprocessing.text.Tokenizer(num_words=max_enc_length-1,
    filters='!"#$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True)
dec_token.fit_on_texts(dec_input)
y_train = inp_token.texts_to_sequences(dec_input)
print(y_train[0])

input_index = inp_token.word_index
target_index = dec_token.word_index

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen = max_enc_length, padding='post', value = 0.0)
y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train, maxlen = max_dec_length, padding='post', value = 0.0)

print(x_train.shape, y_train.shape)

target_index[''] = 0
input_index[''] = 0

x_train = tf.keras.utils.to_categorical(x_train, num_classes = len(list(input_index)))
y_train = tf.keras.utils.to_categorical(y_train, num_classes = len(list(target_index)))


print(x_train.shape, y_train.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(max_enc_length, len(list(input_index)))))## Encoder layer
model.add(tf.keras.layers.RepeatVector(max_dec_length))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))##Decoder_layer
model.add(tf.keras.layers.LSTM(128, return_sequences=True))##Decoder_layer
model.add(tf.keras.layers.LSTM(128, return_sequences=True))##Decoder_layer
model.add(tf.keras.layers.Dense(len(list(target_index)), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

epochs = 30
batch_size = 32
from sklearn.model_selection import train_test_split

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=5)
x_t, x_test, y_t, y_test = train_test_split(x_train, y_train, random_state=20) 


model.fit(
            x_t,
            y_t,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks = [stop_early]
            )
    
inv_target_index = {}
for i, char in target_index.items():
    inv_target_index[char] = i
print(inv_target_index)
inv_inp_index = {}
for i, char in input_index.items():
    inv_inp_index[char] = i
print(inv_inp_index)

yhat = []

for x_t in x_test[:1000]:
    x_t = np.array([x_t])
    pred = model.predict(x_t)
    ind = np.ravel(np.argmax(pred, axis=-1))
    q_ind = np.ravel(np.argmax(x_t, axis=-1))
    
    ques = [ inv_inp_index.get(i) for i in q_ind]
    ans = [ inv_target_index.get(i) for i in ind]
    
    print()
    print(''.join(ques),' = ',''.join(ans))
    print()
    yhat.append(''.join(ans))
print('Done!')
