from flask import Flask, render_template, url_for, redirect, request
import tensorflow as tf
from keras.utils import to_categorical
import numpy as np

app = Flask(__name__, template_folder = 'template2')

dino_path = r'C:\Users\User\Desktop\Jupy2\dino_model.h5'
with open(r'C:\Users\User\Desktop\Jupy2\C5_dataset\dinos.txt') as f:
    data = f.read()
data = data.lower()
max_words = (list(set(data)))

char_2_id = {}
id_2_char = {}

char_2_id = {'f': 0, 'h': 1, 'm': 2, 'x': 3, 'i': 4, 'j': 5, 'z': 6, '\n': 7, 's': 8, 'y': 9, 't': 10, 'q': 11, 'w': 12, 'k': 13, 
             'g': 14, 'o': 15, 'p': 16, 'b': 17, 'v': 18, 'd': 19, 'n': 20, 'a': 21, 'l': 22, 
             'e': 23, 'c': 24, 'u': 25, 'r': 26}

id_2_char = {0: 'f', 1: 'h', 2: 'm', 3: 'x', 4: 'i', 5: 'j', 6: 'z', 7: '\n', 8: 's', 9: 'y', 10: 't', 11: 'q', 12: 'w', 
             13: 'k', 14: 'g', 15: 'o', 16: 'p', 17: 'b', 18: 'v', 19: 'd', 20: 'n', 21: 'a', 22: 'l', 23: 'e',
             24: 'c', 25: 'u', 26: 'r'}
    
## Producing Decent Names
def generator(inp_str, first_model):
    inp = char_2_id.get(inp_str) ## user Input 
    class_range = np.linspace(0,26,27)
    x_test = np.array([inp])
    print('--'*50)
    print('My_Input_char: ', id_2_char.get(inp))
    x_test = to_categorical(x_test, num_classes = 27)
    print(x_test)
    gen_list = []
    for i in range(100):
        yhat = first_model.predict(x_test)
        indx = np.random.choice(class_range, p = np.ravel(yhat))
        gen_list.append(id_2_char.get(indx))
        x_test = np.array([indx])
        x_test = to_categorical(x_test, num_classes = len(max_words))
    gen_names = []
    s = ''
    for st in gen_list:
        if st == '\n':
            if len(s) > 4:
                gen_names.append(s)
            s = ''
        else:
            s+=st
    print(gen_names)
    return gen_names
    
@app.route('/') #home-page binding function
def home_page():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    inp_str = str(request.form['seed'])
    out_str = generator(inp_str, dino_model)
    return render_template('home.html', pred = 'Generated_Dino_Names: {}'.format(out_str))
if __name__ == '__main__':
    dino_model = tf.keras.models.load_model(dino_path)
    app.run()
