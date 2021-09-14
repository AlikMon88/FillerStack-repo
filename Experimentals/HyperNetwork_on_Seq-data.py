import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical

### ------- Loading/preprocessnig_the_data ---------###
                                                    
with open('C5_dataset/dinos.txt') as f:
    data = f.read()
data = data.lower()
max_words = (list(set(data)))
samp = data.split('\n')
print('Train_samp: ',len(samp))
print('Total_chars: ',len(data))
print('Max_chars: ', len(max_words))

lis = []
[lis.append(len(i)) for i in samp]
maxlen = max(lis)
#print(lis)
print('max_len: ', maxlen)

char_2_id = {}
id_2_char = {}

for step, i in enumerate(max_words):
    char_2_id[i] = step
    id_2_char[step] = i
print(char_2_id)
print(id_2_char)

x_t = [char_2_id.get(i) for i in data]
y_t = x_t[1:]
y_t.append(char_2_id.get('\n'))
x_t = np.array(x_t)
y_t = np.array(y_t)
x_t = np.reshape(x_t,(-1,1))
y_t = np.reshape(y_t,(-1,1))
print(x_t.shape, y_t.shape)

print(x_t[:5], y_t[:5])
x_train = to_categorical(x_t, num_classes = len(max_words))
y_train = to_categorical(y_t, num_classes = len(max_words))
print(x_train.shape, y_train.shape)

# ------------------------------------------------------------------------#
# -----------------Training_the_HyperNetwork-----------------------------#

classes = 10
input_dim = 27
output_dim = 32

outer_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(max_words), 32, input_length = input_dim),
    tf.keras.layers.SimpleRNN(32, return_sequences = False),
    tf.keras.layers.Dense(classes)
])

num_weights_to_generate = 3835

inner_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(len(max_words), 32, input_length = input_dim),
        tf.keras.layers.SimpleRNN(32, return_sequences = False),
        tf.keras.layers.Dense(num_weights_to_generate, activation=tf.nn.sigmoid)
    ])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

print(x_train.shape, y_train.shape)

dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

dataset = dataset.batch(1)


#@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Predict weights for the outer model.
        weights_pred = inner_model(x)
        
        # Reshape them to the expected shapes for w and b for the outer model.
        # Layer 0 kernel.
        start_index = 0
        w0_shape = (input_dim, output_dim)
        w0_coeffs = weights_pred[:, start_index : start_index + np.prod(w0_shape)]
        w0 = tf.reshape(w0_coeffs, w0_shape)
        start_index += np.prod(w0_shape)
        # Layer 1 kernel
        b0_shape = (output_dim, output_dim)
        b0_coeffs = weights_pred[:, start_index : start_index + np.prod(b0_shape)]
        b0 = tf.reshape(b0_coeffs, b0_shape)
        start_index += np.prod(b0_shape)
        # Layer 1 kernel.
        w1_shape = (output_dim, output_dim)
        w1_coeffs = weights_pred[:, start_index : start_index + np.prod(w1_shape)]
        w1 = tf.reshape(w1_coeffs, w1_shape)
        start_index += np.prod(w1_shape)
        # Layer 1 bias.
        b1_shape = (output_dim,)
        b1_coeffs = weights_pred[:, start_index : start_index + np.prod(b1_shape)]
        b1 = tf.reshape(b1_coeffs, b1_shape)
        start_index += np.prod(b1_shape)

        #layer 2 kernel
        w2_shape = (output_dim, input_dim)
        w2_coeffs = weights_pred[:, start_index : start_index + np.prod(w2_shape)]
        w2 = tf.reshape(w2_coeffs, w2_shape)
        start_index += np.prod(w2_shape)
        # Layer 2 bias.
        b2_shape = (input_dim,)
        b2_coeffs = weights_pred[:, start_index : start_index + np.prod(b2_shape)]
        b2 = tf.reshape(b2_coeffs, b2_shape)
        start_index += np.prod(b2_shape)
        
        #Set the weight predictions as the weight variables on the outer model.
        outer_model.layers[0].set_weights([w0])
        outer_model.layers[1].set_weights([b0, w1, b1])
        outer_model.layers[2].kernel = w2
        outer_model.layers[2].bias = b2
        
        # Inference on the outer model.
        preds = outer_model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, preds, from_logits=True)


    # Train only inner model.
    grads = tape.gradient(loss, inner_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, inner_model.trainable_weights))
    return loss


losses = []  
epochs = 5
for epoch in range(epochs):
    
    for step, (x, y) in enumerate(dataset):
        loss = train_step(x, y)
        losses.append(float(loss))
        if step % 500 == 0:
            print("step:", step, "Loss:", sum(losses) / len(losses))
        # Logging.
        if step >= 1850:
            break
    
    print("epoch:", epoch, "Loss:", sum(losses) / len(losses))
    
#-------------------- Generating_Dino_names---------------------#

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

print(char_2_id)
inp = char_2_id.get('a') ## user Input 
print(inp)
class_range = np.linspace(0,26,27)
x_test = np.array([inp])
print('--'*50)
print('My_Input_char: ', id_2_char.get(inp))
x_test = to_categorical(x_test, num_classes = len(max_words))
print(x_test)
gen_list = []
for i in range(100):
    yhat = outer_model.predict(x_test)
    yhat = softmax(np.reshape(yhat, (-1,1)))
    indx = np.random.choice(class_range, p = np.ravel(yhat))
    gen_list.append(id_2_char.get(indx))
    x_test = np.array([indx])
    x_test = to_categorical(x_test, num_classes = len(max_words))
gen_names = []
s = ''
for st in gen_list:
    if st == '\n':
        gen_names.append(s)
        s = ''
    else:
        s+=st
print(gen_names)
