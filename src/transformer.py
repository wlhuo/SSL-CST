import os
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[0], "GPU")
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import gc

def z_score_normalization(data):
    means = np.mean(data, axis=(0, 1))
    stds = np.std(data, axis=(0, 1))
    normalized_data = (data - means) / stds
    return normalized_data

def z_score_normalization_(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    normalized_data = (data - means) / stds
    return normalized_data

def dir_to_class(y_dir, class_num):
    y_dir_class = []
    for i in range(len(y_dir)):
        x, y = y_dir[i] 
        if x == -9999:
            y_vec = np.zeros(class_num) 
            y_dir_class.append(y_vec)
        else:
            if y == 0 and x > 0: 
                deg = np.arctan(float('inf'))
            elif y == 0 and x < 0:
                deg = np.arctan(-float('inf'))
            elif y == 0 and x == 0:
                deg = np.arctan(0)
            else:
                deg = np.arctan((x/y))
            if (x > 0 and y < 0) or (x <= 0 and y < 0):
                deg += np.pi
            elif x < 0 and y >= 0:
                deg += 2 * np.pi
            cla = int(deg / (2 * np.pi / class_num))
            y_vec = np.zeros(class_num)
            y_vec[cla] = 1 
            y_dir_class.append(y_vec)
    return np.array(y_dir_class) 

def lr_schedule(epoch):
    initial_learning_rate = 0.1
    decay_rate = 0.1
    decay_steps = 10
    lr = initial_learning_rate * decay_rate**(epoch // decay_steps)
    return lr

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    return x

def repmlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = repmlp_layer(x, units)
        x = layers.Dropout(dropout_rate)(x)

    return x

def repmlp_layer(x, units):
    x = RepMLPLayer(units)(x)

    return x

class RepMLPLayer(layers.Layer):
    def __init__(self, units):
        super(RepMLPLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight("kernel", shape=(input_dim, self.units), initializer="random_normal", trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.kernel)
        x = tf.nn.gelu(x)

        return x

def masked_categorical_cross_entropy(y_true, y_pred):
    cce = keras.losses.CategoricalCrossentropy(from_logits=True)
    return cce(y_true, y_pred, sample_weight=tf.math.reduce_sum(y_true, axis=-1)) 

class PatchEncoder(layers.Layer): 
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Dense(units=projection_dim)
        self.distance_embedding = layers.Dense(units=projection_dim)

    def call(self, patch, position,distance): 
        encoded = self.projection(patch) + self.position_embedding(position) + self.distance_embedding(distance)
        return encoded

def create_transformer_classifier(class_num, input_shape, input_position_shape, input_distance_shape, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units):
    inputs = layers.Input(shape=input_shape) 
    inputs_positions = layers.Input(shape=input_position_shape)
    inputs_distances = layers.Input(shape=input_distance_shape)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(inputs, inputs_positions,inputs_distances)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = repmlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    features = repmlp(representation[:, 0, :], hidden_units=mlp_head_units, dropout_rate=0.5)
    pos = layers.Dense(class_num, name='pos_out')(features)
    binary = layers.Dense(1, activation='sigmoid', name='cat_out')(features)

    model = keras.Model(inputs=[inputs, inputs_positions, inputs_distances], outputs=[pos, binary])
    return model

def run_experiment(startx, starty, patchsize, model, x_train, x_train_pos, x_train_distance, x_train_, x_train_pos_, x_train_distance_, y_train, y_train_, y_binary_train, x_test, x_test_pos, x_test_distance, x_validation, x_validation_pos, x_validation_distance, y_validation, y_binary_validation, learning_rate, weight_decay, batch_size, num_epochs,result_fold):
    
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss={
            'pos_out': masked_categorical_cross_entropy,
            'cat_out': keras.losses.BinaryCrossentropy(from_logits=False),
        },
        metrics={
            'pos_out': keras.metrics.CategoricalAccuracy(name="accuracy"),
        },
    )

    checkpoint_filepath = os.path.join('./ckpt', 'model_' + startx + '_' + starty + '_' + patchsize, 'ckpt')
    
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    if len(x_validation[np.where(y_binary_validation == 1)]) < 100:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="pos_out_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
    else:
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_pos_out_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
    tf_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    
    model.fit(
        x=[x_train, x_train_pos, x_train_distance], 
        y=[y_train, y_binary_train], 
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.0,
        validation_data=([x_validation[np.where(y_binary_validation == 1)], x_validation_pos[np.where(y_binary_validation == 1)], x_validation_distance[np.where(y_binary_validation == 1)]], [y_validation[np.where(y_binary_validation == 1)], y_binary_validation[np.where(y_binary_validation == 1)]]),
        callbacks=[checkpoint_callback,tf_callback],
    )
    
    print('Inference on all the spots...')
    model.load_weights(checkpoint_filepath)
    pred_centers_test_all = []
    pred_binary_test_all = []
    for i in range(int(len(x_test) / 10000) + 1):
        pred_centers_test_, pred_binary_test_ = model.predict(x = [x_test[i*10000: (i+1)*10000], x_test_pos[i*10000: (i+1)*10000], x_test_distance[i*10000: (i+1)*10000]], batch_size=batch_size)
        pred_centers_test_all.append(pred_centers_test_)
        pred_binary_test_all.append(pred_binary_test_)
        gc.collect()
    pred_centers_test = np.vstack(pred_centers_test_all)
    pred_binary_test = np.vstack(pred_binary_test_all)

    pred_centers_train_all = []
    pred_binary_train_all = []
    for i in range(int(len(x_train_) / 10000) + 1):
        pred_centers_train_, pred_binary_train_ = model.predict(x = [x_train_[i*10000: (i+1)*10000], x_train_pos_[i*10000: (i+1)*10000],x_train_distance_[i*10000: (i+1)*10000]], batch_size=batch_size)
        pred_centers_train_all.append(pred_centers_train_)
        pred_binary_train_all.append(pred_binary_train_)
        gc.collect() 
    pred_centers_train = np.vstack(pred_centers_train_all)
    pred_binary_train = np.vstack(pred_binary_train_all)
    x_train_pos__ = np.load('data/x_train_pos_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_train_pos__ = x_train_pos__['x_train_pos']
    x_train_distance__ = np.load('data/x_train_distance_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_train_distance__ = x_train_distance__['x_train_distance']
    x_test_pos_ = np.load('data/x_test_pos_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_test_pos_ = x_test_pos_['x_test_pos']
    x_test_distance_ = np.load('data/x_test_distance_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_test_distance_ = x_test_distance_['x_test_distance']

    print('Write prediction results...')
    with open(os.path.join(result_fold,'spot_prediction_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.txt'), 'w') as fw:
        for i in range(len(y_train_)):
            fw.write(str(x_train_pos__[i][0][0]) + '\t' + str(x_train_pos__[i][0][1]) + '\t' + str(pred_binary_train[i][0]) + '\t' + ':'.join([str(c) for c in pred_centers_train[i]]) + '\n')
        for i in range(len(x_test_pos_)):
            fw.write(str(x_test_pos_[i][0][0]) + '\t' + str(x_test_pos_[i][0][1]) + '\t' + str(pred_binary_test[i][0]) + '\t' + ':'.join([str(c) for c in pred_centers_test[i]]) + '\n')

    return

def train(startx, starty, patchsize, epochs, val_ratio, nucleus_mask, result_fold):
    startx = str(startx)
    starty = str(starty)
    patchsize = str(patchsize)
    try:
        os.mkdir(result_fold)
    except FileExistsError:
        print('{} exists.'.format(result_fold))
    x_train_ = np.load('data/x_train_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_train_ = x_train_['x_train'].astype(np.float32)
    x_train_pos_ = np.load('data/x_train_pos_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_train_pos_ = x_train_pos_['x_train_pos'].astype(np.int32)
    x_train_distance_ = np.load('data/x_train_distance_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_train_distance_ = x_train_distance_['x_train_distance'].astype(np.float32)
    y_train_ = np.load('data/y_train_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    y_train_ = y_train_['y_train']
    y_binary_train_ = np.load('data/y_binary_train_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    y_binary_train_ = y_binary_train_['y_binary_train'].astype(np.int32)
    x_test = np.load('data/x_test_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_test = x_test['x_test'].astype(np.float32)
    x_test_pos = np.load('data/x_test_pos_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_test_pos = x_test_pos['x_test_pos'].astype(np.int32)
    x_test_distance = np.load('data/x_test_distance_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz')
    x_test_distance = x_test_distance['x_test_distance'].astype(np.float32)
    class_num = 16

    x_train_select = []
    x_validation_select = []
    for i in range(len(x_train_pos_)): 
        if x_train_pos_[i][0][0] > int(nucleus_mask.shape[0] * (1 - np.sqrt(val_ratio))) and x_train_pos_[i][0][1] > int(nucleus_mask.shape[1] * (1 - np.sqrt(val_ratio))):
            x_validation_select.append(i) 
        else:
            x_train_select.append(i)

    for i in range(len(y_train_)): 
        if y_train_[i][0] != -1:
            y_train_[i] = y_train_[i] - x_train_pos_[i][0]
        else:
            y_train_[i][0] = -9999
            y_train_[i][1] = -9999
    y_train_ = dir_to_class(y_train_, class_num)
    for i in range(len(x_train_pos_)): 
        for j in range(1, len(x_train_pos_[i])):
            x_train_pos_[i][j] = x_train_pos_[i][j] - x_train_pos_[i][0]
        x_train_pos_[i][0] = x_train_pos_[i][0] - x_train_pos_[i][0]
    for i in range(len(x_test_pos)):
        for j in range(1, len(x_test_pos[i])):
            x_test_pos[i][j] = x_test_pos[i][j] - x_test_pos[i][0]
        x_test_pos[i][0] = x_test_pos[i][0] - x_test_pos[i][0]

    x_train = x_train_[x_train_select]
    x_train_pos = x_train_pos_[x_train_select]
    x_train_distance = x_train_distance_[x_train_select]
    y_train = y_train_[x_train_select]
    y_binary_train = y_binary_train_[x_train_select]
    x_validation = x_train_[x_validation_select]
    x_validation_pos = x_train_pos_[x_validation_select]
    x_validation_distance = x_train_distance_[x_validation_select]
    y_validation = y_train_[x_validation_select]
    y_binary_validation = y_binary_train_[x_validation_select]

    input_shape = (x_train.shape[1], x_train.shape[2])
    input_position_shape = (x_train_pos.shape[1], x_train_pos.shape[2])
    input_distance_shape = (x_train_distance.shape[1],x_train_distance.shape[2])

    learning_rate = 0.002
    weight_decay = 0.0001
    batch_size = 100
    num_epochs = epochs
    num_patches = x_train.shape[1]
    projection_dim = 64
    num_heads = 1
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  
    transformer_layers = 8
    mlp_head_units = [1024, 256] 

    transformer_classifier = create_transformer_classifier(class_num, input_shape, input_position_shape, input_distance_shape, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units)
    run_experiment(startx, starty, patchsize, transformer_classifier, x_train, x_train_pos, x_train_distance, x_train_, x_train_pos_, x_train_distance_, y_train, y_train_, y_binary_train, x_test, x_test_pos, x_test_distance, x_validation, x_validation_pos, x_validation_distance, y_validation, y_binary_validation, learning_rate, weight_decay, batch_size, num_epochs,result_fold)
