import numpy as np
import tensorflow as tf
import datetime
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.python.ops import summary_ops_v2
from sklearn.model_selection import train_test_split
import time
from embedFunction import *
import os
MODEL_DIR = "./models"

import pickle
#
#
#
# Read the stored Data
with open('./Dataset/processedData/preprocess.pkl', mode='rb') as f:
    titleLength, titleMap, genres2int, featuresVal, resVal, ratingsData, \
    usersData, movieData, dataset, movieBackup, userBackup = pickle.load(f)

# The embedding layer dimension
embedDim = 32
# The number of the userId
uidMax = max(featuresVal.take(0, 1)) + 1
# The number of the sex
sexMax = max(featuresVal.take(2, 1)) + 1
# The number of the age
ageMax = max(featuresVal.take(3, 1)) + 1
# The number of the occupation
occMax = max(featuresVal.take(4, 1)) + 1
# The number of the movieId
movieIdMax = max(featuresVal.take(1, 1)) + 1
# The number of the movie genres
movieCateMax = max(genres2int.values()) + 1
# The number of the words in movie title
movieTitleMax = 5217

sentenseSize = titleLength
windowSize = {2, 3, 4, 5}
kernalNum = 8
movieIdDict = {val[0]: i for i, val in enumerate(movieData.values)}

# parameter setting
# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 50

save_dir = './save'


def getInputs():
    uid = tf.keras.layers.Input(shape=(1,), dtype='int32', name='uid')
    userSex = tf.keras.layers.Input(shape=(1,), dtype='int32', name='userSex')
    userAge = tf.keras.layers.Input(shape=(1,), dtype='int32', name='userAge')
    userOcc = tf.keras.layers.Input(shape=(1,), dtype='int32', name='userOcc')
    movieId = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movieId')
    movieCategory = tf.keras.layers.Input(shape=(18,), dtype='int32', name='movieCategory')
    movieTitles = tf.keras.layers.Input(shape=(15,), dtype='int32', name='movieTitle')

    return uid, userSex, userAge, userOcc, \
           movieId, movieCategory, movieTitles

def getUserEmbedding(uid, userSex, userAge, userOcc):
    uidLayer = tf.keras.layers.Embedding(uidMax, embedDim, input_length=1, name='uidLayer')(uid)
    sexLayer = tf.keras.layers.Embedding(sexMax, embedDim // 2, input_length=1, name='sexLayer')(userSex)
    ageLayer = tf.keras.layers.Embedding(ageMax, embedDim // 2, input_length=1, name='ageLayer')(userAge)
    occLayer = tf.keras.layers.Embedding(occMax, embedDim // 2, input_length=1, name='occLayer')(userOcc)
    return uidLayer, sexLayer, ageLayer, occLayer

def getUserFeatureLayer(uidLayer, sexLayer, ageLayer, occLayer):
    # the first layer
    uidLayerFull1 = tf.keras.layers.Dense(embedDim, name="uidLayerFull1", activation='relu')(uidLayer)
    sexLayerFull1 = tf.keras.layers.Dense(embedDim, name="sexLayerFull1", activation='relu')(sexLayer)
    ageLayerFull1 = tf.keras.layers.Dense(embedDim, name="ageLayerFull1", activation='relu')(ageLayer)
    occLayerFull1 = tf.keras.layers.Dense(embedDim, name="occLayerFull1", activation='relu')(occLayer)
    # the second layer
    userLayerFull2 = tf.keras.layers.concatenate([uidLayerFull1, sexLayerFull1, ageLayerFull1, occLayerFull1], 2)
    userLayerFull = tf.keras.layers.Dense(200, activation='tanh')(userLayerFull2)

    userLayerFlat = tf.keras.layers.Reshape([200], name="userLayerFlat")(userLayerFull)
    return userLayerFull, userLayerFlat

# Get moiveData into embedding layer
def getMovieIdLayer(movieId):
    movieIdLayer = tf.keras.layers.Embedding(movieIdMax, embedDim, input_length=1, name='movieIdLayer')(movieId)
    return movieIdLayer

# combine the genres of the movie
def getMovieCategoriesLayers(movieCategory):
    movieCategoryLayer = tf.keras.layers.Embedding(movieCateMax, embedDim, input_length=18, name='movieCategorLayer')(movieCategory)
    movieCategoryLayer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(movieCategoryLayer)

    return movieCategoryLayer

# Use CNN to get the embedding layer of the movie title
def getMovieCnnLayer(movieTitle):
    movieTitleLayer = tf.keras.layers.Embedding(movieTitleMax, embedDim, input_length=15, name='movieTitleLayer')(movieTitle)
    sp=movieTitleLayer.shape
    movieTitleLayerExpand = tf.keras.layers.Reshape([sp[1], sp[2], 1])(movieTitleLayer)
    # max-pool the word coventional layer
    poolLayerList = []
    for size in windowSize:
        convLayer = tf.keras.layers.Conv2D(kernalNum, (size, embedDim), 1, activation='relu')(movieTitleLayerExpand)
        maxpoolLayer = tf.keras.layers.MaxPooling2D(pool_size=(sentenseSize - size + 1 ,1), strides=1)(convLayer)
        poolLayerList.append(maxpoolLayer)
    # Dropout Layer
    poolLayer = tf.keras.layers.concatenate(poolLayerList, 3, name ="poolLayer")
    max_num = len(windowSize) * kernalNum
    poolLayerFlat = tf.keras.layers.Reshape([1, max_num], name = "poolLayerFlat")(poolLayer)
    dropoutLayer = tf.keras.layers.Dropout(dropout_keep, name = "dropoutLayer")(poolLayerFlat)
    return poolLayerFlat, dropoutLayer
# conbine the feature of the movie into fully connected network
def getMovieFeatureLayer(movieIdLayer, movieCategoryLayer, dropoutLayer):
    # First layer: Fully connected
    movieIdLayerFull = tf.keras.layers.Dense(embedDim, name="movieIdLayerFull", activation='relu')(movieIdLayer)
    movieCategoryLayer = tf.keras.layers.Dense(embedDim, name="movieCategoryLayer", activation='relu')(movieCategoryLayer)

    # Second Layer: Fully connected
    movieLayerFull = tf.keras.layers.concatenate([movieIdLayerFull, movieCategoryLayer, dropoutLayer], 2)
    movieLayerFull = tf.keras.layers.Dense(200, activation='tanh')(movieLayerFull)
    movieCombineLayerFlat = tf.keras.layers.Reshape([200], name="movie_combine_layer_flat")(movieLayerFull)
    return movieLayerFull, movieCombineLayerFlat

def getBatches(Xs, ys, batchSize):
    for start in range(0, len(Xs), batchSize):
        end = min(start + batchSize, len(Xs))
        yield Xs[start:end], ys[start:end]

class trainNetwork(object):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.best_loss = 9999
        self.losses = {'train': [], 'test': []}

        # 获取输入占位符
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles = getInputs()
        # 获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = getUserEmbedding(uid, user_gender,
                                                                                                   user_age, user_job)
        # 得到用户特征
        user_combine_layer, user_combine_layer_flat = getUserFeatureLayer(uid_embed_layer, gender_embed_layer,
                                                                             age_embed_layer, job_embed_layer)
        # 获取电影ID的嵌入向量
        movie_id_embed_layer = getMovieIdLayer(movie_id)
        # 获取电影类型的嵌入向量
        movie_categories_embed_layer = getMovieCategoriesLayers(movie_categories)
        # 获取电影名的特征向量
        pool_layer_flat, dropout_layer = getMovieCnnLayer(movie_titles)
        # 得到电影特征
        movie_combine_layer, movie_combine_layer_flat = getMovieFeatureLayer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer)
        # 计算出评分
        # 将用户特征和电影特征做矩阵乘法得到一个预测评分的方案
        inference = tf.keras.layers.Lambda(lambda layer:
                                           tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")(
            (user_combine_layer_flat, movie_combine_layer_flat))
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)

        # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        #         inference_layer = tf.keras.layers.concatenate([user_combine_layer_flat, movie_combine_layer_flat],
        #                                                       1)  # (?, 400)
        # 你可以使用下面这个全连接层，试试效果
        # inference_dense = tf.keras.layers.Dense(64, kernel_regularizer=tf.nn.l2_loss, activation='relu')(
        #    inference_layer)
        #         inference = tf.keras.layers.Dense(1, name="inference")(inference_layer)  # inference_dense

        self.model = tf.keras.Model(
            inputs=[uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles],
            outputs=[inference])

        self.model.summary()

        self.optimizer = Adam(learning_rate=0.001)
        # MSE损失，将计算值回归到评分
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

        if tf.io.gfile.exists(MODEL_DIR):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
        test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')

        #         self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
        #         self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def compute_loss(self, labels, logits):
        return tf.reduce_mean(tf.keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        return tf.keras.metrics.mae(labels, logits)  #

    @tf.function
    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        #         metrics = 0
        with tf.GradientTape() as tape:
            logits = self.model([x[0],
                                 x[1],
                                 x[2],
                                 x[3],
                                 x[4],
                                 x[5],
                                 x[6]], training=True)
            loss = self.ComputeLoss(y, logits)
            # loss = self.compute_loss(labels, logits)
            self.ComputeMetrics(y, logits)
            # metrics = self.compute_metrics(labels, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, logits

    def training(self, features, targets_values, epochs=5, log_freq=50):

        for epoch_i in range(epochs):
            # 将数据集分成训练集和测试集，随机种子不固定
            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = getBatches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            #             with self.train_summary_writer.as_default():
            if True:
                start = time.time()
                # Metrics are stateful. They accumulate values and return a cumulative
                # result when you call .result(). Clear accumulated values with .reset_states()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                #                 avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

                # Datasets can be iterated over like any other Python iterable.
                for batch_i in range(batch_num):
                    x, y = next(train_batches)
                    categories = np.zeros([self.batch_size, 18])
                    for i in range(self.batch_size):
                        categories[i] = x.take(6, 1)[i]

                    titles = np.zeros([self.batch_size, sentenseSize])
                    for i in range(self.batch_size):
                        titles[i] = x.take(5, 1)[i]

                    loss, logits = self.train_step([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                                    np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                                    categories.astype(np.float32),
                                                    titles.astype(np.float32)],
                                                   np.reshape(y, [self.batch_size, 1]).astype(np.float32))
                    avg_loss(loss)
                    #                     avg_mae(metrics)
                    self.losses['train'].append(loss)

                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        #                         summary_ops_v2.scalar('loss', avg_loss.result(), step=self.optimizer.iterations)
                        #                         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=self.optimizer.iterations)
                        # summary_ops_v2.scalar('mae', avg_mae.result(), step=self.optimizer.iterations)

                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetrics.result()), rate))
                        # print('Step #{}\tLoss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                        #     self.optimizer.iterations.numpy(), loss, (avg_mae.result()), rate))
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        # avg_mae.reset_states()
                        start = time.time()

            train_end = time.time()
            print(
                '\nTrain time for epoch #{} ({} total steps): {}'.format(epoch_i + 1, self.optimizer.iterations.numpy(),
                                                                         train_end - train_start))
            #             with self.test_summary_writer.as_default():
            self.testing((test_X, test_y), self.optimizer.iterations)
            # self.checkpoint.save(self.checkpoint_prefix)
        self.export_path = os.path.join(MODEL_DIR, 'export')
        tf.saved_model.save(self.model, self.export_path)

    def testing(self, test_dataset, step_num):
        test_X, test_y = test_dataset
        test_batches = getBatches(test_X, test_y, self.batch_size)

        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        #         avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            categories = np.zeros([self.batch_size, 18])
            for i in range(self.batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([self.batch_size, sentenseSize])
            for i in range(self.batch_size):
                titles[i] = x.take(5, 1)[i]

            logits = self.model([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                 np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                 categories.astype(np.float32),
                                 titles.astype(np.float32)], training=False)
            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            avg_loss(test_loss)
            # 保存测试损失
            self.losses['test'].append(test_loss)
            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            # avg_loss(self.compute_loss(labels, logits))
            # avg_mae(self.compute_metrics(labels, logits))

        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), self.ComputeMetrics.result()))
        # print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), avg_mae.result()))
        #         summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
        #         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=step_num)
        # summary_ops_v2.scalar('mae', avg_mae.result(), step=step_num)

        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            print("best loss = {}".format(self.best_loss))
            self.checkpoint.save(self.checkpoint_prefix)

    def forward(self, xs):
        predictions = self.model(xs)
        # logits = tf.nn.softmax(predictions)

        return predictions



#
# trainNet = trainNetwork()
# trainNet.training(featuresVal, resVal, epochs=num_epochs)