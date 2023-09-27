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
    max_num = len(size) * kernalNum
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