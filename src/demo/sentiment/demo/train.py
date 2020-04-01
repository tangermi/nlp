from gensim.models import KeyedVectors
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import data_helper
import mymodel


def preprocess():
    cn_model = KeyedVectors.load_word2vec_format('data/chinese_word_vectors/sgns.zhihu.bigram', binary=False)
    print("预训练词向量加载完毕")
    train_tokens = data_helper.loadFile_tokenize(cn_model)
    max_tokens = data_helper.getMaxTokens(train_tokens)
    X_train, X_test, y_train, y_test = data_helper.getData(train_tokens, max_tokens)
    embedding_matrix = data_helper.buildEmbeddingLayer(cn_model)
    return X_train, X_test, y_train, y_test, embedding_matrix, max_tokens


def getCallbacks():
    path_checkpoint = 'sentiment_checkpoint.keras'
    checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                 verbose=1, save_weights_only=True,
                                 save_best_only=True)
    try:
        model.load_weights(path_checkpoint)
    except Exception as e:
        print(e)
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
    callbacks = [earlystopping, checkpoint, lr_reduction]
    return callbacks


def train(model):
    # callbacks = getCallbacks()

    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.1, epochs=60, batch_size=256, callbacks=None)
    result = model.evaluate(X_test, y_test)
    print('Accuracy:{0:.2%}'.format(result[1]))
    model.save('my_trained_model.h5')


if __name__ == "__main__":
    num_words = 50000
    embedding_dim = 300
    X_train, X_test, y_train, y_test, embedding_matrix, max_tokens = preprocess()
    model = mymodel.build_model(max_tokens, num_words, embedding_dim, embedding_matrix)
    model.summary()
    train(model)
