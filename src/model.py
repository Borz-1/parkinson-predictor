from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


def build_model(input_shape, use_batchnorm=False, dropout_rate=0.3):
    
    model = Sequential()
    
    model.add(Dense(128, activation='relu', input_shape=(input_shape,)))
    
    if use_batchnorm:
        model.add(BatchNormalization())
    
    #model.add(Dropout(dropout_rate))
    #model.add(Dense(64, activation='relu'))

    #model.add(Dropout(dropout_rate))
    #model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'recall']
    )
    
    return model


