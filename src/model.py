from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD



def build_model(input_shape, hidden_layers=[128, 64], activation='relu', use_batchnorm=True, dropout_rate=0.3, optimizer="adam"):
    
    model = Sequential()
    
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_shape,)))
    
    if use_batchnorm:
        model.add(BatchNormalization())
        
    model.add(Dropout(dropout_rate))
    
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=activation))
        
        if use_batchnorm:
            model.add(BatchNormalization())
            
        model.add(Dropout(dropout_rate))
    
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    
    optimizers = {
        'adam': Adam(),
        'rmsprop': RMSprop(),
        'sgd': SGD()
    }
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.get(optimizer, Adam()),
                  metrics=['accuracy', 'Recall'])
    
    return model


