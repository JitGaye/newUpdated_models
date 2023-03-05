 
from tensorflow import keras


loaded_model = keras.models.load_model('predictionmodel')

print(loaded_model.summary())

