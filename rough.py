
import pickle
prediction_model = pickle.load(open("finalized_model_plant_dis.pkl", 'rb'))

print(prediction_model.summary())
