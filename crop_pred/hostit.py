from flask import Flask,request,jsonify, json

import werkzeug
import numpy
import cv2
import pandas
import os
import pickle
from tensorflow import keras


app = Flask(__name__)

@app.route('/leaf_prediction', methods=['GET'])
def leaf_prediction():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    imagefile.save("./leaf_images_uploaded/"+filename)

    leaf_image = cv2.imread("leaf_images_uploaded/"+filename)
    print("leaf_image is ")
    print(leaf_image)
    leaf_image= cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)
    leaf_image= cv2.resize(leaf_image, (64,64))
    leaf_image.shape=(1,64,64,1)

    trans_model = pickle.load(open("transformation_model.pkl", 'rb'))
    loaded_model = keras.models.load_model('predictionmodel.h5')
    print("model summary")
    print(loaded_model.summary())

    ans = trans_model.inverse_transform(loaded_model.predict(leaf_image).argmax(axis=1))

    final_answer = {
        "disease": ans[0]
    }

    response = app.response_class(response=json.dumps(final_answer),status=200,
                                  mimetype='application/json')
    
    return response


@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message":"hello world"
    })

if __name__ == "__main__":
    app.run(debug=True, port=7070)






