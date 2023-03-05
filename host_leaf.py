from flask import Flask, request, jsonify,json
import werkzeug
import cv2
import pickle


app = Flask(__name__)

@app.route('/disease', methods=['POST'])
def disease():
    if(request.method=="POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploaded_images"+ filename)

        leaf_image = cv2.imread("uploaded_images"+filename)
        leaf_image= cv2.cvtColor(leaf_image, cv2.COLOR_BGR2GRAY)

        leaf_image= cv2.resize(leaf_image, (64,64))
        leaf_image.shape= (1,64,64,1)

        prediction_model = pickle.load(open("finalized_model_plant_dis.pkl", 'rb'))
        trans_model = pickle.load(open("transformation_model.pkl", 'rb'))

        ans = trans_model.inverse_transform(prediction_model.predict(leaf_image).argmax(axis=1))

        final_answer = {
            "disease": ans[0]
        }

        response = app.response_class(response=json.dumps(final_answer), status=200,
                                      mimetype='application/json')
        return response
    

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message":"hello world"
    })

if __name__=="__main__":
    app.run(debug=True, port=8080)





