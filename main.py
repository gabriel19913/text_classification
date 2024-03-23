from flask import Flask, request, Response
import dill
from prediction import predict

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_string():
    string = request.get_data(as_text=True)
    prediction = predict(string)
    return Response(prediction, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
