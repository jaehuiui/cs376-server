from crypt import methods
from distutils.log import debug
from urllib import request
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_prediction(data) :
    return data + ' -> result'

@app.route('/')
def hello():
    return 'Hello World'

@app.route('/model')
def detection():
    data = request.args.get('data')

    if(data == ''):
        return 'hi'

    else:
        return get_prediction(data)

if __name__ == '__main__' :
    app.run(host='0.0.0.0', debug=True, port=8000)