from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_prediction(data) :
    return data + ' -> result'

@app.route('/')
def hello():
    return 'Hello World / Heroku'

@app.route('/model')
def detection():
    data = request.args.get('data')

    if(data == ''):
        return 'hi'

    else:
        return get_prediction(data)

if __name__ == '__main__' :
    app.run(debug=True, port=5000)