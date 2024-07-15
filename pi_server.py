from flask import Flask
from collectdata import collect

app = Flask(__name__)

@app.route('/collect')
def coll():
    collect()
    return open("testing.csv").read()

app.run(host='0.0.0.0', port=5013)