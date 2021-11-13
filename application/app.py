import flask
from flask import request, Response
import glob
import json
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True

data = []
names = []
files = []

os.chdir((os.path.dirname(os.path.realpath(__file__))))
with open('config.txt', 'r') as file:
    db_path = file.read().split('##',2)[0].strip()

db = glob.glob(db_path+"/*.json")

for filename in db:
       with open(filename) as fh:
            json_str = fh.read()
            json_dict = json.loads(json_str)
            json_dict['file'] = 'https://github.com/Code-Maven/wis-advanced-python-2021-2022/blob/main/students/' + os.path.basename(filename).replace(" ","")
            data.append(json_dict)
            data = sorted(data, key=lambda d: d['name'])
for i in data:
        name = i['name']
        file = i['file']
        names.append(name)
        files.append(file)
@app.route("/")
def index():
    return '<a href="/calc">Find Your Classmate</a>'

@app.route("/calc", methods=['GET'] )
def calc_get():
    return '''<form method="POST" action="/calc">
        <input name="a">
        <input type="submit" value="Search">
        </form>'''

@app.route("/calc", methods=['POST'] )
def calc_post():
    keyword = request.form.get('a')
    keyword = str(keyword)
    zz = list(zip(names, files))
    n = [nam for nam in zz if keyword in nam[0]]
    return Response(json.dumps(n), mimetype='application/json')

app.run()

