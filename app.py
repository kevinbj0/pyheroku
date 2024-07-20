import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user = request.get_json()
        if user and user.get('username') == 'admin' and user.get('password') == '1234':
            return jsonify({'success': True, 'message': 'Valid user'})
        return jsonify({'success': False, 'message': 'Invalid user'})
    return '''<h1>Hello World</h1>'''

@app.route("/login", methods=['POST'])
def login():
    user = request.get_json()
    if user and user.get('username') == 'admin' and user.get('password') == '1234':
        return jsonify({'success': True, 'message': 'Valid user'})
    return jsonify({'success': False, 'message': 'Invalid user'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
