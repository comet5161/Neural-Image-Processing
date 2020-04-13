from flask import Flask
from flask_socketio import SocketIO,emit
from flask_sockets import Sockets
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import os
import datetime
import time
import random
import glob
import json

import transfer

app = Flask(__name__,static_url_path='')
app.config['SECRET_KEY'] = 'secret!'
sockets = Sockets(app)
# socketio = SocketIO(app)

# socketio.run(app, debug=True)

def getResponse(status = "ok", message = ""):
    return {"status": status, "message": message}

@sockets.route('/echo')
def echo_socket(ws):
    while not ws.closed:
        now = datetime.datetime.now().isoformat() + 'Z'
        ws.send(now)  #发送数据
        time.sleep(1)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    #return render_template('html/index.html')
    return app.send_static_file('index.html')

@app.route('/api/get_styles_list', methods=['GET', 'POST'])
def get_styles_info():
        dirs = glob.glob('models/style_*')
        data = {}
        data["type"] = "styles_info"
        data["num"] = len(dirs)
        data["styles"] = []
        data['status'] = "ok"
        i = 0
        for x in dirs:
            basename = os.path.basename(x)
            data["styles"].append({
                "id": i,
                "name": basename[len('style_'):]
            })
            i += 1
        return data

@app.route('/api/begin_style_transfer', methods=['GET', 'POST'])
def begin_transfer():
    print('begin_transfer')
    data = request.get_data()
    json_data = json.loads(data)

    style_id = int( json_data.get('style_id') )
    file_name = json_data.get('file_name')

    style_name = glob.glob('models/style_*')[style_id]
    style_name = os.path.basename(style_name)

    content_img_path = "uploads/" + file_name
    new_name = transfer.addPostfix(file_name, "_" + style_name)
    transfer_img_path = "static/img/" + new_name

    transfer.beginTransfer(content_img_path, transfer_img_path, style_id)

    res = getResponse(message="transfer success!")
    res['url'] = "img/" + new_name
    res['style_id'] = style_id
    return res



@app.route('/api/upload_img', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("upload" )
        print(request)
        print(request.files)
        f = request.files['file']
        print(f.filename)
        new_name = transfer.addPostfix(secure_filename(f.filename),  str(time.time()) )
        path = os.path.join('uploads', new_name)
        f.save(path)
        return {
            "status":"ok",
            "file_name": new_name
        }
    else:
        return {
            "status":"error",
            "message": "please use POST method!"
        }
        

    
if __name__ == "__main__":
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    print('server start')
    server.serve_forever()