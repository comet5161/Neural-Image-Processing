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
import sqlite3

import transfer
#803dfc4a25ef2ab07807b0c1c3da72fd
conn = sqlite3.connect('db/data.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE  if not exists md5_to_file
       (md5 char(32) PRIMARY KEY     NOT NULL,
       name TEXT    NOT NULL,
       L);''')
conn.commit()

def getFilenameByMD5(md5):
    res = cursor.execute('select name from md5_to_file where md5=?', [md5])
    for row in res:
        return row[0]
    return None

def addFilenameByMD5(md5, filename):
    cursor.execute('insert into md5_to_file(md5, name) values(?,?)', [md5, filename])


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

#主页
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    #return render_template('html/index.html')
    return app.send_static_file('index.html')

#获取风格列表
@app.route('/api/get_styles_list', methods=['GET', 'POST'])
def get_styles_info():
        dirs = glob.glob('models/style_*/')
        data = {}
        data["type"] = "styles_info"
        data["num"] = len(dirs)
        data["styles"] = []
        data['status'] = "ok"
        i = 0
        for x in dirs:
            basename = os.path.basename(x[:-1])
            data["styles"].append({
                "id": i,
                "name": basename[len('style_'):]
            })
            i += 1
        return data

#判断图片是否已上传过
@app.route('/api/is_file_exist', methods=['GET', 'POST'])
def isFileExist():
    print('/api/is_file_exist')
    data = request.get_data()
    json_data = json.loads(data)
    md5 = json_data.get('md5')
    filename = getFilenameByMD5(md5)
    if(filename != None):
        return {"status":"ok",     'file_exist':'true',      "file_name": filename}
    return getResponse(message='false')


#上传图片
@app.route('/api/upload_img', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("upload" )
        print(request)
        print(request.files)
        f = request.files['file']
        print(f.filename)
        if(f.filename.find('/') >= 0 or f.filename.find('\\') >= 0):
            return {"status":"error", 'message':"unvalid filename!"}

        md5 = request.form.get('md5')
        filename = getFilenameByMD5(md5)
        if(filename == None):
            filename = transfer.addPostfix( f.filename, '_' + md5)
            addFilenameByMD5(md5, filename)
            path = os.path.join('uploads', filename)
            f.save(path)
        return {
            "status":"ok",
            "file_name": filename
        }

#开始风格迁移
@app.route('/api/begin_style_transfer', methods=['GET', 'POST'])
def begin_transfer():
    print('begin_transfer')
    data = request.get_data()
    json_data = json.loads(data)

    style_id = int( json_data.get('style_id') )
    file_name = json_data.get('file_name')

    style_name = glob.glob('models/style_*/')[style_id][:-1]
    style_name = os.path.basename(style_name)

    content_img_path = "uploads/" + file_name
    new_name = transfer.addPostfix(file_name, "_" + style_name)
    transfer_img_path = "static/img/" + new_name
    if( os.path.exists(transfer_img_path) == False):
        transfer.beginTransfer(content_img_path, transfer_img_path, style_id)

    res = getResponse(message="transfer success!")
    res['url'] = "img/" + new_name
    res['style_id'] = style_id
    return res



        

    
if __name__ == "__main__":
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    print('server start')
    server.serve_forever()