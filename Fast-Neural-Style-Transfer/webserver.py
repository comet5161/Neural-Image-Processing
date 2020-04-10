from flask import Flask
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import os
import transfer
import time
import glob

app = Flask(__name__,static_url_path='')

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    #return render_template('html/index.html')
    return app.send_static_file('index.html')


@app.route('/upload_img', methods=['GET', 'POST'])
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
        

        style_id = 4
        model_dir = glob.glob('models/style_*')[style_id]
        model_name = os.path.basename(model_dir)

        style_img_name = transfer.addPostfix(new_name, "_"+model_name)
        transfer_path = "static/img/" + style_img_name

        res = transfer.beginTransfer(path, transfer_path, style_id)

        if(res != None):
            return {"status": "ok", "url": "img/"+style_img_name}
        else :
            return {"status":"failed"}
    return 'upload failed!'

    
