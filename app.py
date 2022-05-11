from flask import Flask,request,render_template
from lib.recognize import recognize
import argparse
from easydict import EasyDict as edict
import yaml
import os
import socket
app = Flask(__name__)
#app.config.from_object('./lib/config//flask_config.py')
#img_dir = app.config['img_dir']
img_dir = 'e:/data'
output_dir = 'e:\data'
@app.route('/',methods=['get'])
def index():
    #return render_template('index.html')
    return '皮肤病v1'

@app.route('/recognize',methods=['POST'])
def recogize():
    img = request.files.get('img')
    if img:
        img_name = img.filename
        img_path = os.path.join(img_dir,img_name)
        img.save(img_path)
        return recognize(img_path,output_dir)
    else:
        return 'upload failed'

def parse_arg():
    parser = argparse.ArgumentParser(description="皮肤病识别")

    parser.add_argument('--cfg',default='./lib/config//config.yaml', help='configuration',  type=str)

    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    return config
if __name__ == '__main__':
    config = parse_arg()
    HOST = socket.gethostbyname(socket.gethostname())
    print(HOST)
    app.run(config.HOST,config.PORT)
    #app.run(HOST, config.PORT)
