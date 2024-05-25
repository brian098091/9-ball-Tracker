from flask import Flask, request, jsonify
import subprocess  # 導入 subprocess 模塊
from flask_cors import CORS
import numpy as np
import base64
import cv2
import sys
import random
from view import View
from queue import Queue
from game import Game

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/process_frame": {"origins": "http://127.0.0.1:5500"}})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    VideoMsec = int(data['msec'])
    X = round(float(data['x']))
    Y = round(float(data['y']))
    H = int(data['h'])
    S = int(data['s'])
    V = int(data['v'])
    fileName = str(data['file_name'])
    try:
        vidcap = cv2.VideoCapture('./resources/'+fileName)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, VideoMsec)
        ret, j = vidcap.read()
        j = cv2.cvtColor(j, cv2.COLOR_BGR2HSV)

        g = Game(vidcap, 4)
        mask = g.set_tcrange_ff(j, (Y, X), gap=np.array([H,S,V], dtype='uint8'))
        # print('Please wait, 3Q')
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'mask_image': 'data:image/png;base64,' + mask_base64})
    except subprocess.CalledProcessError as e:
        print(f"Error running game.py: {e}")
        return jsonify({"error": "Error processing game.py"}), 500
    
    return jsonify({"message": "Game processing completed"}), 200

if __name__ == '__main__':
    app.run(debug=True)
