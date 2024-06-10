from flask import Flask, request, jsonify
import subprocess  
from flask_cors import CORS
import numpy as np
import base64
import cv2
import sys
import random
from game import Game
import os
from pytube import YouTube

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/process_frame": {"origins": "http://127.0.0.1:5500"}})
# CORS(app, resources={r"/find_ball_baize": {"origins": "http://127.0.0.1:5500"}})

g = None
DOWNLOAD_FOLDER = 'downloads'
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

@app.route('/download_video', methods=['POST'])
def download_video():
    
    data = request.json
    video_url = data.get('url')
    start = data.get('start_time')
    end = data.get('endTime')
    
    yt = YouTube(video_url)
    stream = yt.streams.filter(res="720p", progressive=True, file_extension='mp4').first()
    download_path = './downloads'
    downloaded_file = stream.download(output_path=download_path)

    # video = VideoFileClip(downloaded_file).subclip(start, end)

    # edited_file = download_path + f'/{yt.title}_edit.mp4'
    # video.write_videofile(edited_file, codec='libx264')
    
    return jsonify({'file_path': downloaded_file})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global g  
    data = request.json
    VideoMsec = int(data['msec'])
    X = round(float(data['x']))
    Y = round(float(data['y']))
    H = int(data['h'])
    S = int(data['s'])
    V = int(data['v'])
    fileName = str(data['file_name'])
    try:
        vidcap = cv2.VideoCapture('./downloads/'+fileName)
        vidcap.set(cv2.CAP_PROP_POS_MSEC, VideoMsec)
        ret, j = vidcap.read()
        j = cv2.cvtColor(j, cv2.COLOR_BGR2HSV)

        g = Game(vidcap, 2)
        ret = g.set_tcrange_ff(j, (Y, X), gap=np.array([H,S,V], dtype='uint8'))
        # print('Please wait, 3Q')
        print(ret.shape)
        _, buffer = cv2.imencode('.jpg', ret)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'mask_image': 'data:image/png;base64,' + mask_base64})
    except subprocess.CalledProcessError as e:
        print(f"Error running game.py: {e}")
        return jsonify({"error": "Error processing game.py"}), 500
    
@app.route('/find_ball_baize', methods=['POST'])
def find_ball_baize():
    global g  
    print('Please wait, 3Q')
    g.sep_views(log_images=True)
    
    data = request.json
    balls_group = data['selectedImageOption']

    if balls_group == 1:
        dir = './balls'
    else:
        dir = './balls2'
    ball_dists = []
    for num in range(10):
        img = cv2.imread(f'{dir}/{num}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        size = (img.shape[0], img.shape[1])
        mask = np.full(size, 0, dtype=np.uint8)
        cv2.circle(mask, (size[0]//2, size[1]//2), min(*size)//2, 1, -1)
        fill_size = np.sum(mask)
        h = cv2.calcHist([img],[0],mask,[180],[0, 180]) / fill_size
        s = cv2.calcHist([img],[1],mask,[256],[0, 256]) / fill_size
        v = cv2.calcHist([img],[2],mask,[256],[0, 256]) / fill_size
        ball_dists.append([h,s,v])
    # ball_dists.shape == (10, 3, 180/256)

    g.proc_frames(ball_dists, True)
    return jsonify({"status": "success", "message": "Mask processing confirmed"})

if __name__ == '__main__':
    app.run(debug=True)
