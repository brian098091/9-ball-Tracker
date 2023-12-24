import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
import numpy as np
import cv2
from table import Table
from findTable import FindTable

video_name = "resources/edited.mp4"

def update_duration(event):
    """ updates the duration after finding the duration """
    duration = vid_player.video_info()["duration"]
    end_time["text"] = str(datetime.timedelta(seconds=duration))
    progress_slider["to"] = duration


def update_scale(event):
    """ updates the scale value """
    progress_value.set(vid_player.current_duration())


def seek(value):
    """ used to seek a specific timeframe """
    vid_player.seek(int(value))


def skip(value: int):
    """ skip seconds """
    vid_player.seek(int(progress_slider.get())+value)
    progress_value.set(progress_slider.get() + value)


def play_pause():
    """ pauses and plays """
    if vid_player.is_paused():
        vid_player.play()
        play_pause_btn["text"] = "Pause"

    else:
        vid_player.pause()
        play_pause_btn["text"] = "Play"

def finish():
    x = label.winfo_x()
    y = label.winfo_y()
    time = vid_player.current_duration()
    print('coordinate = ',round(x/3),round(y/3))
    print('time = ',time)
    cap = cv2.VideoCapture(video_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES,time*30)
    ret,frame = cap.read()
    
    print('color= ',frame[round(y/3),round(x/3)])
    color_hls = cv2.BGR2HLS(frame[round(y/3), round(x/3)])
    table = Table(color_hls)
    four_points = FindTable(frame, table)
    print(four_points)
    if four_points != None:
        four_points[2],four_points[3] = four_points[3],four_points[2]
        points = np.array(four_points, np.int32)
        image = cv2.polylines(frame,pts=[points],isClosed=True,color=(255,0,255))
        cv2.imwrite('output.jpg', image)
    
def video_ended(event):
    """ handle video ended """
    progress_slider.set(progress_slider["to"])
    play_pause_btn["text"] = "Play"
    progress_slider.set(0)

def drag_start(event):
    widget = event.widget
    widget.startX = event.x
    widget.startY = event.y

def drag_motion(event):
    widget = event.widget
    x = widget.winfo_x() - widget.startX + event.x
    y = widget.winfo_y() - widget.startY + event.y
    widget.place(x=x,y=y)
    

root = tk.Tk()
root.title("Tkinter media")
root.geometry("1920x1184")
root.resizable(width=False,height=False)


vid_player = TkinterVideo(scaled=True, master=root)

vid_player.load(video_name)

label = tk.Label(root,bg="red",width=2,height=1,text='color')
label.place(x=255,y=255)


label.bind("<Button-1>",drag_start)
label.bind("<B1-Motion>",drag_motion)


vid_player.pack(expand=True, fill="both")

play_pause_btn = tk.Button(root, text="Play", command=play_pause)
play_pause_btn.pack()

finish_btn = tk.Button(root, text="update point", command=finish)
finish_btn.pack()

skip_plus_5sec = tk.Button(root, text="Skip -5 sec", command=lambda: skip(-5))
skip_plus_5sec.pack(side="left")

start_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
start_time.pack(side="left")

progress_value = tk.IntVar(root)

progress_slider = tk.Scale(root, variable=progress_value, from_=0, to=0, orient="horizontal", command=seek)

progress_slider.pack(side="left", fill="x", expand=True)

end_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
end_time.pack(side="left")

vid_player.bind("<<Duration>>", update_duration)
vid_player.bind("<<SecondChanged>>", update_scale)
vid_player.bind("<<Ended>>", video_ended )

skip_plus_5sec = tk.Button(root, text="Skip +5 sec", command=lambda: skip(5))
skip_plus_5sec.pack(side="left")

root.mainloop()