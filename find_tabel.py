import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo
import cv2

video_name = "test.mp4"

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
    print('coordinate = ',x,y)
    print('time = ',time)
    cap = cv2.VideoCapture(video_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES,time)
    ret,frame = cap.read()
    # cv2.imshow('My Image', frame)
    print('color= ',frame[y,x])
    # print('bot left = ',label2.winfo_x(),label2.winfo_y())
    # print('top right = ',label3.winfo_x(),label3.winfo_y())
    # print('bot right = ',label4.winfo_x(),label4.winfo_y())
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

vid_player = TkinterVideo(scaled=True, master=root)

vid_player.load(video_name)

label = tk.Label(root,bg="red",width=2,height=1,text='tl')
label.place(x=255,y=60)


# label2 = tk.Label(root,bg="red",width=2,height=1,text='bl')
# label2.place(x=160,y=380)

# label3 = tk.Label(root,bg="red",width=2,height=1,text='tr')
# label3.place(x=590,y=60)

# label4 = tk.Label(root,bg="red",width=2,height=1,text='br')
# label4.place(x=690,y=380)

label.bind("<Button-1>",drag_start)
label.bind("<B1-Motion>",drag_motion)

# label2.bind("<Button-1>",drag_start)
# label2.bind("<B1-Motion>",drag_motion)

# label3.bind("<Button-1>",drag_start)
# label3.bind("<B1-Motion>",drag_motion)

# label4.bind("<Button-1>",drag_start)
# label4.bind("<B1-Motion>",drag_motion)


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