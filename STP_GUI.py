import tkinter.messagebox
import customtkinter
from tkinter import *
from PIL import Image , ImageTk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo , showerror
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import threading
from datetime import datetime
import time
import os
import librosa 
import numpy as np
import scipy.stats
import python_speech_features as mfcc
import audio_ML
import Image_functions
import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import *
from PIL import Image , ImageTk
import sys
import os

        
def select_photo(var):       
    if(var == "1"):
        filetypes = (('All files', '*.*'),('text files', '*.txt'))
        img1 = fd.askopenfile(title='Open a file',initialdir='/',filetypes=filetypes)
    
    elif(var == "2"):
        filetypes = (('All files', '*.*'),('text files', '*.txt'))
        img2 = fd.askopenfile(title='Open a file',initialdir='/',filetypes=filetypes)
    
    elif(var == "3"):
        filetypes = (('All files', '*.*'),('text files', '*.txt'))
        img3 = fd.askopenfile(title='Open a file',initialdir='/',filetypes=filetypes)
    
    elif(var == "4"):
        filetypes = (('All files', '*.*'),('text files', '*.txt'))
        img4 = fd.askopenfile(title='Open a file',initialdir='/',filetypes=filetypes)
    ans = Image_functions.imagawy(img1.name(),img2.name(),img3.name(),img4.name())
    return ans
        


def audio_feature(fileName):
        x, sr = librosa.load(fileName)
        freqs = np.fft.fftfreq(x.size)
        
        meanfun = np.mean(freqs)
        std = np.std(freqs) 
        median = np.median(freqs)
        q1 = np.quantile(freqs, 0.25)
        q3 = np.quantile(freqs, 0.75)
        iqr = scipy.stats.iqr(freqs)
        skew = scipy.stats.skew(freqs)
        kurt = scipy.stats.kurtosis(freqs)
        maxfun = np.amax(freqs) 
        minfun = np.amin(freqs) 
        mode = scipy.stats.mode(freqs)[0][0]

        features = [[meanfun,std,median,q1,q3,iqr,skew,kurt,maxfun,minfun,mode],[0,0,0,0,0,0,0,0,0,0,0]]
        
        ans = audio_ML.prediction(features)
        return ans 
        


customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Recognizer")
        self.iconbitmap('icon.ico')
        # self.attributes('-fullscreen',True)
        self.geometry(f"{1150}x{600}")
        
        # app_w =1080
        # app_h =580
        # screen_w = self.winfo_screenwidth()
        # screen_h = self.winfo_screenheight()
        # x=(screen_w / 2) - (app_w / 2)
        # y=(screen_h / 2) - (app_h / 2)
        
        # self.geometry(f"{app_w}x{app_h}+{int(x)}+{int(y)}")
        # self.eval('tk::PlaceWindow . center')
        # bg = PhotoImage(file="law1.png")
        # limg= Label(self, i=bg)
        
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
       
        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
       
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Recognizer", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
       
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,text="Face Recognition",command=self.sidebar_button1_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
       
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Voice Recognition",command=self.sidebar_button2_event)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Restart",fg_color="#FF5D5D",command=self.reset)
        self.sidebar_button_2.grid(row=9, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Exit",fg_color="#FF5D5D",command=self.destroy)
        self.sidebar_button_2.grid(row=10, column=0, padx=20, pady=10)
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=[ "Dark","Light", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
       
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button1_event(self):
        self.button1_frame = customtkinter.CTkFrame(self, width=820, corner_radius=0)
        self.button1_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.button1_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = customtkinter.CTkLabel(self.button1_frame, text="Face Recognition", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=1, padx=20, pady=(20, 10))

        self.ans1_frame = customtkinter.CTkFrame(self,width=140, corner_radius=0)
        self.ans1_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")
        self.ans1_frame.grid_rowconfigure(10, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.ans1_frame, text="Answer", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        

        open_button1 = customtkinter.CTkButton(self.button1_frame,text='Select the first picture',command=self.button_callback1)
        open_button1.grid(row=1, column=1, padx=20, pady=(20, 10))
        
        open_button2 = customtkinter.CTkButton(self.button1_frame,text='Select the second picture', command=self.button_callback2)
        open_button2.grid(row=1, column=2, padx=20, pady=(20, 10))
        
        open_button3 = customtkinter.CTkButton(self.button1_frame,text='Select the third picture', command=self.button_callback3)
        open_button3.grid(row=2, column=1, padx=20, pady=(20, 10))
        
        open_button4 = customtkinter.CTkButton(self.button1_frame,text='Select the prediction picture', command=self.button_callback4)
        open_button4.grid(row=2, column=2, padx=20, pady=(20, 10))

    def sidebar_button2_event(self):
        self.button1_frame = customtkinter.CTkFrame(self, width=820, corner_radius=0)
        self.button1_frame.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.button1_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = customtkinter.CTkLabel(self.button1_frame, text="Voice Recognition", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=1, padx=20, pady=(20, 10))


        self.ans1_frame = customtkinter.CTkFrame(self,width=140, corner_radius=0)
        self.ans1_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")
        self.ans1_frame.grid_rowconfigure(10, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.ans1_frame, text="Answer", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        open_button = customtkinter.CTkButton(self.button1_frame,text='Select an audio',command=self.select_audio)
        open_button.grid(row=1, column=1, padx=20, pady=(20, 10))
        # creating a ttk label
        duration_label = customtkinter.CTkLabel(self.button1_frame, text='OR', font=customtkinter.CTkFont(size=17, weight="bold"))
        duration_label.grid(row=2, column=1, padx=20, pady=(20, 10))

        duration_label = customtkinter.CTkLabel(self.button1_frame, text='Enter Recording Duration in Seconds:', font=customtkinter.CTkFont(size=13, weight="bold"))
        duration_label.grid(row=3, column=1, padx=0, pady=(0, 0))

        duration_entry = customtkinter.CTkEntry(self.button1_frame, width=76)
        duration_entry.grid(row=3, column=2, padx=0, pady=(0, 0))

        progress_label = customtkinter.CTkLabel(self.button1_frame, text='')
        progress_label.grid(row=3, column=1, padx=150, pady=(150, 75))

        # function for recording sound
        def record_voice():
            # the try statement is for 
            try:
                # this is the frequence at which the record will happen   
                freq = 44100
                # getting the recording duration from the entry
                duration = int(duration_entry.get())
                # calling the recorder via the rec() function
                recording  = sd.rec(duration*freq, samplerate=freq, channels=2)
                # declaring the counter
                counter = 0
                # the loop is for displaying the recording progress
                while counter < duration:
                    # updating the window
                    self.update()
                    # this will help update the window after every 1 second
                    time.sleep(1)
                    # incrementing the counter by 1
                    counter += 1
                    # displaying the recording duration
                    progress_label.configure(text=str(counter))

                # this records audio for the specified duration 
                sd.wait()

                # writing the audio data to recording.wav
                write('recording.wav', freq, recording)
                # looping through all the files in the current folder
                for file in os.listdir():
                    # checking if the file name is recording.wav
                    if file == 'recording.wav':
                        # spliting the base and the extension
                        base, ext = os.path.splitext(file)
                        # getting current time
                        current_time = datetime.now()
                        # creating a new name for the recorded file
                        new_name = 'recording'+ ext
                        # renaming the file
                        os.rename(file, new_name)
                # display a message when recording is done  
                showinfo('Recording complete', 'Your recording is complete')
            # function for catching all errors   
            except:
                # display a message when an error is caught
                showerror(title='Error', message='An error occurred' \
                        '\nThe following could ' \
                        'be the causes:\n->Bad duration value\n->An empty entry field\n' \
                        'Do not leave the entry empty and make sure to enter a valid duration value')
            ans = audio_feature(os.getcwd()+'\\recording.wav')
            if (ans==1):
                ans_label = customtkinter.CTkLabel(self.ans1_frame, text='MALE Voice', font=customtkinter.CTkFont(size=17, weight="bold"))
                ans_label.grid(row=2, column=0, padx=20, pady=(20, 10))
            else:
                ans_label = customtkinter.CTkLabel(self.ans1_frame, text='FEMALE Voice', font=customtkinter.CTkFont(size=17, weight="bold"))
                ans_label.grid(row=2, column=0, padx=20, pady=(20, 10))

        def recording_thread():
            # creating the thread whose target is record_voice()
            t1 = threading.Thread(target=record_voice)
            # starting the thread
            t1.start()
  
        record_button = customtkinter.CTkButton(self.button1_frame, text='Record', command=recording_thread)
        record_button.grid(row=3, column=1, padx=300, pady=(300, 150))
       

          
    def select_audio(self):
        filetypes = (('All files', '*.*'),('text files', '*.txt'))
        file = fd.askopenfile(title='Open a file',initialdir='/',filetypes=filetypes)
        ans = audio_feature(file.name)
        if (ans==1):
            ans_label = customtkinter.CTkLabel(self.ans1_frame, text='MALE Voice', font=customtkinter.CTkFont(size=17, weight="bold"))
            ans_label.grid(row=2, column=0, padx=20, pady=(20, 10))
        else:
            ans_label = customtkinter.CTkLabel(self.ans1_frame, text='FEMALE Voice', font=customtkinter.CTkFont(size=17, weight="bold"))
            ans_label.grid(row=2, column=0, padx=20, pady=(20, 10))
    
    
    def button_callback1(self): 
        select_photo('1')
    def button_callback2(self): 
        select_photo('2')
    def button_callback3(self): 
        select_photo('3')
    def button_callback4(self): 
        ans = select_photo('4')
        if (ans==1):
            ans_label = customtkinter.CTkLabel(self.ans1_frame, text='They are the same', font=customtkinter.CTkFont(size=17, weight="bold"))
            ans_label.grid(row=2, column=0, padx=20, pady=(20, 10))
        else:
            ans_label = customtkinter.CTkLabel(self.ans1_frame, text='They are not the same', font=customtkinter.CTkFont(size=17, weight="bold"))
            ans_label.grid(row=2, column=0, padx=20, pady=(20, 10))


    def reset(self):
        self.destroy()
        app = App()
        app.mainloop()

if __name__ == "__main__":
    app = App()
    app.mainloop()
