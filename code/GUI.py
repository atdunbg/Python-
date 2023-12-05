import tkinter as tk
from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk


from main import *



class DisplayImage:
    '''用于展示选择的图片'''
    def __init__(self, master):
        self.master = master
        master.title("GUI")
        self.image_frame = Frame(master, bd=0, height=250, width=250, bg='black', highlightthickness=2,
                                 highlightbackground='gray', highlightcolor='black')
        self.image_frame.pack()

        self.Text_label = Label(master, text='图像预览')
        self.Text_label.pack()

        # 显示输出数字
        self.layout1 = Label(master, text='数字输出将显示在这里',relief=GROOVE, pady=5, width=20)
        self.layout1.pack()

        self.Choose_image = Button(master, command=self.display_image, text="选择图片",width=17, default=ACTIVE)
        self.Choose_image.pack()


        
        self.filenames = []
        self.pic_filelist = []
        self.imgt_list = []
        self.image_labellist = []

    def display_image(self, event=None):
        #在重新选择图片时清空原先列表
        self.pic_filelist.clear()
        self.imgt_list.clear()
        self.image_labellist.clear()

        self.filenames.clear()
        self.filenames += filedialog.askopenfilenames()
        #清空框架中的内容
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        #布局所选图片
        for i in range(len(self.filenames)):
            self.pic_filelist.append(Image.open(self.filenames[i]))
            self.imgt_list.append(ImageTk.PhotoImage(image=self.pic_filelist[i]))
            self.image_labellist.append(Label(self.image_frame, highlightthickness=0, borderwidth=0))
            self.image_labellist[i].configure(image=self.imgt_list[i])
            self.image_labellist[i].pack(side=LEFT, expand=True)
        # print(self.filenames[0])
        
        #进行图片识别

        self.Text_label.config(text=self.filenames[0])
        self.layout1.config(text=demo(self.filenames[0]))



def main():
    window = tk.Tk()
    GUI = DisplayImage(window)
    window.title('数字识别GUI')
    window.geometry('600x400')
    window.mainloop()


if __name__ == '__main__':
    main()
