import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
filename = filedialog.askopenfilename(filetypes=(("jpeg, png files", "*.jpg, *.png"), ("all files", "*.*")),
                                      title="Choose an Image File")
