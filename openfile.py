import tkinter as tk
from tkinter import filedialog


def get_file():
    root = tk.Tk()
    root.withdraw()
    file_types = [('Image files', ('.png', '.jpg'))]
    filename = filedialog.askopenfilename(filetypes=file_types, title="Choose an Image File")
    return filename


def save_file():
    return filedialog.asksaveasfile(mode='w', defaultextension=".pkl")
