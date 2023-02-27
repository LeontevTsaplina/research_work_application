from tkinter import *
import tkinter as tk
import threading
from tkinter.ttk import Style
from tkinter.ttk import Combobox
from tkinter import messagebox
from os import listdir
from os.path import isfile, join
from contextlib import contextmanager
from PIL import Image, ImageTk

from DQN import dqn
from Dataset import Dataset


class MyApp(tk.Tk):

    def __init__(self, width, height, title, files_path):
        tk.Tk.__init__(self)

        self.__files_path = files_path

        self.geometry(f'{width}x{height}')
        self.resizable(False, False)
        self.title(title)
        self.iconbitmap("Images/icon.ico")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.__frame = None
        self.switch_frame(ChooseFilePage)

    def switch_frame(self, frame_class, chosen_file=None, agent=None):
        if self.__frame is not None:
            self.__frame.destroy()

        if frame_class == ChooseFilePage:
            new_frame = frame_class(self, self.__files_path)
        elif frame_class == Preloader:
            try:
                new_frame = frame_class(self, chosen_file)
            except:
                messagebox.showinfo("Notification", "Not valid file. Try another one")
                new_frame = ChooseFilePage(self, self.__files_path)
        elif frame_class == PredictorPage:
            new_frame = frame_class(self, agent)
        else:
            new_frame = frame_class(self)

        self.__frame = new_frame
        self.__frame.pack(expand=True)

    def get_file_path(self):

        return self.__files_path


class ChooseFilePage(tk.Frame):

    def __init__(self, parent, files_path):
        tk.Frame.__init__(self, parent)

        self.__files_path = files_path

        group = LabelFrame(self, borderwidth=0)
        group.pack()

        Label(self, text="Choose file with COVID-19 patients data", font=("Arial Bold", 30)).pack()
        Label(self, text=f'(File must be in path: "{files_path}" and be .pkl)', font=("Arial", 10, 'italic')).pack()

        combo = Combobox(self, width=25, state='readonly', background=self.cget('background'),
                         font=("Arial", 13),
                         values=[f for f in listdir(self.__files_path) if
                                 isfile(join(self.__files_path, f)) and f.endswith('.pkl')])

        button = Button(self, text="Choose", width=20, height=1, background='#ff5858', font=("Arial Bold", 15),
                        fg='white', command=lambda: parent.switch_frame(Preloader, chosen_file=combo.get()))

        if len(listdir(self.__files_path)) > 0:
            combo.current(0)
        else:
            button.configure(state='disabled')

        combo.pack(pady=20)
        button.pack()


class Preloader(tk.Frame):

    def __init__(self, parent, chosen_file):
        tk.Frame.__init__(self, parent)

        self.__parent = parent

        self.__chosen_file = chosen_file

        self.__dataset = Dataset(f'{parent.get_file_path()}/{chosen_file}')

        Label(self, text='Loading dataset and training agent...', font=("Arial Bold", 15)).pack(
            fill=tk.BOTH, side=tk.TOP, expand=True, pady=10)

        self.pack(expand=True)

        self.wait_visibility()
        self.update()

        self.__loader = Image.open('Images/loading.gif')
        self.__frames = []

        for frame in range(0, self.__loader.n_frames):
            self.__loader.seek(frame)
            frame_image = self.__loader.convert("RGBA")
            self.__frames.append(ImageTk.PhotoImage(frame_image))

        self.__animator = Label(self, image=self.__frames[0], bg=self.cget("background"))
        self.__animator.pack()

        self.current_frame = 0
        self.animate()

        self.__progress = Label(self, text=f'Episode 1 from {self.__dataset.episodes_count()}')
        self.__progress.pack(pady=30)

        self.thread = threading.Thread(target=self.context_call)
        self.thread.start()

    def context_call(self):
        with self.start_dqn() as context:
            self.destroy()
            self.__parent.switch_frame(PredictorPage, agent=context)

    @contextmanager
    def start_dqn(self):
        gen = dqn(self.__dataset)
        all_iters = self.__dataset.episodes_count()
        while True:
            try:
                value = next(gen)
                self.__progress.configure(text=f'Episode {value} from {all_iters}')
            except StopIteration as e:
                result = e.value
                break
        yield result

    def animate(self):
        self.current_frame += 1
        if self.current_frame == len(self.__frames):
            self.current_frame = 0
        try:
            self.__animator.configure(image=self.__frames[self.current_frame])
        except TclError:
            pass
        self.__parent.after(50, self.animate)


class PredictorPage(tk.Frame):

    def __init__(self, parent, agent):
        tk.Frame.__init__(self, parent)

        self.__agent = agent
        self.__dataset = agent.return_dataset()

        self.__stat_control = [column.replace('_stat_control', '') for column in self.__dataset.columns if
                               column.endswith('_stat_control')]
        self.__stat_fact = [column.replace('_stat_fact', '') for column in self.__dataset.columns if
                            column.endswith('_stat_fact')]
        self.__dinam_fact = [column.replace('_dinam_fact', '') for column in self.__dataset.columns if
                             column.endswith('_dinam_fact')]

        self.__stat_control_group = LabelFrame(self, text='Stat control parameters')
        self.__stat_control_group.pack(anchor=tk.W, pady=10)

        self.__stat_fact_group = LabelFrame(self, text='Stat fact parameters')
        self.__stat_fact_group.pack(anchor=tk.W, pady=10)

        self.__dinam_fact_group = LabelFrame(self, text='Dinam fact parameters')
        self.__dinam_fact_group.pack(anchor=tk.W, pady=10)

        for index, st_control in enumerate(self.__stat_control):
            group = LabelFrame(self.__stat_control_group, borderwidth=0)
            group.grid(column=index % 7, row=index // 7, padx=10, pady=10)

            Label(group, text=st_control, font=("Arial Bold", 8)).pack(anchor=tk.NW, expand=True)

            combo = Combobox(group, values=['Yes', 'No'], state='readonly',)
            combo.current(1)
            combo.pack(side=tk.LEFT)

        for index, st_fact in enumerate(self.__stat_fact):
            group = LabelFrame(self.__stat_fact_group, borderwidth=0)
            group.grid(column=index % 7, row=index // 7, padx=10, pady=10)

            Label(group, text=st_fact, font=("Arial Bold", 8)).pack(anchor=tk.NW, expand=True)

            Entry(group, validate='key', validatecommand=(self.register(self.validate_input), "%P")).pack(side=tk.LEFT)

        for index, d_fact in enumerate(self.__dinam_fact):
            group = LabelFrame(self.__dinam_fact_group, borderwidth=0)
            group.grid(column=index % 7, row=index // 7, padx=10, pady=10)

            Label(group, text=d_fact, font=("Arial Bold", 8)).pack(anchor=tk.NW, expand=True)

            Entry(group, validate='key', validatecommand=(self.register(self.validate_input), "%P")).pack(side=tk.LEFT)

        Button(self, text="Predict", width=20, height=1, background='#ff5858', font=("Arial Bold", 15),
               fg='white', command=lambda: self.predict()).pack(pady=10)

        self.__result = ''
        self.__result_label = Text(self, height=1, width=50, font=("Arial Bold", 15), background=self.cget('background'))
        self.__result_label.pack()

    def predict(self):
        labelframes = [labelframe for labelframe in self.__stat_control_group.winfo_children() if
                       isinstance(labelframe, LabelFrame)] + \
                      [labelframe for labelframe in self.__stat_fact_group.winfo_children() if
                       isinstance(labelframe, LabelFrame)] + \
                      [labelframe for labelframe in self.__dinam_fact_group.winfo_children() if
                       isinstance(labelframe, LabelFrame)]
        labelframes = [labelframe.winfo_children() for labelframe in labelframes]
        labelframes = [item for sublist in labelframes for item in sublist]

        labels = [widget.get() for widget in labelframes if isinstance(widget, Combobox) or isinstance(widget, Entry)]

        if '' in labels:
            messagebox.showinfo("Notification", "You need to fill all inputs")
        else:
            state = []

            for elem in labels:
                if elem == 'Yes':
                    state.append(1)
                elif elem == 'No':
                    state.append(0)
                else:
                    state.append(float(elem))

            self.__result = self.__agent.return_predict(state)

            self.__result_label.delete('1.0', 'end')

            self.__result_label.insert(tk.END, 'The best treatment for patient is: ', 'black')
            self.__result_label.insert(tk.END, f'{self.__result}', 'red')

            self.__result_label.tag_configure('black', foreground='#000000', justify="center")
            self.__result_label.tag_configure('red', foreground='#ff5858', justify="center")

            self.update()

    @staticmethod
    def validate_input(input_value):
        if input_value == "":
            return True

        for char in input_value:
            if not char.isdigit() and char != ".":
                return False

        if input_value.count(".") > 1:
            return False

        return True
