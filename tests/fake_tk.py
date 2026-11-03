"""In-memory stand-in for the Tk modules the optional window uses."""

import time
import types

SETTLE_SECONDS = 10
NORMAL = "normal"


class FakeScreen:
    """Records widgets and dialogs; answers file dialogs from queued choices."""

    def __init__(self):
        self.widgets = []
        self.shown = []
        self.chosen_files = []
        self.chosen_directories = []
        self.window = None

    def modules(self) -> dict:
        tkinter = types.ModuleType("tkinter")
        ttk = types.ModuleType("tkinter.ttk")
        filedialog = types.ModuleType("tkinter.filedialog")
        messagebox = types.ModuleType("tkinter.messagebox")
        widget = self._widget_class()
        tkinter.Tk = self._window_class()
        tkinter.StringVar = _variable_class("")
        tkinter.BooleanVar = _variable_class(False)
        tkinter.Text = type("Text", (widget,), {})
        for name in ("Frame", "Label", "Entry", "Button", "Checkbutton", "Notebook"):
            setattr(ttk, name, type(name, (widget,), {}))
        filedialog.askopenfilename = lambda **_: self.chosen_files.pop(0)
        filedialog.askdirectory = lambda **_: self.chosen_directories.pop(0)
        messagebox.showerror = lambda title, text: self.shown.append(
            ("error", title, text)
        )
        messagebox.showinfo = lambda title, text: self.shown.append(
            ("info", title, text)
        )
        tkinter.ttk, tkinter.filedialog, tkinter.messagebox = (
            ttk,
            filedialog,
            messagebox,
        )
        return {
            "tkinter": tkinter,
            "tkinter.ttk": ttk,
            "tkinter.filedialog": filedialog,
            "tkinter.messagebox": messagebox,
        }

    def button(self, text):
        return self._widget("Button", text=text)

    def press(self, text):
        self.button(text).options["command"]()

    def entries(self):
        return [w.options["textvariable"] for w in self._all("Entry")]

    def option(self, text):
        return self._widget("Checkbutton", text=text).options["variable"]

    def status(self):
        labels = [w for w in self._all("Label") if "textvariable" in w.options]
        return labels[0].options["textvariable"].get()

    def view(self, label):
        return self._all("Notebook")[0].tabs[label].content

    def busy(self):
        return any(
            b.options.get("state", NORMAL) != NORMAL for b in self._all("Button")
        )

    def tick(self):
        self.window.scheduled()

    def settle(self):
        """Poll the window until every job has finished."""
        deadline = time.monotonic() + SETTLE_SECONDS
        while self.busy():
            if time.monotonic() > deadline:
                raise TimeoutError("The window job did not finish.")
            time.sleep(0.01)
            self.tick()

    def _widget(self, kind, text):
        return next(w for w in self._all(kind) if w.options.get("text") == text)

    def _all(self, kind):
        return [w for w in self.widgets if type(w).__name__ == kind]

    def _widget_class(self):
        screen = self

        class Widget:
            def __init__(self, parent=None, **options):
                self.options = options
                self.content = ""
                self.tabs = {}
                screen.widgets.append(self)

            def pack(self, **_):
                pass

            def config(self, **options):
                self.options.update(options)

            def add(self, child, text):
                self.tabs[text] = child

            def delete(self, start, end):
                self.content = ""

            def insert(self, index, text):
                self.content = text + self.content

        return Widget

    def _window_class(self):
        screen = self

        class Window:
            def __init__(self):
                self.handlers = {}
                self.scheduled = None
                self.destroyed = False
                screen.window = self

            def title(self, text):
                pass

            def geometry(self, size):
                pass

            def protocol(self, name, handler):
                self.handlers[name] = handler

            def after(self, milliseconds, callback):
                self.scheduled = callback

            def mainloop(self):
                pass

            def destroy(self):
                self.destroyed = True

        return Window


def _variable_class(initial):
    class Variable:
        def __init__(self, value=initial):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    return Variable
