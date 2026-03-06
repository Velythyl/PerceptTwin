import threading
import time

from tqdm import tqdm

def spinner(desc, stop_event):
    from itertools import cycle

    spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    with tqdm(total=None, desc=desc, bar_format='{desc}') as pbar:
        while not stop_event.is_set():
            pbar.set_description(f"{desc} - {next(spinner)}")
            time.sleep(0.1)
            #pbar.update()

def start_spinner_thread(desc) -> (threading.Event, threading.Thread):
    stop_event = threading.Event()
    spinner_thread = threading.Thread(
        target=spinner,
        args=(desc, stop_event)
    )
    spinner_thread.start()

    return stop_event, spinner_thread

def stop_spinner_thread(stop_event, spinner_thread=None):
    if isinstance(stop_event, tuple):
        stop_event, spinner_thread = stop_event
    stop_event.set()
    spinner_thread.join(timeout=5)