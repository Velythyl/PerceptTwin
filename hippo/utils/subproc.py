import dataclasses
import select
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from io import StringIO
import queue
from typing import List, Union, Any


@dataclasses.dataclass
class SubprocessResult:
    cmd: Union[str, List[str]]
    returncode: int
    stdout: str
    stderr: str
    stdout_stderr: str
    process: Any = None

    @property
    def success(self):
        return self.returncode == 0

    @property
    def failure(self):
        return not self.success


def run_subproc(cmd, callback=None, shell=False, timeout=None, timeout_cleanup_func=None, raise_timeout_exception=False, immediately_return=False):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        text=True,
        universal_newlines=True,
        bufsize=1,  # line-buffered
        executable='/bin/bash'
    )

    stdout_buf = StringIO()
    stderr_buf = StringIO()
    combined_buf = StringIO()
    print_queue = queue.Queue()
    callback_queue = queue.Queue()
    lock = threading.Lock()

    if callback is None:
        class NullContext:
            def __enter__(self):
                pass  # Do nothing on enter

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass  # Do nothing on exit
        callback_lock = NullContext()
    else:
        callback_lock = threading.Lock()

    def reader(stream, buf, label, stop_event):
        while not stop_event.is_set():
            r, _, _ = select.select([stream], [], [], 0.1)
            if r:
                line = stream.readline()
                if not line:  # EOF
                    break
                with lock:
                    buf.write(line)
                    combined_buf.write(line)
                print_queue.put((label, line))
                callback_queue.put((label, line))
            if stop_event.is_set():
                break
        stream.close()

    class STOPPRINT:
        pass

    def printer():
        while True:
            line = print_queue.get()

            if line is None:
                time.sleep(0.01)
                continue
            elif isinstance(line, STOPPRINT):
                break
            else:
                print(line[1], end='', file=sys.stderr if line[0] == 'stderr' else sys.stdout)

    def callbacker():
        while True:
            time.sleep(1)
            line = callback_queue.get()
            if line is None:
                time.sleep(0.01)
                continue
            elif isinstance(line, STOPPRINT):
                break
            else:
                partial_result = SubprocessResult(
                    returncode=None,
                    stdout=combined_buf.getvalue(),
                    stderr=stderr_buf.getvalue(),
                    stdout_stderr=combined_buf.getvalue(),
                )
                callback(partial_result)

    stdout_event = threading.Event()
    stdout_thread = threading.Thread(target=reader, args=(process.stdout, stdout_buf, 'stdout', stdout_event))
    stdout_thread.daemon = True
    stderr_event = threading.Event()
    stderr_thread = threading.Thread(target=reader, args=(process.stderr, stderr_buf, 'stderr', stderr_event))
    stdout_thread.daemon = True
    printer_thread = threading.Thread(target=printer)
    printer_thread.daemon = True

    stdout_thread.start()
    stderr_thread.start()
    printer_thread.start()
    if callback is not None:
        assert immediately_return is False
        callback_thread = threading.Thread(target=callbacker)
        callback_thread.daemon = True
        callback_thread.start()

    if immediately_return:
        return SubprocessResult(
            cmd=cmd,
            returncode=None,
            stdout=stdout_buf,
            stderr=stderr_buf,
            stdout_stderr=combined_buf,
            process=process
        )

    TIMED_OUT = False
    TIMEOUT_EXCEPTION = None
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired as timeout_exception:
        TIMEOUT_EXCEPTION = timeout_exception
        process.terminate()

        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()
            time.sleep(5)

        if timeout_cleanup_func is not None:
            timeout_cleanup_func()

        TIMED_OUT = True
        stdout_event.set()
        stderr_event.set()

    stdout_thread.join()
    stderr_thread.join()

    # Signal printer to stop
    print_queue.put(STOPPRINT())
    printer_thread.join()
    if callback is not None:
        callback_queue.put(STOPPRINT())
        callback_thread.join()

    result = SubprocessResult(
        cmd=cmd,
        returncode=process.returncode,
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        stdout_stderr=combined_buf.getvalue()
    )

    if callback:
        callback(result)

    if raise_timeout_exception and TIMED_OUT:
        raise TIMEOUT_EXCEPTION

    return result

if __name__ == "__main__":
    run_subproc("which python && ejcka", lambda result: print(result), shell=True)
