import json
import socket
import time


class SmplxFrameStreamer:
    def __init__(
        self,
        host="127.0.0.1",
        port=8765,
        connect_timeout=30.0,
        retry_interval=0.5,
    ):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.retry_interval = retry_interval
        self.sock = None
        self.writer = None

    def connect(self):
        deadline = None
        if self.connect_timeout is not None and self.connect_timeout > 0:
            deadline = time.monotonic() + self.connect_timeout

        last_error = None
        while True:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=1.0)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.writer = self.sock.makefile("w", encoding="utf-8")
                return
            except OSError as exc:
                last_error = exc
                self.close()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out connecting to SMPL-X stream receiver at {self.host}:{self.port}"
                    ) from last_error
                time.sleep(self.retry_interval)

    def send(self, payload):
        if self.writer is None:
            self.connect()

        message = json.dumps(payload, separators=(",", ":"))
        try:
            self.writer.write(message)
            self.writer.write("\n")
            self.writer.flush()
        except OSError:
            self.close()
            self.connect()
            self.writer.write(message)
            self.writer.write("\n")
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except OSError:
                pass
            self.writer = None
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
