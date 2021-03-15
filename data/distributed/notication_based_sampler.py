import zmq
import multiprocessing


class Response:
    def __init__(self):
        self.status_code = 200
        self.body = ''

    def set_status_code(self, code):
        self.status_code = code

    def set_body(self, body):
        self.body = body


def _server_entry(socket_address: str, callback):
    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(socket_address)

    while True:
        command = socket.recv_pyobj()
        if command == 'hello':
            socket.send_pyobj((200,))
        elif command == 'shutdown':
            socket.close()
            return
        else:
            response = Response()
            callback(command, response)
            socket.send_pyobj((response.status_code, response.body))


class ServerLauncher:
    def __init__(self, socket_address: str, callback):
        self.socket_address = socket_address
        self.callback = callback
        self.process = None

    def launch(self, wait_for_ready=True):
        self.process = multiprocessing.Process(target=_server_entry, args=(self.socket_address, self.callback))
        self.process.start()
        if wait_for_ready:
            socket = zmq.Context.instance().socket(zmq.REQ)
            socket.connect(self.socket_address)
            socket.send_pyobj('hello')
            recv = socket.recv_pyobj()
            socket.close()
            if recv != 200:
                self.process.kill()
                raise Exception('Unexpected value')

    def stop(self, wait_for_stop=False):
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.connect(self.socket_address)
        socket.send_pyobj('shutdown')
        socket.close()
        if wait_for_stop:
            self.process.join()


class Client:
    def __init__(self, socket_address: str):
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.connect(socket_address)

    def __call__(self, *args):
        self.socket.send_pyobj(args)
        response = self.socket.recv_pyobj()
        if response[0] != 200:
            raise RuntimeError(f'remote procedure failed with code {response[0]}')
        else:
            return response[1]
