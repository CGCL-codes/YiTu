import threading

"""
线程安全队列
"""
class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def push(self, tensor):
        self.cv.acquire()
        self.queue.append(tensor)
        self.cv.notify()
        self.cv.release()

    def pop(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        tensor = self.queue.pop(0)
        self.cv.release()
        return tensor
