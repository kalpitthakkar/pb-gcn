from __future__ import print_function
import time

class Timer(object):
    def __init__(self,
                 print_log=True,
                 work_dir=''):
        self.curr_time = time.time()
        self.log = print_log
        self.work_dir = work_dir

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.log:
            with open('{}/log.txt'.format(self.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.curr_time = time.time()
        return self.curr_time

    def split_time(self):
        split_time = time.time() - self.curr_time
        self.record_time()
        return split_time