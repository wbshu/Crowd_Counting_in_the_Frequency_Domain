'''
This python file possess all kinds of metrics, measures, counters, and other data observers.
'''

import time

class Metric:
    def reset(self):
        raise NotImplementedError

class Average_Metric(Metric):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.sum += val * num
        self.count += num

    def get_avg(self):
        if self.count != 0:
            return self.sum / self.count
        else:
            return -1

    def get_count(self):
        return self.count

    def reset(self):
        self.sum = 0
        self.count = 0

class Time_Metric(Metric):
    def __init__(self):
        self.time = time.time()

    def get_time_consumption(self):
        return time.time() - self.time

    def reset(self):
        self.time = time.time()
