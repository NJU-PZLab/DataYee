# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 22:59:59 2021

@author: ZhengBin
"""
import time
def timer(function):
    def wrapper(*args, **kwargs):
        time_start = time.perf_counter()
        res = function(*args, **kwargs)
        cost_time = time.perf_counter() - time_start
        print("{}:{}s".format(function.__name__,cost_time*1e3))
        return res
    return wrapper
