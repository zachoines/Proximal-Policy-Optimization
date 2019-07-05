# Python core libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from collections import OrderedDict
from multiprocessing import Process, Queue, Lock
import threading
import time

# OpenAI Gym classes
import gym
from gym.core import RewardWrapper

# An action Wrapper class for environments
class Stats(RewardWrapper):
    def __init__(self, env, collector):
        super(Stats, self).__init__(env)
        self.numSteps = 0
        self.collector = collector
        self.CMA = 0

    def reset(self, **kwargs):
        if self.numSteps > 0:
            self.collector.collect('CMA', self.CMA)
        self.numSteps = 0
        self.CMA = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.numSteps += 1
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward), done, info

    def reward(self, reward):
        # Cumulative moving average
        self.CMA = (reward + (( self.numSteps - 1 ) * self.CMA )) / self.numSteps
        return reward

class AsynchronousPlot(threading.Thread):
    def __init__(self, collector, live = False):
        super(AsynchronousPlot, self).__init__()
        
        self._live = live
        self._lines = []
        self._axis = []
        self.collector = collector
        self.stoprequest = threading.Event()
        self.kill = threading.Event()
         
    def run(self):
        # style.use('fivethirtyeight')
        # Writer = animation.writers['html']
        # self.writer = Writer(fps=15, metadata=dict(artist='Zach Oines'))

        if self._live:
            self._fig = plt.figure(num = 0, figsize = (12, 8), dpi = 100)

            dims = self.collector.get_dimensions()
            
            if (len(dims) > 0):
                
                counter = 1
                for name in dims:

                    axis = self._fig.add_subplot(2,1, counter)
                    # line = matplotlib.lines.Line2D([], [], label = name)
                    line, = axis.plot([0], [0], label = name)
                    axis.set_xlabel('time')
                    axis.set_ylabel('data')
                    axis.add_line(line)
                    self._lines.append(line)
                    self._axis.append(axis)
                    counter += 1

            self._ani = animation.FuncAnimation(self._fig, self._update_graph, 
                interval = 1000, 
                blit = False, 
                save_count = 100,
                repeat=True)

            plt.show()
        
        else:
            while not self.kill.isSet():
                self._write_results()
                time.sleep(10)
    
    def continue_request(self):
        self.stoprequest.clear()
       
    def stop_request(self):
        self.stoprequest.set() 
    
    # if self._live:
    #         self._ani.save('Results.png', writer = "imagemagick", dpi = 80)
        
    #     # Signal the end of internal processes
    #     self.stoprequest.set()
    #     self.kill.set()
        
    def join(self, timeout=None):
        
        if self._live:
            self._ani.save('Results.png', writer = "imagemagick", dpi = 80)
        
        # Signal the end of internal processes
        self.stoprequest.set()
        self.kill.set()

        super(AsynchronousPlot, self).join(timeout)

    def _draw_final_graph(self):
        pass

    def _write_results(self):
        if not self.stoprequest.isSet():
            
            if (len(self.collector.dimensions) > 0):
                
                try:
                    with open("stats.txt", 'w') as f:

                        for name in self.collector.dimensions:

                            data = self.collector.get_dimension(name)

                            if  data == None or len(data) == 0:
                                continue

                            f.write(name + ":\n")
                            
                            for x, y in data.items():
                                f.write("\t" + str(x) + ", " + str(y) + "\n")  
                
                except:
                    
                    print("Issue writing results to 'stats.txt'.")
                    raise
                
                finally:
                    
                    f.close()

    def _unpack_args(self, *args):
        return args
    
    def _update_graph(self, i):
        
        if not self.stoprequest.isSet():
            
            if (len(self.collector.dimensions) > 0):
               
                for name in self.collector.dimensions:

                    data = self.collector.get_dimension(name)

                    if data == None or len(data) == 0:
                        continue

                  

                    for i in range(len(self._lines)):
                        line = self._lines[i]
                        label = line.get_label()
                        
                        if label == name:
                            xs = []
                            ys = []
                            for x, y in data.items():
                                xs.append(x)
                                ys.append(y)
                            
                            self._axis[i].clear()
                            line, = self._axis[i].plot(xs, ys)
                            self._lines[i] = line
                            line.set_label(name)
                            # line.set_data(x, y)
            

                return self._unpack_args(*self._lines)

            else:

                # Not ready to show anything
                return self._unpack_args(*self._lines)

        else: 
            return self._unpack_args(*self._lines)
        

# Data structor for collecting multiple 2d dimentions
class Collector:
    def __init__(self):
        # datastructure is this: [( Data_name : { x_0 : y_0, x_1 : y_1, ....}), ....]
        self.current_milli_time = lambda: int(round(time.time()))
        self.tstart = self.current_milli_time()
        self.dimensions = []
        self._data = []
        self._init = False

    def collect(self, name, data):
        
        current_time = self._current_run_time()

        if (name not in self.dimensions):
            self.dimensions.append(name)
            entry = (name, OrderedDict([(current_time, data)]))
            self._data.append(entry)

        elif self.get_dimension(name) == None:
            entry = (name, OrderedDict([]))
            self._data.append(entry)
        else:
            
            dim_values = None
            
            try:
                ( _ , dim_values) = next((value for key, value in enumerate(self._data) if value[0] == name), None)
                dim_values[current_time] = data 
            except GeneratorExit:
                pass


    def set_dimensions(self, dims):
        self.dimensions = dims
                
    def get_dimensions(self):
        return self.dimensions


    def get_dimension(self, name):

        if self._data == []:
            for n in self.dimensions:
                entry = (n, OrderedDict([]))
                self._data.append(entry)
            return None

        dim_values = None
        try:
            ( _ , dim_values) = next((value for key, value in enumerate(self._data) if value[0] == name), None)
        except GeneratorExit:
            pass
            

        return dim_values
            
    def remove(self, name):
        if name in self.dimensions:
            try:
                index = next((key for key, value in enumerate(self._data) if value[0] == name), None)
                return self._data.pop(index)
            except GeneratorExit:
                pass
            
            return None

    def _current_run_time(self):
        return (self.current_milli_time() - self.tstart)
  
  

    
        


        