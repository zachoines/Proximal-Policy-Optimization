# Python core libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from collections import OrderedDict
from multiprocessing import Process, Lock
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
        self._busy = threading.Event()
        self._stoprequest = threading.Event()
        self._kill = threading.Event()
         
    def run(self):
        # style.use('fivethirtyeight')
        # Writer = animation.writers['html']
        # self.writer = Writer(fps=15, metadata=dict(artist='Zach Oines'))

        if self._live:
            self._fig = plt.figure(num = 0, figsize = (10, 6), dpi = 100)

            dims = self.collector.get_dimensions()
            
            if (len(dims) > 0):
                
                counter = 1
                for name in dims:

                    axis = self._fig.add_subplot(2,1, counter)
                    axis.autoscale_view()
                    # axis.set_title(name)
                    # line = matplotlib.lines.Line2D([0], [0], label = name)
                    line, = axis.plot([], [], label = name)
                    axis.set_xlabel('time')
                    axis.set_ylabel(name)
                    axis.add_line(line)
                    self._lines.append(line)
                    self._axis.append(axis)
                    counter += 1

            self._ani = animation.FuncAnimation(self._fig, self._update_graph, 
                interval = 50, 
                blit = True, 
                save_count = 100,
                repeat=True)

            plt.show()
        
        else:
            while not self._kill.isSet():
                self._write_results()
                time.sleep(10)
    
    def continue_request(self):
        self._stoprequest.clear()
       
    def stop_request(self):
        self._stoprequest.set() 
        
    def join(self, timeout=None):
        
        if self._live:
            self._ani.save('Results.png', writer = "imagemagick", dpi = 80)
        
        # Signal the end of internal processes
        self._stoprequest.set()
        self._kill.set()

        super(AsynchronousPlot, self).join(timeout)

    def _draw_final_graph(self):
        pass

    def _write_results(self):

        if not self._stoprequest.isSet():
            
            if (len(self.collector.dimensions) > 0):
                
                try:
    
                    for name in self.collector.dimensions:
                        
                        with open(".\stats\\" + name + ".txt", 'a') as f:
                            data = self.collector.get_dimension(name)

                            if  data == None or len(data) == 0:
                                continue

                            while len(data) > 0:
                                (x, y) = data.popitem(last = True)
                                f.write(str(x) + ", " + str(y) + "\n")  
                
                except:
                    
                    f.close()
                    print("Issue writing results.")
                    raise
                

    def _update_graph(self, i):
        
        if not self._stoprequest.isSet():
            
            if (len(self.collector.dimensions) > 0):
                
                for name in self.collector.dimensions:

                    data = self.collector.pop(name)

                    if data == None or len(data) == 0:
                        continue

                    (x, y) = data

                    for i in range(len(self._lines)):
                        line = self._lines[i]

                        label = line.get_label()
                        
                        if label == name:
                            
                            (xs, ys) = line.get_data()
                            xs = np.append(xs, [x], 0)
                            ys = np.append(ys, [y], 0)

                            line.set_xdata(xs)
                            line.set_ydata(ys)
                            
                            self._axis[i].set_xlim(min(xs), max(xs))
                            self._axis[i].set_ylim(min(ys), max(ys))
                            
                            return self._lines
                
                return self._lines

            else:

                # Not ready to show anything
                return self._lines 

        else: 
            return self._lines
        

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
    
    def pop(self, name):
        data = self.get_dimension(name)
        
        if data == None or len(data) == 0:
            return None

        
        return data.popitem(last = True)

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
  
  

    
        


        