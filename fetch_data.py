import numpy as np
import pandas as pd
import json


#loading in my data
class fetchdata:
    def __init__(self, path):
        self.path = path
        self.file =pd.read_csv(self.path)
