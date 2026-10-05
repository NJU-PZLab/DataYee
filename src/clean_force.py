from src.loadjpk import zipfileopera
import numpy as np
class pick_dlc():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        self.fc.data = self.zpo[index]
        if not self.fc.data['artificial_judge']:
            self.zpo.deletingforce(self.fc)
    def end(self):
        self.zpo1.deletedforce()
        pass