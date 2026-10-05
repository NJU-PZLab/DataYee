from src.loadjpk import zipfileopera
import numpy as np
class clean_force():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
    def run(self,index):
        self.fc.data = self.zpo[index]
        print("{}/{}".format(index,len(self.zpo)))
        if not self.fc.data['artificial_judge'] or self.fc.data['class']=='F':
            self.zpo.deletingforce(self.fc)
        self.fc.clean_force()
    def end(self):
        self.zpo.deletedforce()
        pass