from src.loadjpk import zipfileopera
import numpy as np
class pick_dlc():
    def __init__(self,zpo,ljp,fc):
        self.zpo = zpo
        self.ljp = ljp
        self.fc = fc
        self.zpo1 = zipfileopera('1.DataYee-force')
    def run(self,index):
        self.fc.data = self.zpo[index]
        dlc = self.fc.data['dlc']
        if len(np.where((dlc>14)&(dlc<25))[0])>0 and  len(np.where((dlc>45)&(dlc<70))[0])>0 and len(dlc)<5 and len(dlc)>=3:
            self.zpo1.changingforce(self.fc)
            print('ok')
        else:
            self.fc.data['artificial_judge'] = False
        self.zpo.changingforce(self.fc)
    def end(self):
        self.zpo1.changedforce()
        pass