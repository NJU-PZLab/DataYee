import numpy as np
#import operator
#from sklearn import svm
#import os
#from PIL import Image
import torch
#from sklearn.model_selection import train_test_split,StratifiedKFold
import torchvision.transforms as transforms
#from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,VotingClassifier
#from scipy import stats
#import itertools
#from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
#import toolz
#import dask
import torch.nn as nn
import pickle
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader
from datayee.resources import asset_path
 
'''
def loaddata(train,value,imgsize=8):
    img_lst = []
    data_lst = []
    val_target_array = np.array([])
    val_img_lst = []
    val_data_lst = []
    target_array = np.array([])
    for a,b,c in os.walk(train):
        target = os.path.basename(a)
        if target.isdigit():
            for file in c:
                if file.endswith('.jpg'):
                    fname = os.path.join(a,file)
                    target_array = np.append(target_array,int(target))
                    img = Image.open(fname)
                    img_lst.append(img)
                    img = img.resize((imgsize,imgsize))
                    img = img.convert('L')
                    data_lst.append(np.array(img).reshape(-1))
    data_array = np.array(data_lst)
    for a,b,c in os.walk(value):
        target = os.path.basename(a)
        if target.isdigit():
            for file in c:
                if file.endswith('.jpg'):
                    fname = os.path.join(a,file)
                    val_target_array = np.append(val_target_array,int(target))
                    img = Image.open(fname)
                    val_img_lst.append(img)
                    img = img.resize((imgsize,imgsize))
                    img = img.convert('L')
                    val_data_lst.append(np.array(img).reshape(-1))
    val_data_array = np.array(val_data_lst)
    return img_lst,data_array,target_array,val_img_lst,val_data_array,val_target_array
class kmeans:
    def __init__(self,train,value):
        self.train = train
        self.value = value
        pass
    def classfy(self,x,data,target,k=10):
        dataSetSize = data.shape[0]
        diffMat = np.tile(x, (dataSetSize, 1)) - data
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(k):
            voteIlabel = target[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
    def fit(self):
        self.trainset,self.traintarget = loaddata(self.train,self.value)[1:3]
    def predict(self,x):
        return np.array([self.classfy(i,self.trainset, self.traintarget) for i in x])

b, a = signal.butter(8, 0.08, 'lowpass')
def feature_extract(fc):
    data = fc.get_prodata(tip_correc=False)['retract']
    data_y = data['vDeflection']*1e12
    d = np.diff(data_y.reshape(-1),prepend=data_y[0])
    p,_ = signal.find_peaks(gaussian_filter(data_y.reshape(-1),11),height=20,prominence=10,width=10)
    data_y[np.delete(np.arange(len(data_y)),p)] = 0
    data_y = data_y/data_y.max()
    d = signal.filtfilt(b, a, d)
    d = d/np.abs(d).max()
    d[np.where(d>-0.11)] = 0
    fig,ax = plt.subplots(figsize=(2.24, 2.24))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.1)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.plot(data_y, color='#364fc7', linewidth=1)
    ax.plot(d , '#c92a2a', linewidth=1)
    plt.close()
    return fig
'''
class VotingClassify:
    def __init__(self,modeldir='./model/voting_clf_20210514_acc0.80_svm_lr_rf.model'):
        self.modeldir = modeldir
        self.loadmodel()
    def loadmodel(self):
        with open(self.modeldir,'rb') as f:
            self.model = pickle.load(f)
    def predict_batchs(self,data_array):
        return self.model.predict(data_array)
b, a = signal.butter(8, 0.08, 'lowpass')
def feature_extract(fc,get_data=False,data_len = 250):
    #methods=1.0
    data = fc.get_prodata(tip_correc=False)['retract']
    data_y = data['vDeflection']*1e12
    d = np.diff(data_y.reshape(-1),prepend=data_y[0])
    p,_ = signal.find_peaks(gaussian_filter(data_y.reshape(-1),11),height=20,prominence=10,width=10)
    data_y[np.delete(np.arange(len(data_y)),p)] = 0
    data_y = data_y/data_y.max()
    d = signal.filtfilt(b, a, d)
    d = d/np.abs(d).max()
    d[np.where(d>-0.11)] = 0
    if get_data:
        d = np.abs(d)
        pk = find_peaks(d)[0]
        new_d = np.zeros(len(d))
        new_d[pk] = d[pk]
        d = new_d
        data_y = data_y.reshape(-1)
        zoom_d = (np.where(d>0)[0]/len(d)*data_len).astype(np.int16)
        zoom_datay = (np.where(data_y>0)[0]/len(data_y)*data_len).astype(np.int16)
        new_d,new_datay = np.zeros(data_len),np.zeros(data_len)
        new_d[zoom_d],new_datay[zoom_datay] = d[np.where(d>0)[0]],data_y[np.where(data_y>0)[0]]
        return new_d.reshape(-1),new_datay.reshape(-1)
    fig,ax = plt.subplots(figsize=(2.24, 2.24))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0.1, right=1, left=0.1)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ax.plot(data_y, color='#364fc7', linewidth=1)
    ax.plot(d , '#c92a2a', linewidth=1)
    plt.close()
    return fig

# 兼容旧版 torchvision 模型 (加载 torchvision 0.8.0 保存的 MobileNetV2)
import torchvision.models.mobilenet as _mobilenet
class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
class _InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(_ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            _ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)
_ConvBNReLU.__module__ = 'torchvision.models.mobilenet'
_InvertedResidual.__module__ = 'torchvision.models.mobilenet'
setattr(_mobilenet, 'ConvBNReLU', _ConvBNReLU)
setattr(_mobilenet, 'InvertedResidual', _InvertedResidual)

class MobileNet:
    def __init__(self,modeldir=None):
        if modeldir is None:
            modeldir = asset_path('model/2021-05-20-08-method1.0-acc82-1.7.1+cpu.model')
        self.modeldir = modeldir
        self.loadmodel()
        pass
    def loadmodel(self):
        self.model = torch.load(self.modeldir, map_location='cpu', weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
    #@dask.delayed
    def predict_batch(self,batch):
        with torch.no_grad():
            out = self.model(batch)
            _, predicted = torch.max(out, 1)
            predicted = predicted.numpy()
        return predicted
    '''
    def predict_batchs(self,img_lst):
        tensors =  [self.transform(img) for img in img_lst]
        batches = [dask.delayed(torch.stack)(batch) for batch in toolz.partition_all(10, tensors)]
        delays = [self.predict_batch(batch) for batch in batches]
        res = dask.compute(delays)
        s = np.append([],res[0][:-1]).astype(np.int32)
        s = np.append(s,res[0][-1].astype(np.int32))
        return  s
    '''
    def predict(self,img):
        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            py = self.model(img)
        pb = torch.nn.functional.softmax(py, dim=1)
        _, predicted = torch.max(pb, 1)
        classIndex_ = predicted[0]
        return classIndex_.item()
class myNet():
    def __init__(self,datatype='img'):
        resnetdir = asset_path('model/2021-06-19-12-method3.0-acc80-lr0.004-batch30-1.7.1+cpu.pt')
        mobilenetdir = asset_path('model/2021-05-20-08-method1.0-acc82-1.7.1+cpu.model')
        self.datatype = datatype
        if self.datatype=='img':
            self.modeldir = mobilenetdir
        elif self.datatype=='series':
            self.modeldir = resnetdir
        self.loadmodel()
    def loadmodel(self):
        if self.datatype=='series':
            self.model = Net()
            pt = torch.load(self.modeldir, map_location='cpu', weights_only=True)
            self.model.load_state_dict(pt)
        elif self.datatype=='img':
            self.model = torch.load(self.modeldir, map_location='cpu', weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.datatype == 'img':
            self.transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])
    def predict(self,data):
        if self.datatype == 'img':
            img = self.transform(data)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            data = img
        elif self.datatype == 'series':
            data = torch.tensor(data)
        with torch.no_grad():
            outputs = self.model(data)
            _, preds = torch.max(outputs, 1)
        return preds.detach().numpy()[0]

class NetM3(nn.Module):
    #lens=250
    def __init__(self,n_output=6):
        super(NetM3, self).__init__()
        self.n_output = n_output
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=2,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.MaxPool1d(kernel_size=5),
            )
        self.rblock1 = ResidualBlock(32)
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3,padding=1),
            nn.MaxPool1d(kernel_size=2),
            )
        self.rblock2 = ResidualBlock(8)
        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=5),
            )
        self.rblock3 = ResidualBlock(8)
        self.L = nn.Linear(200,self.n_output)
        self.r = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.bn1d = nn.BatchNorm1d(40)
    def forward(self,x):
        x = self.c1(x)
        x = self.rblock1(x)
        x = self.c2(x)
        x = self.rblock2(x)
        #x = self.s(x)
        x = x.view(x.size(0),-1)
        x = self.bn1d(x)
        x = self.L(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv1d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(channels,channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        #self.bn1 = nn.BatchNorm1d(channels)
        #self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        return self.relu(x+y)
class Net(nn.Module):
    def __init__(self,n_output=6):
        super(Net, self).__init__()
        self.n_output = n_output
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=2,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,padding=1),
            nn.MaxPool1d(kernel_size=5),
            )
        self.rblock1 = ResidualBlock(32)
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3,padding=1),
            nn.MaxPool1d(kernel_size=2),
            )
        self.rblock2 = ResidualBlock(8)
        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=8,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=5),
            )
        self.rblock3 = ResidualBlock(8)
        self.L = nn.Linear(200,self.n_output)
        self.r = nn.ReLU(True)
        self.leakr = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self,x):
        x = self.c1(x)
        x = self.rblock1(x)
        x = self.rblock1(x)
        x = self.c2(x)
        x = self.rblock2(x)
        x = self.rblock2(x)
        x = x.view(x.size(0),-1)
        x = self.L(x)
        return x
        
class DataSet(Dataset):
    import pickle
    def __init__(self,path,task='train'):
        with open(path,'rb') as f:
            self.dic = pickle.load(f)
        self.data = self.dic[task]
        self.seq = torch.tensor(np.vstack([i for i in self.data.values()]).astype(np.float32))
        self.classes = torch.tensor(np.hstack([[k]*len(v) for k,v in self.data.items()]).astype(np.int64))
    def __len__(self):
        return len(self.seq)
    def __getitem__(self,index):
        data = self.seq[index]
        cla = self.classes[index]
        return data,cla
if __name__=='__main__':
    import torch.optim as optim
    from torch.optim import lr_scheduler
    import copy
    import time
    datapath = r'D:\code\py\CreateData/seqdataset.pkl'
    train_data = DataSet(datapath,'train')
    traindataloader = DataLoader(train_data,batch_size=30,shuffle=True)
    val_data = DataSet(datapath,'val')
    valdataloader = DataLoader(val_data,batch_size=30,shuffle=True)
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 80
    since = time.time()
    val_lens = len(val_data)
    train_lens = len(train_data)
    loss_lst,acc_lst,step_lst = [],[],[]
    step=0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        now = int(time.time())
        timeArray = time.localtime(now)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        print(otherStyleTime)
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dloader = traindataloader
            else:
                model.eval()   # Set model to evaluate mode
                dloader = valdataloader
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                step+=1
                step_lst.append(step)
                loss_lst.append(loss.item())
                acc_lst.append(torch.sum(preds == labels.data)/len(preds))
            if phase == 'train':
                scheduler.step()
            if phase == 'train':
                epoch_loss = running_loss / train_lens
                epoch_acc = running_corrects.double() / train_lens
            else:
                print(phase)
                print(running_corrects)
                epoch_loss = running_loss / val_lens
                epoch_acc = running_corrects.double() / val_lens
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, '.\\parameter.pkl')
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    
    
    
    
    
    
    
    
    
    r'''
    train = r'D:\code\py\SMFS\20210503-train-data\train'
    value = r'D:\code\py\SMFS\20210503-train-data\val'
    img_lst,data_array,target_array,val_img_lst,val_data_array,val_target_array = loaddata(train,value)
    target_array,val_target_array = target_array.astype(int),val_target_array.astype(int)
    svc = svm.SVC(gamma=0.001, C=100)
    rf = RandomForestClassifier(n_estimators=1000,  max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True, n_jobs=1, random_state=1)
    svc.fit(data_array,target_array)
    print("accuracy of svc is {}".format(sum(svc.predict(val_data_array)==val_target_array)/len(val_target_array)))
    rf.fit(data_array,target_array)
    print("accuracy of rf is {}".format(sum(rf.predict(val_data_array)==val_target_array)/len(val_target_array)))
    m = mobilenet(train,value,8)
    print("accuracy of mb is {}".format(sum(m.predict(val_img_lst)==val_target_array)/len(val_target_array)))
    #k = kmeans(train,value)
    #k.fit()
    lg = LogisticRegression(max_iter=50000)
    lg.fit(data_array,target_array)
    print("accuracy of lg is {}".format(sum(lg.predict(val_data_array)==val_target_array)/len(val_target_array)))
    svc_re = svc.predict(val_data_array)
    rf_re = rf.predict(val_data_array)
    m_re = m.predict(val_img_lst)
    lg_re = lg.predict(val_data_array)
    lst_blend = []
    score_max = 0
    weight_max = np.array([0,0,0,0])
    
    for weight in itertools.product(np.arange(7),np.arange(7),np.arange(7),np.arange(7)):
        #weight = np.random.randint(0,10,size=(5))
        #weight = [1,3, 6, 5, 9]
        weight = np.array(weight)
        if np.sum(weight)==0 or weight.min()!=1 and np.sum(weight%weight.min())==0:
            continue
        for i in range(len(weight)):
            for n in range(weight[i]):
                lst_blend.append(lst_re[i])
        
        blendtrain = np.dstack(lst_blend)[0].astype(np.int)
        score = sum(np.array([stats.mode(i)[0][0] for i in blendtrain])==val_target_array)/len(val_target_array)
        if score>score_max:
            weight_max = weight
            score_max = score
            print(score_max)
            print(weight_max)
        if score>0.9:
            break
    print(score_max)
    print(weight_max)
    voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression(max_iter=50000)),
    ('svm_clf', svm.SVC(probability=True)),
('rf_clf',RandomForestClassifier(n_estimators=1000,  max_features='sqrt', max_depth=None, min_samples_split=2, bootstrap=True, n_jobs=1, random_state=1))]
        , voting='soft')
    voting_clf.fit(data_array, target_array)
    score = voting_clf.score(val_data_array, val_target_array)
    print("accuracy of voting is {}".format(score))
    
    blendtrain = np.dstack((svc.predict(val_data_array),
                            rf.predict(val_data_array),
                            m.predict(val_img_lst),
                            m.predict(val_img_lst),
                            k.predict(val_data_array),
                            lg.predict(val_data_array),
                            lg.predict(val_data_array),
                            lg.predict(val_data_array)))[0].astype(np.int)
    '''
    #print(sum(np.array([stats.mode(i)[0][0] for i in blendtrain])==val_target_array)/len(val_target_array))
    
