# MNIST_CNN
---
본 Repository는 패스트캠퍼스의 딥러닝 초급을 참고하여 제작되었습니다.
---
## 0.Code Composition
- mnist_classification : 학습에 필요한 모델,data_loader,Trainer가 포함되어 있는 폴더
  - data_loader.py : MNIST 데이터를 다운로드하고 데이터를 Pytorch의 DataLoader에 싣는 역할을 하는 코드
  - utils.py : Gradient norm 과 Parameter norm를 연산하는 코드
  - trainer.py : Pytorch Ignite를 기반으로 학습과정에 필요한 메서드들과 클래스가 존재하는 코드
  - models 폴더 : ConvolutinalClassifer , FullyConnectedLayer 모델과 관련된 클래스가 존재하는 코드
- train.py : 학습에 필요한 Argument를 설정하고 학습을 실행하는 코드
- predict.ipynb : 학습 결과를 확인하기 위한 jupyter notebook 코드
---
## 1. data_loader.py
data_loader.py는 MNIST 데이터를 다운로드 받는 메서드와 해당 데이터를 Pytorch의 DataLoader에 싣기 위한 MNISTDataset 클래스를 구성한 파일입니다. 
- Library Import
```
import torch
from torch.utils.data import Dataset,DataLoader
```  

- MNISTDataset
```
class MNISTDataset(Dataset): # pytorch.utils.data의 Dataset 상속
    def __init__(self,data,labels,flatten=True):
        self.data = data # MNIST 데이터
        self.labels = labels # MNIST 데이터의 레이블
        self.flatten = flatten # Flatten 여부

        super().__init__()

    def __len__(self): # data의 length Return
        return self.data.shape[0] # (N,28,28)
    
    def __getitem__(self,idx): # data안에 idx에 해당하는 데이터&레이블 Return
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten: # 만약 Flatten이 True라면
            x = x.reshape(-1) # (28,28) ->(784)
        
        return x,y
```  

- load_mnist : MNIST 데이터를 다운로드 받는 메서드
```
def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y
```  

- get_loaders : 데이터를 Pytorch의 DataLoader의 싣는 메서드
```
def get_loaders(config):
    x,y = load_mnist(is_train=True,flatten=False)

    # Define train/valid Count - train_cnt,valid_cnt
    train_cnt = int( x.shape[0] * config.train_ratio )
    valid_cnt = x.shape[0] - train_cnt

    flatten = True if config.model == 'fc' else False

    indices = torch.randperm(x.shape[0]) # Data Index shuffle

    # Data Shuffle -> Train/Valid Split
    train_x,valid_x = torch.index_select(x,dim=0,index=indices).split([train_cnt,valid_cnt])
    train_y,valid_y = torch.index_select(y,dim=0,index=indices).split([train_cnt,valid_cnt])

    # Train/Valid Data Loading DataLoader
    train_loader = DataLoader(
        dataset = MNISTDataset(train_x,train_y,flatten),
        batch_size = config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset = MNISTDataset(valid_x,valid_y,flatten),
        batch_size = config.baatch_size,
        shuffle=True,
    )

    # Test data Loading DataLoader
    test_x,test_y = load_mnist(is_train=False,flatten=False)
    test_loader = DataLoader(
        dataset = MNISTDataset(test_x,test_y),
        batch_size = config.batch_size,
        shuffle=False,
    )

    return train_loader,valid_loader,test_loader
```
---
## 2. utils.py 
utils.py 는 학습 시 Gradient의 크기와 Weight Parameter들의 크기를 구하기 위한 get_grad_norm와 get_parameter_norm이 구현되어 있는 파일 입니다.
- Library import
```
import torch
import numpy as np
```  
- get_grad_norm 
```
def get_grad_norm(parameters,norm_type=2):
    parameters = list(filter(lambda p : p.grad is not None,parameters))

    total_norm = 0
    
    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm +=param_norm ** norm_type
        total_norm = total_norm ** (1./norm_type)
    excep Exception as e:
        print(e)

    return total_norm
```  
- get_parameter_norm
```
def get_parameter_norm(parameters,norm_type=2):
    total_norm=0
    
    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm **(1./norm_type)
    except Exception as e:
        print(e)

    return total_norm
```




