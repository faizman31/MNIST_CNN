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


