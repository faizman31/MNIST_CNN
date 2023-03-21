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
---
## 3. trainer.py
Pytorch Ignite를 활용한 Trainer를 구현해놓은 파일입니다. Trainer.py 에서는 model,loss_function,optimizer,config 를 전달받아 train_loop와 validation_loop 를 실행하고
check_best 메서드를 통해 최적의 모델을 갱신하고 save_model 메서드를 통해 모델을 저장합니다.
- Library Import
```
from copy import deepcopy

import torch 
import torch.nn as nn 
import torch.nn.utils as torch_utils
import torch.optim as optim

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from mnist_classification.utils import get_grad_norm,get_parameter_norm
```

- MyEngine 
```
class MyEngine(Engine):
    def __init__(
        self,
        func,
        model,
        crit, # == Loss Function
        optimizer,
        config,
    ):
        self.model=model
        self.crit=crit
        self.optimizer=optimizer
        self.config=config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine,mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()
        
        x,y = mini_batch
        x,y = x.to(engine.device),y.to(engine.device)

        y_hat = engine.model(x)
        
        loss=engine.crit(y_hat,y)
        loss.backward()

        if isinstance(y,torch.LongTensor) or isinstace(y,torch.cuda.LongTensor):
            accuracy = (torch.argmax(y,dim=-1)==y).sum() / float(y.shape[0])
        else:
            accuracy = 0
        
        p_norm = get_parameter_norm(engine.model.parameters())
        g_norm = get_grad_norm(engine.model.parameters())

        return {
            'loss':float(loss),
            'accuracy':float(accuracy),
            '|param|':p_norm,
            '|g_param|':g_norm
        }
    @staticmethod
    def validate(engine,mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x,y = mini_batch
            x,y = x.to(device) , y.to(device)
            
            y_hat = engine.model(x)

            loss = engine.crit(y_hat,y)

            if isinstance(y,torch.LongTensor) or isinstance(y_hat,torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat,dim=-1)==y).sum() / float(y.shape[0])
            else:
                accuracy = 0

            return {
                'loss':loss,
                'accuracy':accuracy
            }
    
    @staticmethod
    def attach(train_engine,valid_engine,train_loader,valid_loader,verbose=VERBOSE_BATCH_WISE):
        def attach_running_average(engine,metric_name):
            RunningAverage(output_transform=lambda x : x[metric_name]).attach(
                engine,
                metric_name
            )

        training_metric_names=['loss','accuracy','|param|','|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine,metric_name)
        
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None,n_cols=120)
            pbat.attach(train_engine,training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('EPOCH {} - |param|={:.4e} |g_param|={:.4e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'].
                ))
        
        validation_metric_names = ['loss','accuracy']

        if verbose >= VEBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None,ncols=120)
            pbar.attach(valid_engine,validation_metric_names)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @valid_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))
        

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine,train_engine,config,**kwargs):
        torch.save({
            'model' : engine.best_model,
            'config' : config,
            **kwargs
        },config.model_fn)
```  

- Trainer
```
class Trainer():
    def __init__(config):
        self.config=config

    def train(self,model,crit,optimizer,train_loader,valid_loader):
        train_engine = MyEngine(
            MyEngine.train,
            model,crit,optimizer,self.config
        )
        valid_engiine = MyEngine(
            MyEngine.validate,
            model,crit,optimizer,self.config
        )

        MyEngine.attach(train_engine,valid_engine,verbose=self.config.verbose)

        def run_validation(engine,valid_engine,valid_loader):
            valid_engine.run(valid_loader,max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            valid_engine,valid_loader
        )
        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            MyEngine.check_best,
        )
        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            MyEngine.save_model,
            train_engine,self.config
        )

        train_engine.run(train_loader,max_epochs=self.config.n_epochs)

        model.load_state_dict(valid_engine.best_model)

        return model
```


