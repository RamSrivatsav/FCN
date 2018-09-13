## FCN on KITTI dataset

The program performs semantic segmentation on [KITTI dataset](http://www.cvlibs.net/datasets/kitti/). 

Data loader, model, multi-GPU training, evaluation and visualization are all involved in the implementation. The data loader and model part are flexible. 

Install the requirements first. 

```powershell
pip install -r requirements.txt
```

To run the code, read the help first. Or, you can directly run train.py and see what will happen.

```commonlisp
python train.py -h
usage: train.py [-h] [--batch-size N] [--split Split] [--resize-ratio Resize]
                [--numloader Nl] [--epochs E] [--lr Lr] [--lr-pretrain LR-P]
                [--momentum M] [--decay D] [--step Step] [--gamma Gamma]
                [--logdir Log] [--vgg Vgg] [--fcn FCN] [--mode Mode]
                [--model Model]

pytorch FCN on Kitti data set

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        the training batch size(default: 8)
  --split Split         the split ratio indicates how much percentage of train
                        set, the rest will for validation (default: 0.8)
  --resize-ratio Resize
                        how much ratio to resize(shrink) the training
                        image(default: 0.6)
  --numloader Nl        the num of CPU for data loading. 0 means only use one
                        CPU. (default: 8)
  --epochs E            the required total training epochs(default: 100)
  --lr Lr               the learning rate for decoder part(default: 1e-3)
  --lr-pretrain LR-P    the learning rate for encoder(pre-trained)
                        part(default: 1e-4)
  --momentum M          the momentum of optimizer(default: 0)
  --decay D             the weight for L2 regularization(default: 1e-5)
  --step Step           learning rate will decay after the step(default: 10)
  --gamma Gamma         learning rate will decay gamma percent after few
                        steps(default: 0.5)
  --logdir Log          the folder to store the tensorboard logs(default: log)
  --vgg Vgg             the configuration of vgg(default: 11)
  --fcn FCN             the configuration of FCN(default: 1)
  --mode Mode           train or test(default: train)
  --model Model         the pre-trained model or the model for test. if not
                        specify, it will initial a new model (default: )
```

The training model of each epoch will be stored on your drive (in ./store folder). 

Run tensorboard to tracing the training state:

```powershell
tensorboard --logdir log
```

