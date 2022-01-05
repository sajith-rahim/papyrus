
████████████████████████████████████████████
█▄─▄▄─██▀▄─██▄─▄▄─█▄─█─▄█▄─▄▄▀█▄─██─▄█─▄▄▄▄█
██─▄▄▄██─▀─███─▄▄▄██▄─▄███─▄─▄██─██─██▄▄▄▄─█
▀▄▄▄▀▀▀▄▄▀▄▄▀▄▄▄▀▀▀▀▄▄▄▀▀▄▄▀▄▄▀▀▄▄▄▄▀▀▄▄▄▄▄▀


*Generic template to bootstrap your PyTorch project.*

* Defines a generic folder structure
* Configuration based 
* Command line option support to overide loaded config file
* Tensorboard Support


## Getting Started



### Prerequisites

| Package     | Version      |
|:----------------|:---------------|
| torch| 1.10.0 |
| torchvision| 0.11.1 |
| omegaconf| 2.1.1 |
| validators| 1.18.5 |
| matplotlib|3.4.1 |
| requests|2.22.0 |
| hydra| 2.5 |
| dataclasses| 0.8 |
| numpy| 1.18.5 |
| tqdm| 4.62.3 |


### Installing

```powershell
pip install -r requirements.txt
```



### Download scripts

Example: MNIST

Run `down_mnist.sh` to download using curl

OR 

Programaticaly download using 
```powershell
utils.download_utils.FileDownloader
```


## Running 

From the root directory run

```powershell
python main.py
```


#### Output

```powershell
files:
  test_data: t10k-images-idx3-ubyte.gz
  test_labels: t10k-labels-idx1-ubyte.gz
  train_data: train-images-idx3-ubyte.gz
  train_labels: train-labels-idx1-ubyte.gz
paths:
  log: <RESOLVED_DIR_PATH>/logs
  data: <RESOLVED_DIR_PATH>/data/raw
params:
  epoch_count: 20
  lr: 5.0e-05
  batch_size: 128
  shuffle: true
  num_workers: 2

Active device: cpu
MnistModel(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
)
Trainable parameters: 21840
Train Progress:  64%|█████████████████████████████████████████▋                       | 301/469 [00:15<00:08, 19.13it/s]
```

To override params from CLI:

```powershell
python main.py params.num_workers=2
```

To override params from CLI if exists else add:

```powershell
python main.py ++params.other_params=abc
```


#### For grouping configs for experiments:
<div style='page-break-after: always'></div>


```
├─ files
│  ├─ mnist.yaml
│  └─ fmnist.yaml
└── config.yaml
```

```powershell
    python main.py +files=fmnist
```
*You can find more info about grouping* <a href="https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/" target="_blank">here</a>




## Tensorboard Server

Check the name of your log folder under `config.defaults.log`

```yaml
*config.yaml*
------------
defaults:
  ...
paths:
  log: ${hydra:runtime.cwd}/runs
  data: ...
params:
  ...
```


From project root run
```powershell
tensorboard --logdir runs
```

tensorboard server will open at

    http://localhost:6006


## Spawn a new Project


```powershell
python spawn.py ../new-project
```
```powershell
─────────────╔═╗
╔═╦═╗╔═╦╦╦╦╦╦╣═╣
║╬║╬╚╣╬║║║╔╣║╠═║
║╔╩══╣╔╬╗╠╝╚═╩═╝
╚╝───╚╝╚═╝

New project initialized at <PATH>\new-project
```


## Deployment

 🚧 🅆🄾🅁🄺 🄸🄽 🄿🅁🄾🄶🅁🄴🅂🅂
    
## Contributing

Feel free to fork.

## License

    MIT

## Future

 * Resume and Checkpointing
 * Add WANDB support
 * UI

## Acknowledgments

  Inspired from <a href="https://github.com/victoresque/pytorch-template" target="_blank">victoresque</a>'s putorch template project.