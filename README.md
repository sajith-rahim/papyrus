
████████████████████████████████████████████
█▄─▄▄─██▀▄─██▄─▄▄─█▄─█─▄█▄─▄▄▀█▄─██─▄█─▄▄▄▄█
██─▄▄▄██─▀─███─▄▄▄██▄─▄███─▄─▄██─██─██▄▄▄▄─█
▀▄▄▄▀▀▀▄▄▀▄▄▀▄▄▄▀▀▀▀▄▄▄▀▀▄▄▀▄▄▀▀▄▄▄▄▀▀▄▄▄▄▄▀



*Generic template to bootstrap your PyTorch project.*

* Defines a generic folder structure
* Configuration based 
* Command line option support to overide loaded config file
* Tensorboard Support
* Checkpointing and Resume

<p>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white" />

<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" />
<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Shell_Script-121011?style=for-the-badge&logo=gnu-bash&logoColor=white" />
</p>
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
| hydra_core| 1.1.1 |
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


## Checkpointing

Set the resume flag and checkpoint name in `config.yaml`

```yaml
*config.yaml*
------------
defaults:
  ...
paths:
  ...
params:
  ...
checkpoint:
  save_interval: 5 #iter interval for checkpointing
  resume: False # resume flag if True set checkpoint_id
  checkpoint_id: <ID>.pt.zip # saved checkpoint filename
  path: ${hydra:runtime.cwd}/checkpoints #save path
```

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

# Folder Structure

```powershell
|   down_mnist.sh
|   LICENSE
|   main.py
|   note.txt
|   README.md
|   requirements.txt
|   spawn.py
+---base
|   |   base_dataloader.py
|   |   base_dataset.py
|   |   base_model.py
|   |   __init__.py
|
+---config
|   |   config.py
|   |   mnist_config.py
|   |   __init__.py
|   |
|   +---conf
|   |   |   config.yaml
|   |   |
|   |   \---files
|   |           mnist.yaml
|
+---data
|   \---raw
|           t10k-images-idx3-ubyte.gz
|           t10k-labels-idx1-ubyte.gz
|           train-images-idx3-ubyte.gz
|           train-labels-idx1-ubyte.gz
|
+---dataloader
|   |   mnist_dataloader.py
|   |   __init__.py
|   |
|   +---dataset
|   |   |   mnist.py
|   |   |   __init__.py
|
+---logs
|   +---0
|   |       events.out.tfevents.1641064646.Drovahkin.7748.0
|
+---metrics
|   |   losses.py
|   |   metric.py
|   |   __init__.py
|
+---models
|   |   mnist_model.py
|   |   __init__.py
|
+---task_runner
|   |   task_runner.py
|   |   __init__.py
|
+---tracker
|   |   phase.py
|   |   tensorboard_experiment.py
|   |   track.py
|   |   __init__.py
|   
|
\---utils
    |   config_utils.py
    |   data_utils.py
    |   device_utils.py
    |   download_utils.py
    |   os_utils.py
    |   tracker_utils.py
    |   __init__.py
```



## Deployment

 🚧 🅆🄾🅁🄺 🄸🄽 🄿🅁🄾🄶🅁🄴🅂🅂
    
## Contributing

Feel free to fork.

## License

    MIT

## Future

<img align="right" style="float:right;border:3px solid black" width=64 height=92 src="https://raw.githubusercontent.com/sajith-rahim/cdn/main/content/blog/media/warn_tag.png" />

 * ~~Resume and Checkpointing~~
 * Add WANDB support
 * UI

## Acknowledgments

  Inspired from <a href="https://github.com/victoresque/pytorch-template" target="_blank">victoresque</a>'s putorch template project.
