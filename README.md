# Transfer learning for semi supervised siamese networks in the domain of cell counting
These is the sourcecode for the paper “Transfer learning for semi supervised siamese networks in the domain of cell counting”.

## Usage instructions
Use Python 3 (perferably 3.7.3) and the requirements.txt and your favorite module management tool to make sure all required modules are installed. CUDA is required to run the project. We recommend to use CUDA 11.0, but 10.1 or 10.2 should work too. Should you have any trouble to install torch and torchvision with CUDA enabled, you can append to the module installation command to get the packages directly from pytorch.org.

```
-f https://download.pytorch.org/whl/torch_stable.html
```

Make sure the data directory contains the folders 128p_bf etc. and has the same parental path as the vae folder. It should look like this:

 .
    ├── ...
    ├── data 
	│	├── 128p_bf
	│	└── ...
    ├── ...
    ├── vae
    └── ...

The checkpoints have been stored as archives, split into 4 parts, to not exceed the Github file limit size of 100MB. Unzipping the archive checkpoints.zip is required to use the checkpoints.

Call cli.py with adjusted parameters, if desired.

```
python cli.py 
```

should work, but can be specified to e.g.

```
python cli.py --epochs 10000 --lr 2e-4
```

If the images can not be found, adjust dataloader.py

```
prefix = os.getcwd().replace("\\", "/")[:-4]  # gets the current path up to /vae and removes the /vae to get to the data directory
```

to point to the data folder containing the images.

### Notes
The bayes_opt package as well as radam.py have only been marginally edited for this project. Their structure, mandatory content, documentation etc. remain the same. Credits are given within the code.

### MEAN & STD of data
PC:  
mean: 0.2400501283139721  
std:  0.042060589085307146  
mean: 0.2401007512954981  
std:  0.04210882536082098  
mean: 0.2332135032730054  
std:  0.05502266365054745  
mean: 0.23327550803772126  
std:  0.05510969431303569  

BF:  
mean: 0.7437175269126892  
std:  0.03268717645443394  
mean: 0.7438292989730835  
std:  0.0326244184570096  
mean: 0.7073599379668114  
std:  0.03215151615350316  
mean: 0.7098200608784123  
std:  0.03197213160801554  
