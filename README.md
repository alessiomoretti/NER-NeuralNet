# NER project for Italian tweets #
This project aims to solve the task stated in [NEEL homepage](http://neel-it.github.io)
The aim is to use a Bi-LSTM neural network in order to solve the NER task stated before.
The project relies also upon the DBPedia API in order to link the found entities.

### PRE-REQUISITES ###
To run the project you will need the following setup:
1. Python 3.x environment
2. Tensorflow 1.2 +
3. Python Requests module (`pip3 install requests`)
4. Fill the dataset (embeddings too) with your own (no working copy of the original dataset is included for license issues)

### USAGE QUICKSTART ###
Run in a terminal:
```bash
python3 main.py --pretrain --train_nfold
```
This will get the model to be pre-trained and cross validated.
To run the interactive shell (the model is the one saved in the `trained` folder) you will simply need to run the following:
```bash
python3 main.py --interactive
```
For further info please run the following:
```bash
python3 main.py --help
```

### METRICS ###
When the train and testing routines are finished, a `metrics.txt` file will be produced in the working directory.

### MACROS ###
Please, use the `macros.py` file to edit preferences. 

### Drop me a line! ###
For any issue, please contact me [here](mailto:alessio.moretti@live.it)

