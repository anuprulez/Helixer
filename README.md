# HelixerPrep
Gene calling with Deep Neural Networks.

## Disclaimer
This is beta, or maybe alpha... there is nothing stable here.

## Goal
Setup and train models for _de novo_ prediction of gene structure.
That is, to perform "gene calling" and identify
which base pairs in a genome belong to the UTR/CDS/Intron of genes. 
Train one model for a wide variety of genomes.

## Install 
Preferably in a virtual environment:

### Non PyPi Requirements
Both can best be installed according to their own installation instructions
* dustdas, https://github.com/janinamass/dustdas
* geenuff, https://github.com/weberlab-hhu/GeenuFF

### The Rest
```
pip install -r requirements.txt
```

### HelixerPrep itself

```
# from the HelixerPrep directory
python setup.py develop  # or `install`, if someone who isn't working on this actually installs it
```

## Example
We set up a working example with a limited amount of data. For this, we will utilize the GeenuFF database in located at `example/three_algae.sqlite3` to generate a training, validation and testing dataset and then use those to first train a model and then to make gene predictions with Helixer.

Our example database contains the genetic information (that is the information of the FASTA and GFF3 files from the Phytosome 13 database) of three very small algae: *Ostreococcus_sp._lucimarinus*, *Micromonas_sp._RCC299* and *Micromonas_pusilla*. We will train our model with the first two and then predict on the third.

```
# generating the training and validation dataset
# the genome names are a bit different due to the naming used in Phytosome
python3 export.py --db-path-in example/three_algae.sqlite3 --genomes Olucimarinus,MspRCC299 --out-dir example/train 
```

```
# generating the test dataset
python3 export.py --db-path-in example/three_algae.sqlite3 --genomes MpusillaCCMP1545 --out-dir example/test --only-test-set 
```

We should now have the following files in `example/`. Note that the test data was not split into a training and validation set due to the `--only-test-set` option: 
```
example/
├── test
│   └── test_data.h5
├── three_algae.sqlite3
└── train
    ├── training_data.h5
    └── validation_data.h5
```
Now we use the datasets in `example/train/` to train a model with our LSTM architeture for 5 epochs and save the best iteration (according to the Genic F1 on the validation dataset) to `example/best_helixer_model.h5`.
```
python3 helixerprep/prediction/LSTMModel.py --verbose --data-dir example/train/ --save-model-path example/best_helixer_model.h5 --epochs 5 --units 64 --pool-size 10
```

Right before the training starts we may get one or two warnings about the data that are not relevant for this example. This trains a very small single layer BLSTM model with about 60.000 parameters that receives 10 basepairs for each time step. If we would like to train a model with exactly the architecture and weights that was used to train our actual plant models one would use the following command. 

```
python3 helixerprep/prediction/LSTMModel.py --verbose --data-dir example/train/ --save-model-path example/best_helixer_model.h5 --epochs 5 --units 256 --pool-size 10 --batch-size 52 --layers 4 --layer-normalization --class-weights '[0.7, 1.6, 1.2, 1.2]'
```

We can now use this model to produce predictions for our test dataset.






