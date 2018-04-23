# VGG Siamese

A siamese model designed to recognise natural landscape 
pictures taken at the same place but at different time 
of the year.

## Requirement

 - Python >= 3.6
 - [keras](https://github.com/keras-team/keras) (only tested with TensorFlow backend)
 - numpy
 - opencv (cv2)
 - pandas
 - matplotlib
 - argparse
 
(Also use standard lib such as `os`, `glob` and `pickle`)

## How to Run ?

The access point to the siamese network is `VVGsiamese.py`

For example you can run it with:

```bash
python3.6 -m siamese_network.VGGsiamese -o output_dir -e 1 -m dot -dp ./data
```
   
This will train the dense part at the end of the network during one epoch on the dataset located at './data'
using the merge stategy 'dot' and will write all the results into the 'output_dir' folder.

To get a description of the effect of each parameters you can run the following:

```bash
python3.6 -m siamese_network.VGGsiamese -h
```

Which should give you something similar to:
 
```commandline
usage: VGGsiamese.py [-h] [-b BATCH_SIZE] [-bn USE_BATCH_NORM]
                     [-m MERGE_LAYER] [-vl VGG_FROZEN_LAYER]
                     [-vrb VGG_NB_BLOCK_TO_REMOVE] [-o OUTPUT_DIR]
                     [-e NUMBER_OF_EPOCH] [-d DROPOUT] [-lr LEARNING_RATE]
                     [-lrd LEARNING_RATE_DECAY] [-op OPTIMIZER]
                     [-f FINE_TUNING_ITERATION] [-dp DATA_SET_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -bn USE_BATCH_NORM, --batch-norm USE_BATCH_NORM
  -m MERGE_LAYER, --merge-layer MERGE_LAYER
  -vl VGG_FROZEN_LAYER, --vgg-frozen-limit VGG_FROZEN_LAYER
  -vrb VGG_NB_BLOCK_TO_REMOVE, --vgg-nb-block-to-remove VGG_NB_BLOCK_TO_REMOVE
  -o OUTPUT_DIR, --out-dir OUTPUT_DIR
  -e NUMBER_OF_EPOCH, --epochs-per-step NUMBER_OF_EPOCH
  -d DROPOUT, --dropout DROPOUT
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
  -lrd LEARNING_RATE_DECAY, --learning-rate-decay LEARNING_RATE_DECAY
  -op OPTIMIZER, --optimizer OPTIMIZER
  -f FINE_TUNING_ITERATION, --fine-tuning-iteration FINE_TUNING_ITERATION
  -dp DATA_SET_PATH, --dataset-path DATA_SET_PATH
``` 
 
 
