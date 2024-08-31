# SSL-CST
A cell segmentation method for single-cell spatial transcriptome based on self-supervised learning

<img src="./data/ssl_network2.jpg" alt="" width="30%">

## Installation
SST-CST depends on the  packages in requirements.txt. To speed up the training process,SST-CST  relies on Graphic Processing Unit (GPU). If no GPU device is available, the CPU will be used for model training.

```conda create -n SSL-CST python=3.8```  
```pip install -r requirements.txt```

## Usage
The data needed for the method is segregated into two components: spatial transcriptome sequencing data and the accompanying nuclear staining images. The sequencing data should encompass essential information, including the gene type, its coordinates, and the intensity of gene expression.

You can run a demop from the command line:


```python SSL-CST.py```

## Results
The final output reports the information of cells in the spatial transcriptome stained image.


