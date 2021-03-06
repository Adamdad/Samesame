# Same, Same But Different
An Tensorflow.keras implementation of ICML2019 paper 
"Same, Same But Different - Recovering Neural Network Quantization Error Through Weight Factorization"(https://arxiv.org/pdf/1902.01917.pdf)
![motivation](inverse_frac.png)
## Quick Start
This code has been tested on Ubuntu 18.04, Python 3.7, Tensorflow 2.0

- Clone this repository

        git clone https://github.com/Adamdad/Samesame.git 
 
- Network equalization

        python equalization.py
     
     In this code, I default equalize the _Inception_v3_ implemented by **keras.application** 

- Network visualization(in visual_weights_per_channel.py)
    - Per-channel convolution kernel weight visualization
            
            def visual_weight(model)
    - Per-channel activation feature map visualization
            
            def visual_activation(model,x)
    - Network architecture visualization
            
            def visual_graph(model,fig_name='model.png')

## Quantization
The post-training quantization is implemented with TF-Lite, with two mode of "weight quantization" and "full quantization". See details in *def qevaluate*

## Performance Compare
#### Imagenet Validation result
| Network       | 32bit      |          | 32bit-Equalized|             | 8bit-weight |          | 8bit-Equalized-weight |             |   
|---------------|-----------|----------|--------------|----------------|-------------|----------|---------------|--------------|
|               |  top1    |     top5   |     top1    |     top5       |   top1      |    top5  |    top1       |    top5      |
| Inception-v3  | 76.276   |    93.032  |     75.61   |     92.58      |             |          |               |                    |
    
## Citation
    @article{meller2019same,
      title={Same, same but different-recovering neural network quantization error through weight factorization},
      author={Meller, Eldad and Finkelstein, Alexander and Almog, Uri and Grobman, Mark},
      journal={arXiv preprint arXiv:1902.01917},
      year={2019}
    }
