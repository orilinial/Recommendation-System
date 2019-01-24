# Recommendation System Using Deep Auto Encoder
In this project we created a recommendation system for the rating prediction task. We used the Netflix Prize dataset, where users' ratings are given per movie between 1 to 5. </br>
Our model is a Deep Auto Encoder and based on the one in [Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/abs/1708.01715) by Kuchaiev & Ginsburg 2017. </br>
A typical auto encoder architecture:

<p align="center">
<img align="center" src="https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png" alt="Typical auto encoder architecure">
</p>

</br>
Our model has 6 layers and is trained without any layer-wise pre-training. We used the SELU activation function, and high dropout in the middle layer as suggested in the original paper. </br>
In our analysis we empirically argue that: </br> 
<p>
<ol>
<li>Using the same weights in the encoder and the decoder (tied networks) achieve almost the same results with half the amount of parameters.</li>
<li>The network without batch-norm achieves better results than the one with batch norm.  We also demonstrate that there is a connection between the norm of the weights to the chosen learning rate, this is also discussed in <a href="https://arxiv.org/abs/1803.01814">Norm Matters</a> by Hoffer et al.</li>
<li>The newly proposed training algorithm which re-feeds the outputs of the model back to the input (in order to overcome the natural sparseness of the rating prediction task), does not improve results. The same results are obtained with re-feeding and without. This is probably since the network already learns that the output is a fixed point by itself.</li>
<li>Dropping some of the inputs in order to let the network learn from unknown inputs, does not improve results.</li>
</li>
</p>

## Requirements
Python 3.6 </br>
Pytorch </br>
Numpy </br>

## Netflix Prize data:
The Netflix Prize data can be downloaded [here]('ModelArch.png').

In order to create the pytorch supported dataset, we provided 4 different functions in the file `utils/data_utils.py`:
```
neflix_full()
netflix_3months()
neftlix_6months()
netlix_1year()
```
These functions create the different datasets we ran our experiments on. Each dataset corresponds to a different time interval of ratings (e.g. 3 months corresponds to September - November 2005, as they describe in the original paper).

## Running experiments
We provided the file `run.sh` which is a BASH script for running all the required experiments. This script first creates the relevant datasets (as described above), and then runs the model using its different variations - Input dropout, refeeding iterations, batchnorm, output clipping and tied networks as we described in the introduction.
