# PyTorchFederated
Experimenting with Federated learning using PyTorch
<p><h1> What is Federated learning? </h1><p>
<p align="center">
  <img src="https://blogs.nvidia.com/wp-content/uploads/2019/10/federated_learning_animation_still_white.png" width="500" title="scheme of FL">
  
</p>
<p>
  <a href="https://medium.com/@ODSC/what-is-federated-learning-99c7fc9bc4f5">Federated learning (FL)</a> is an approach that downloads the current model and computes an updated model at the device itself (ala edge computing) using local data. These locally trained models are then sent from the devices back to the central server where they are aggregated, i.e. averaging weights, and then a single consolidated and improved global model is sent back to the devices.
<p>
<p><h1> Description </h1><p>  
<p>
  This project allows you to estimate federated model performance compared to centralized trained using single PC. Emulating clients, federated model being computed by splitting given dataset on equal parts, implementing simple federated averaging weights algorythm. Then single model with same architecture learns on whole given dataset. Performance graph of both models on validation dataset is plotted after end of experiment along with saving training logs.
<p>
<p><h1> Usage </h1><p>  
<pre><code>
<p>from federated import experiment
<p>experiment(arch, train_ds, val_ds, pref, n_ep, n_clients, n_exp, lr, epoch_step, bs)
<p>arch: clients and main model architecture
train_ds: nn.Dataset, training data
val_ds: nn.Dataset, validation data
pref: prefix to add in experiment result folder name
n_ep: total number of training epochs
n_clients: number of client models
n_exp: number of experiments with different initial weights
lr: learning rate
epoch_step: number of local training epochs
bs: batch size
</code></pre>
