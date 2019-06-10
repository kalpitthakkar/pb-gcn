# HOW-TO: Configuration File

The default structure of the configuration file for running the
training / testing for the model is:

```
work_dir: <path_to_work_directory>
data_path: <path_to_directory_with_skeleton_files>
missing_txt: <path_to_missing_samples_txt>

# Data Loader
loader: <name_of_dataset_loader_class>
train_loader_args:
  <keyword_args_for_the_loader_training>
  arg1: val1
  arg2: val2
  ...
  ...
test_loader_args:
  is_training: False
  <keyword_args_for_the_loader_testing>
  arg1: val1
  arg2: val2
  ...
  ...

# Model
model: <model_name_as_exposed_by_imports>
model_args:
  layers_config: <list_of_shapes>
  num_class: <num_action_classes>
  channel: <number_of_input_feature_channels>
  window_size: <max_num_frames>
  num_joints: <num_of_tracked_joints>
  num_actors: <num_skeletons_per_frame>
  graph: <dataset_graph_class_name>
  graph_args:
    labeling_mode: <adjacency_matrix_construction_method>
  mask_learning: <mask_adjacency_weights_or_not>
  use_data_bn: <batch_normalize_inputs_or_not>

# Optimization
weight_decay: <weight_decay_constant>
base_lr: <init_learning_rate>
step: <list_of_epochs_to_decay_lr>

# Training
device: <list_of_GPU_IDs>
batch_size: <training_batch_size>
test_batch_size: <testing_batch_size>
num_epoch: <num_epochs_training>
nesterov: <use_nesterov_momentum_updates_or_not>
save_interval: <epochs_between_saving_model>

# Evaluation
eval_interval: <epochs_between_evaluating_model>

# Initialization from checkpoint
# start_epoch: <epoch_to_begin_training_from>
# weights: <path_to_pretrained_weights>
```

The meanings of each of the arguments are self-explanatory or are explained
through the brief descriptions given. The configuration used for training
the model on cross-subject split on NTURGBD given is:

```
work_dir: /media/data_cifs/Kalpit/NTURGB+D/work_dir/cs/STGCN_parts_noaug
data_path: /media/ssd_storage/NTURGB+D/nturgb+d_skeletons
missing_txt: /media/ssd_storage/NTURGB+D/samples_with_missing_skeletons.txt

# Data Loader
loader: NTULoader
train_loader_args:
  split_dir: /media/data_cifs/Kalpit/NTURGB+D/data/NTURGB+D/cs
  signals:
    temporal_signal: True
    spatial_signal: True
    all_signal: False
test_loader_args:
  split_dir: /media/data_cifs/Kalpit/NTURGB+D/data/NTURGB+D/cs
  is_training: False
  signals:
    temporal_signal: True
    spatial_signal: True
    all_signal: False

# Model
model: ST_GCONV_RESNET
model_args:
  layers_config: [[64, 64, 1], [64, 64, 1], [64, 64, 1], [64, 128, 2], [128, 128, 1], 
      [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
  num_class: 60
  channel: 15
  window_size: 300
  num_joints: 25
  num_actors: 2
  graph: NTUGraph
  graph_args:
    labeling_mode: 'parts'
  mask_learning: True
  use_data_bn: True

# Optimization
weight_decay: 0.0005
base_lr: 0.1
step: [20, 50, 70]

# Training
device: [0,1,2,3]
batch_size: 64
test_batch_size: 64
num_epoch: 80
nesterov: True
save_interval: 5

# Evaluation
eval_interval: 5

# Initialization from checkpoint
# start_epoch: 5
# weights: /media/data_cifs/Kalpit/NTURGB+D/work_dir/cs/STGCN_parts_noaug/epoch65_model.pt
```

* If the number of GPUs you use is less than 4, please change the list in
the argument "device". Depending on the number of GPUs you have and the
memory they have, you might have to change the batch sizes as well.

* The signals used in the config shown above is temporal (displacement) as
well as spatial (relative coordinates). Hence, the number of input channels
is 15. Refer the paper to understand what these signals are.

* If you want to introduce new signals, add the script that calculates the
new feature in the folder `data/signals`. If you use it, change the number
of input channels. If you remove any signal (by setting it to `False`),
reduce the number of input channels.

* The different ways of constructing the adjacency matrices (labeling mode)
are given in `pb-gcn/models/graph/graph.py`. Change this file / add to this
file for any new way of labeling the nodes.

* For testing the model using saved / downloaded weights, use the test config files given.

That's all. If there are any questions, please create an issue and we can
take it up there!
