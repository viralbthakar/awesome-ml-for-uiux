import os
import argparse
import numpy as np
import tensorflow as tf
from trainer import Trainer
from model import ModelBuilder
from datapipeline import DataPipeline
from utils import load_npy_files, extract_data_from_json, create_dir, plot_distribution, create_validation_split

parser = argparse.ArgumentParser(
    description='Icon Classification Dataset Creation')
parser.add_argument('--exp-id', type=str, required=True,
                    help="Unique ID for experiment.")
parser.add_argument('--training-data', type=str, required=True,
                    help="Path to training data npy file.")
parser.add_argument('--training-labels', type=str, required=True,
                    help="Path to training labels npy file.")
parser.add_argument('--test-data', type=str, required=True,
                    help="Path to test data npy file.")
parser.add_argument('--test-labels', type=str, required=True,
                    help="Path to test labels npy file.")
parser.add_argument('--metadata', type=str, required=True,
                    help="Path to metadata json file.")
parser.add_argument('--artifacts', type=str, required=True,
                    help="Path to save artifacts.")

parser.add_argument('--batchsize', type=int, default=32,
                    help="Batch size for data pipeline.")
parser.add_argument('--epochs', type=int, default=1000,
                    help="Maximum number of training epochs.")
parser.add_argument('--patience', type=int, default=20,
                    help="Patience amount for early stopping.")
parser.add_argument('--learning-rate', type=float, default=1e-4,
                    help="Learning rate")

parser.add_argument('--loss-id', type=str, default='cce',
                    choices=['cce'],
                    help="Loss ID.")
parser.add_argument('--optimizer-id', type=str, default='adam',
                    choices=['adam'],
                    help="Optimizer ID.")
parser.add_argument('--metrics-id', type=str, nargs='+',
                    default=['f1', 'binary-accuracy', 'precision', 'recall',
                             'tp', 'tn', 'fp', 'fn', 'auc', 'prc'],
                    choices=['f1', 'binary-accuracy', 'precision', 'recall',
                             'tp', 'tn', 'fp', 'fn', 'auc', 'prc'],
                    help="Metrics ID.")

parser.add_argument('--feature-ext', type=str, default='vgg16',
                    choices=['vgg16', 'vgg19', 'resnet50',
                             'mobilenetv1', 'mobilenetv2'],
                    help="Feature extractor to use.",)
parser.add_argument('--global-pooling', type=str, default='avg',
                    choices=['avg', 'max'],
                    help="Path to test labels npy file.")
parser.add_argument('--weights', type=str, default='imagenet',
                    choices=['imagenet'],
                    help="Weights to initialize the model.")

opt = parser.parse_args()

artifacts = create_dir(opt.artifacts)
eda_dir = create_dir(artifacts, "eda")
tensorboard_dir = create_dir(os.path.join(
    artifacts, "tensorboard"), opt.exp_id)
checkpoint_dir = create_dir(os.path.join(artifacts, "checkpoints"), opt.exp_id)
csv_logger = os.path.join(create_dir(os.path.join(
    artifacts, "csvlogs"), opt.exp_id), "logs.csv")

class_names = extract_data_from_json(opt.metadata, key="class_names")
print("# - "*5, f"{os.path.basename(opt.metadata)}", " - #"*5)
print(f"Found {len(class_names)} in the dataset.")

data = load_npy_files([opt.training_data, opt.training_labels], summary=True)
training_data, training_label = data["data"]
training_label = np.squeeze(training_label)
plot_distribution(x=data["unique"][0], y=data["counts"][0], title="Training Data Distribution Plot", figsize=(
    24, 24), dpi=300, save_flag=True, file_path=os.path.join(eda_dir, "training_data.png"), orient='v')
plot_distribution(x=data["counts"][1], y=class_names, title="Training Labels Distribution Plot", figsize=(
    24, 24), dpi=300, save_flag=True, file_path=os.path.join(eda_dir, "training_labels.png"), xticks_ct=1)

data = load_npy_files([opt.test_data, opt.test_labels], summary=True)
test_data, test_label = data["data"]
test_label = np.squeeze(test_label)
plot_distribution(x=data["unique"][0], y=data["counts"][0], title="Test Data Distribution Plot", figsize=(
    24, 24), dpi=300, save_flag=True, file_path=os.path.join(eda_dir, "test_data.png"), orient='v')
plot_distribution(x=data["counts"][1], y=class_names, title="Test Labels Distribution Plot", figsize=(
    24, 24), dpi=300, save_flag=True, file_path=os.path.join(eda_dir, "test_labels.png"), xticks_ct=1)

training_data_generator = DataPipeline(
    training_data, training_label, num_classes=len(class_names)).create_pipeline(opt.batchsize)
training_data_generator, validation_data_generator = create_validation_split(
    training_data_generator, validation_split=0.2, shuffle=True, seed=1)

test_data_generator = DataPipeline(
    test_data, test_label, num_classes=len(class_names)).create_pipeline(opt.batchsize)

image, label = next(iter(training_data_generator))
print(image.shape)
print(label)

model_config = {
    "feature_extractor_id": opt.feature_ext,
    "global_pooling_type": opt.global_pooling,
    "layers": [
        {"num_nodes": 128, "activation": "relu"},
        {"num_nodes": 64, "activation": "relu"},
        {"num_nodes": 32, "activation": "relu"}
    ],
    "output_activation": 'softmax',
    "weights": opt.weights,
}

training_config = {
    "epochs": opt.epochs,
    "loss_id": opt.loss_id,
    "optimizer_id": opt.optimizer_id,
    "learning_rate": opt.learning_rate,
    "metrics_id": opt.metrics_id,
    "tensorboard_dir": tensorboard_dir,
    "checkpoint_dir": checkpoint_dir,
    "csv_logger": csv_logger,
    "patience": opt.patience,
    "class_names": class_names,
}

model = ModelBuilder(
    input_shape=training_data.shape[1:],
    output_shape=len(class_names),
    model_config=model_config
).create_model_architecture()

model, history = Trainer(model, training_config).launch_trainer(
    train_dg=training_data_generator,
    validation_dg=validation_data_generator,
    test_dg=test_data_generator)
