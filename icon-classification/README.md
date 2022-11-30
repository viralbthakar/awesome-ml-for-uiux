# Icon Classification


## Dataset
- Download the `data.zip` file from original project repository [semantic-icon-classifier](https://github.com/datadrivendesign/semantic-icon-classifier).
- Move `data.zip` to `data` directory.
- Open terminal and execute following set of commands.
```
unzip -j data.zip data/training_x.npy data/training_y.npy data/validation_x.npy data/validation_y.npy data/validation_metadata.json
```
- Rename files for appropriate representation. Even  though files are named as `validation_*.npy` in our implementation we treat them as `Test Data`. Validation data is randomly sampled from training data. 
```
mv validation_x.npy test_x.npy
mv validation_y.npy test_y.npy
mv validation_metadata.json metadata.json
```

## Training
```
python main.py \
--exp-id "vgg16-imagenet" \
--training-data \path\to\training_x.npy \
--training-labels \path\to\training_y.npy \
--test-data \path\to\test_x.npy \
--test-labels \path\to\test_y.npy \
--metadata \path\to\metadata.json \
--artifacts \path\to\artifacts
```