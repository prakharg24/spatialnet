# Spatialnet

## Downloading Imagenet dataset

Simply run the following to download the Imagenet dataset. The sudo privileges are required as additional packages needs to be installed to enable multi-threading. This can take upto a day to completely download the dataset.

```
sudo sh download_dataset.sh
```

Note : If you already have the Imagenet dataset downloaded elsewhere, you can skip the step above and simply link the relevant tar files inside this repo. Follow the steps below for the same
1. Create a folder `data` and inside that create folder `imagenet`
2. Inside the `imagenet` folder, there should be two tar files present, `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`

## Extracting and processing Imagenet dataset

The following script will extract the tar files and split the data into relevant folders, so that it is easily accessible for the training code.

```
sudo sh process_dataset.sh
```

Note : If you already have the Imagenet dataset extracted into folders, you can simply link them here. The expected directory structure is very straightforward.

```
train/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ......
├── ......
val/
├── n01440764
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   ├── ......
├── ......
```

## Installing relevant features and environment

1. Create a new conda environment
```
conda create -n spatialnet python=3.6 -y & conda activate spatialnet
```
2. Install pytorch
```
conda install -c pytorch pytorch -y
```
3. Install other dependencies
```
conda install attrs numpy torchvision pillow tqdm -y
```

## Start training

### Understanding Training configuration flags
Training can be done using the `main_train_scratch.py` file as below. The model name can be `ResNet50`, `EfficientNet:n` or `SpatialNet:n`, where n is an integer between 0 to 7.
```
python main_train_scratch.py --architecture <model_name> --job-id some_id --batch 64 --num-tasks 8 --learning-rate 2e-1
```

Note that the num-tasks flag represents the number of GPUs available on the machine and the batch size represents the batch size on each GPU. For different configuration of the machine, adjust the num-tasks accordingly. **Make sure that the num-tasks are set properly according to the number of available GPUs on the machine, otherwise the code will be stuck (without errors) and won't train.** If possible, make sure the num-tasks * batch product remains constant. For example, on a machine with 4 GPUs, run the following if enough memory is available per GPU.
```
python main_train_scratch.py --architecture <model_name> --job-id some_id --batch 128 --num-tasks 4 --learning-rate 2e-1
```

If the total batch size (num-tasks * batch) is not 512, learning rate also needs to be adjusted. The following formula can be used
```
learning-rate = 0.1*(Total Batch Size)/256
```

### Training ResNet50

First train Resnet-50 on Imagenet to check the sanity of the code and replicate the expected accuracy. Run the command below to execute training and output the results in a logfile.
```
python -u main_train_scratch.py --architecture ResNet50 --job-id resnet --batch 64 --num-tasks 8 --learning-rate 2e-1 > logfile_resnet50.txt
```

### Training EfficientNet

After we are sure the code runs properly, by verifying the ResNet50 accuracy, we can shift to EfficientNet training to see if the same parameters work properly for this too. Run the command below to execute training and output the results in a logfile.
```
python -u main_train_scratch.py --architecture EfficientNet:0 --job-id efficientnet_0 --batch 64 --num-tasks 8 --learning-rate 2e-1 > logfile_efficientnet0.txt
```

### Training SpatialNet

Finally run our own model to compare with EfifcientNet. Run the command below to execute training and output the results in a logfile.
```
python -u main_train_scratch.py --architecture SpatialNet:0 --job-id spatialnet_0 --batch 64 --num-tasks 8 --learning-rate 2e-1 > logfile_spatialnet0.txt
```
