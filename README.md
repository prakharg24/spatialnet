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

## Installing relevant features and environment
