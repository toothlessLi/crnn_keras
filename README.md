# CRNN-Keras
A simple CRNN training and test system

## Install

**step 1**
```
clone the project
git clone https://github.com/toothlessLi/crnn_keras.git --recursive
```
**step 2**
```
install python requirements
pip install -r requirements.txt
```

**step 3**
```
install warp-ctc for tensorflow
cd 3rd_party
mkdir build
cd build
cmake ..
make 

cd -
cd tensorflow-binding
export TENSORFLOW_SRC_PATH=/path/to/tensorflow 
# example: export TENSORFLOW_SRC_PATH=/home/username/bin/python3/lib/python3.5/site-packages/tensorflow
python setup.py install
```

## Dataset

Download synthetic chinese string dataset from Baidu Yun: https://pan.baidu.com/s/1IhAx4oogNSGJ1Kxzc_QfPg&shfl=sharepset password: t96p

For this project, you need prepare file_list.txt, the format of each line as bellow.
```
/home/path/to/dataset/images/xxxxxx.jpg xxxxx
in python code it is: path_to_image + '\t' + label_string + '\n'
```

## Usage

### Training
```
python main.py train --config ./configs/crnn_train.yaml
```

### Test
```
python main.py test --config ./configs/crnn_test.yaml
```

### Inference
```
python main.py inference --config ./configs/crnn_test.yaml --img ./data/demo.jpg --viz
```