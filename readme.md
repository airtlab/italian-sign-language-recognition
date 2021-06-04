# Italian sign language alphabet recognition from surface EMG and IMU sensors with a deep neural network

This repository contains the source code of the experiments presented in the paper

>P. Sernani, I. Pacifici, N. Falcionelli, S. Tomassini, and A.F. Dragoni, *Italian sign language recognition from surface electromyography and inertial measurement unit sensors with a deep neural network*.

The paper is published in the Proceedings of the 4th International Conference Recent Trends and Applications In Computer Science And Information Technology (RTA-CSIT 2021), which can be found at the following link:

><http://ceur-ws.org/Vol-2872/>

Specifically, the source code is contained in a Jupyter notebook, which is available in the “notebook” directory of this repository.

The experiments are accuracy tests on the classification of the gestures of the Italian Sign Language (LIS) alphabet, based on surface electromyography (EMG) and Inertial Measurement Unit data collected with the Myo Gesture Control Armband. The classification is performed with a deep neural network, based on the Bidirectional Long Short Term Memory (Bi-LSTM) architecture.

The experiments were run on Google Colab, using the GPU runtime and Keras 2.4.3, with the TensorFlow 2.4.1 backend, and scikit-learn 0.22.2.post1.

You can directly check the notebook in this repository, or open it in Google Colab by clicking on the following badge.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/airtlab/italian-sign-language-recognition/blob/master/notebook/Italian_Sign_Language_Recognition_with_EMG_and_IMU_data.ipynb)

## Data Description

The experiments are based on the EMG and IMU data available in the following GitHub repository

><https://github.com/airtlab/An-EMG-and-IMU-Dataset-for-the-Italian-Sign-Language-Alphabet>

The dataset contains 30 gesture samples for each letter of the LIS alphabet, with time series of 8 EMG-sensors readings the IMU readings collected in a 2 seconds time window at 200 Hz. The full specification of the dataset is available in the repository and in a dedicated open-access data paper, which can be found at the following link

><https://www.sciencedirect.com/science/article/pii/S2352340920313378>.

With the twofold objective of using more training data and trying to prevent overfitting, the dataset was synthetically augmented in some of the tests, by simulating a rotation of the Myo Armband worn by the users. The data augmentation formulation is available in the paper.

## Neural Network architecture

The layers of the implemented neural network are listed in the following table.

| Layer Type                                     | Output Shape     | Parameter # |
|------------------------------------------------|:----------------:|------------:|
| Bi-LSTM, *64 hidden units*, *return sequences* | (None, 400, 128) |       40448 |
| Bi-LSTM, *32 hidden units*                     | (None, 64)       |       41216 |
| Dropout, *0.5*                                 | (None, 64)       |           0 |
| Dense, *64 hidden units*, *ReLU activation*    | (None, 64)       |        4160 |
| Dropout, *0.5*                                 | (None, 64)       |           0 |
| Dense, *26 hidden units*, *Softmax activation* | (None, 26)       |        1690 |

## Experiments

The notebook contains four experiments, based on the Stratified Shuffle Split strategy to split the available data in 70% for training, 10% for validation, and 20% for testing. All the experiments include a classification report and a confusion matrix for each split of the dataset, an the mean value of the accuracy score is computed as well. Specifically, the four experiments are based on:
1. the repetition of 5 randomized splits on the original dataset;
2. the repetition of 5 randomized splits on the augmented dataset;
3. the repetition of 30 randomized splits on the original dataset;
4. the repetition of 30 randomized splits on the augmented dataset.

## Source code release agreement

The source code of the experiments is freely released for research and educational purposes. Please cite as
- P. Sernani, I. Pacifici, N. Falcionelli, S. Tomassini, A. F. Dragoni, Italian sign language alphabet recognition from surface EMG and IMU sensors with a deep neural network, in: Proceedings of the 4th International Conference on Recent Trends and Applications in Computer Science and Information Technology, volume 2872 of CEUR Workshop Proceedings, 2021, pp. 74–83.
	 
Bibtex entry:

	 @inproceedings{Sernani2021,
	  author={Sernani, Paolo and Pacifici, Iacopo and Falcionelli, Nicola and Tomassini, Selenete and Dragoni, Aldo Franco},
	  title={Italian sign language alphabet recognition from surface {EMG} and {IMU} sensors with a deep neural network},
	  booktitle={Proceedings of the 4th International Conference on Recent Trends and Applications in Computer Science and Information Technology},
	  series={CEUR Workshop Proceedings},
	  year={2021},
	  volume={2872},
	  pages={74-83},
	  url={http://ceur-ws.org/Vol-2872/paper11.pdf},
	 }

The paper is open access and available here <http://ceur-ws.org/Vol-2872/paper11.pdf>.

