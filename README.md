# Bursting the Filter Bubble: Fairness-Aware Network Link Prediction
This repository provides a reference implementation of FLIP and greedy post processing algorithms as described in [1].

## How to use:

The file main.py contains the example code to use flip and greedy post-processing algorithms. Each algorithm expects as input the protected feature value(community membership), training graph, and negative/positive link examples. The input files for Dutch School friendship network and facebook can be fined in ./data/graph/ directory in pickle 3 binary format. 


To run the algorithm on Dutch School friendship network, execute the following command from the project home directory:
```shell-script
python main.py
```
you can set the baseline algoirthm as input to greedy post-processing using --algorithm option, for example:

```shell-script
python main.py --algorithm jac
```

you can change the network to facebook as follow:

```shell-script
python main.py --file-name facebook.pk
```
you can check other available options by following command

```shell-script
python main.py --help
usage: main.py [-h] [--folder-path [FOLDER_PATH]] [--file-name [FILE_NAME]]
               [--walk-path [WALK_PATH]] [--embedding-path [EMBEDDING_PATH]]
               [--log-path [LOG_PATH]] [--dimensions DIMENSIONS]
               [--epochs EPOCHS] [--window-size WINDOW_SIZE]
               [--len-of-walks LEN_OF_WALKS] [--batch-size BATCH_SIZE]
               [--test-size TEST_SIZE] [--number-of-walks NUMBER_OF_WALKS]
               [--learning-rate LEARNING_RATE] [--beta-g BETA_G]
               [--beta-d BETA_D] [--beta-l BETA_L] [--cuda CUDA]
               [--report-acc REPORT_ACC] [--opt [OPT]] [--sparse SPARSE]
               [--mini-batchs-lp MINI_BATCHS_LP] [--log-file [LOG_FILE]]
               [--algorithm [ALGORITHM]] [--change_percent CHANGE_PERCENT]
               [--file FILE]

Run fair gan.

optional arguments:
  -h, --help            show this help message and exit
  --folder-path [FOLDER_PATH]
                        path to input graph directory
  --file-name [FILE_NAME]
                        data file name
  --walk-path [WALK_PATH]
                        path to generated walk folder
  --embedding-path [EMBEDDING_PATH]
                        path to generated embedding folder
  --log-path [LOG_PATH]
                        path to input logs directory
  --dimensions DIMENSIONS
                        Number of dimensions. Default is 128.
  --epochs EPOCHS       Number of gradient descent iterations. Default is 3.
  --window-size WINDOW_SIZE
                        Skip-gram window size. Default is 10.
  --len-of-walks LEN_OF_WALKS
                        length of random walks. Default is 80.
  --batch-size BATCH_SIZE
                        minibach size. Default is 32.
  --test-size TEST_SIZE
                        link prediction test size. Default is 0.8 .
  --number-of-walks NUMBER_OF_WALKS
                        number of random walks. Default is 10
  --learning-rate LEARNING_RATE
                        Gradient descent learning rate. Default is 0.001.
  --beta-g BETA_G       generator hyperparameter. Default is 0.9
  --beta-d BETA_D       discriminator hyperparameter. Default is 0.1
  --beta-l BETA_L       link prediction hyperparameter for integerated
                        version. Default is 0.5
  --cuda CUDA           to use gpu set it 1. Default is 0
  --report-acc REPORT_ACC
                        report acc during embedding training. Default is 0
  --opt [OPT]           choose optimization algorithm. Can be 'adam' or
                        'adagrad'. Default is adam
  --sparse SPARSE       choose whether to use sparse or dense tensors. Default
                        is dense
  --mini-batchs-lp MINI_BATCHS_LP
                        choose whether to use sparse or dense tensors. Default
                        is dense
  --log-file [LOG_FILE]
                        log file name. Default is log.txt
  --algorithm [ALGORITHM]
                        name of link prediciton baseline algorithm to use: jac
                        for Jacard, adar for adamic_adar and prf for
                        preferential_attachment. Default is n2v.
  --change_percent CHANGE_PERCENT
                        Greedy post-processing hyper-parameter. Percent of
                        predictions that can be changed to reduce modularity.
                        Default 0.03.
  --file FILE
```
### Refrence
[1] **Bursting the Filter Bubble: Fairness-Aware Network Link Prediction** F Masrour, T Wilson, H Yan, P Tan, A Esfahanian, in Association for the Advancement of Artificial Intelligence (AAAI), 2020.
