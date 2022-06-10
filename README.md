## File structure：

```
└─shapelet_anomaly
    │  README.md
    │           
    ├─datasets
    │  ├─Epilepsy
    │  │      
    │  ├─FingerMovements
    │  │      
    │  ├─RacketSports
    │  │      
    │  └─SelfRegulationSCP2
    │          
    ├─data_utils
    │      load_ucr_dataset_by_name.py
    │      plot_data.py
    │      plot_time_series.py
    │      time_series_dataset.py
    │      __init__.py
    │      
    ├─models
    │      config.py
    │      lstm.py
    │      LSTM_AE.py
    │      LSTM_AE_2.py
    │      LSTM_AE_FINAL.py
    │      transformer.py
    │      __init__.py
    │      
    ├─scripts
    │      ae_bench.py
    │      shapelet_bench.py
    │      shapelet_visualization.py
    │      
    ├─shapelets_utils
    │      learning_shapelets.py
    │      transform_shapelets.py
    │      __init__.py
    │      
    └─train_utils
            eval_.py
            train_.py
            __init__.py      
```

| 文件、目录          | 功能                                                         |
| ------------------- | ------------------------------------------------------------ |
| README.md           | 你正打开着的...                                              |
| datasets目录        | 1. 用于存放所有的数据集<br />2. 该目录下分别有实验用到的四个数据集，存放数据集的训练集和测试集的各个维度 |
| data_utils目录      | 数据集读取相关，绘制时间序列（适用于单维）                   |
| models目录          | 用于存放所有使用过的深度学习模型，最终版使用`lstm.py`        |
| scripts目录         | 测试脚本（适用于高维版本）                                   |
| shapelets_utils目录 | 生成shapelet的模型                                           |
| train_utils目录     | 训练函数、评估函数（适用于单维）                             |

## How to use

### 1. 生成shapelet + shapelet转换

运行方法：`python ./scripts/shapelet_bench.py`

修改`./scripts/shapelet_bench.py`文件：

1. 第96行-103行修改转换序列的输出目录和文件名

	```python
	mts_path = os.path.join('data', dataset_name)
	if not os.path.exists(mts_path):
		os.mkdir(mts_path)
	# print(mts_train.shape)
	numpy.save(os.path.join(mts_path, f'x_train_{n}'), X_train)
	numpy.save(os.path.join(mts_path, f'y_train_{n}'), y_train)
	numpy.save(os.path.join(mts_path, f'x_test_{n}'), X_test)
	numpy.save(os.path.join(mts_path, f'y_test_{n}'), y_test)
	```

2. 第105-124行修改模型参数：

    ```python
    # model params
    n_ts, n_channels, len_ts = X_train.shape
    loss_func = nn.CrossEntropyLoss()
    num_classes = len(set(y_train))
    # learn X shapelets of length Y
    shapelet_length = int(len_ts * length)
    shapelet_size = n
    shapelets_size_and_len = {shapelet_length: shapelet_size}
    dist_measure = "euclidean"
    lr = 4e-3
    wd = 1e-4
    epsilon = 1e-7
    batch_size = 128
    shuffle = True
    drop_last = False
    epochs = 200
    n_epoch_steps = 5 * 8
    l1 = 0.2
    l2 = 0.01
    k = int(0.05 * batch_size) if batch_size <= X_train_size else X_train_size
    ```

3. 第194行修改数据集字符串数组：

    ```python
    # you need to input the dataset name in this array
    dataset_names = [
        'SelfRegulationSCP2',
        'FingerMovements',
        'RacketSports',
        'Epilepsy']
    ```

4. 第202-204行修改shapelet的数量和长度，数量默认范围[20, 60]，长度为原始时序长度的0.05倍：

    ```python
    for name in dataset_names:
        for num in range(20, 60):
            fit(name, num, 0.05)
    ```

### 2. 异常检测

运行方法：`python ./scripts/ae_bench.py`

修改`./scripts/ae_bench.py`文件：

1. 第102-106行修改读取的转换序列文件名

	```python
	dataset_name = 'Epilepsy'
	numas = 40  # number of shapelets
	mts_path = os.path.join('data', dataset_name)
	train_dataset_x = np.load(os.path.join(mts_path, f'x_train_{numas}.npy'))
	train_dataset_y = np.load(os.path.join(mts_path, f'y_train_{numas}.npy'))
	```

2. 第147行修改`train_model`方法；

3. 第196行修改随机种子，默认为10的n次方

## Credits

The code was developed with the aid of the following libraries:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)
