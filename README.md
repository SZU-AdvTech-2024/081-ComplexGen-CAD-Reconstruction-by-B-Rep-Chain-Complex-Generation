# 模型文件
模型文件存放在了`../deepcad_whole_test_v4`文件夹下，文件夹结构如下：
```
../deepcad_whole_test_v4
│
└─── 00000093
│   │
│   └─── data.npz
│   │
|   └─── mesh.ply
|        
└─── 00000797
│   │
│   └─── data.npz
│   │
|   └─── mesh.ply
│
└─── ...
```

其中`data.npz`为处理后模型数据，`mesh.ply`为来自ABC数据集的GT模型数据

# 运行
由于Gurobi库的运行需要，必须修改`GlobOptimize.py`中`option`变量内的Gurobi证书信息，一台电脑或一个Docker container只能有一个证书,如：
```
options = {
    "WLSACCESSID": "65efab4a-0464-4cdb-aeba-30565a53583b",
    "WLSSECRET": "ea97b187-f36a-4aec-beb8-b672f7568f76",
    "LICENSEID": 2538923,
}
```

证书信息修改完成后，运行命令即可开始优化:

```
python GlobOptimize.py --model_folders /path/to/model/folders
```
