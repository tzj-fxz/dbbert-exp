# Baseline of dbbert for Xinyi Zhang's paper
- 关于脚本：路径为src/run/new_version_multi_doc.py
  - 训练模式：在仓库目录下使用命令PYTHONPATH='src' python3 src/run/new_version_multi_doc.py config/ms_tpch_manydocs_masked.ini > train.log
  - 测试模式：在仓库目录下使用命令PYTHONPATH='src' python3 src/run/new_version_multi_doc.py config/ms_tpch_manydocs.ini > test.log
  - 使用脚本前，请确保mysqld未在运行

- 关于ini文件：

  - 一些共有参数：

    - device：cpu或cuda，建议在允许的情况下改成cuda加快模型更新速度
    - nr_episodes：即iteration数
    - timeout_s：运行时间限制
    - mode：masked表示训练，unmasked表示测试
    - recovery_cmd：请忽略这个参数，这是dbbert自带的用于作者pg数据库的重启命令
    - workload_time：每个sysbench workload运行的时间 

  - ms_tpch_manydocs_masked.ini：用于训练

    - input：待训练的bert模型的路径（均为huggingface官网上下载的模型，我存在本地，如果网好，也可以直接连接到huggingface在线获取模型）
    - output：训练完成后输出的bert模型路径，文件夹里面包括一个config。json和pytorch_model.bin模型

  - ms_tpch_manydocs.ini：用于测试
        model_path：用于测试的bert模型的路径，可以与上面的output相同以使用刚才训练好的模型

    

    


# dbbert
DB-BERT is a database tuning tools that exploits natural language text as additional input. It extracts recommendations for database parameter settings from tuning-related text via natural language analysis. It optimizes parameter settings for a given workload and performance metric using reinforcement learning.

# Quickstart
- DB-BERT was tested on an Amazon EC2 p3.2xlarge instance using Deep Learning AMI (Ubuntu 18.04) Version 43.0.
- Install Postgres 13.2 and the TPC-H benchmark database.
- Install required Python packages, including Huggingface Transformers, stable-baselines-3, psycopg2, Numpy, Pytorch, and streamlit.
- DB-BERT uses configuration files, an example file can be found under `config/pg_tpch_manydocs.ini`.
- For a first try, update credentials in the `[DBMS]` section. You may adapt tuning time and other parameters in the `[BENCHMARK]` section.
- Execute `src/run/tuning_run.py`, passing the configuration file as first (and only) argument, e.g. (from main directory):
`PYTHONPATH='src' python3 src/run/tuning_run.py config/pg_tpch_manydocs.ini`
- DB-BERT will parse input text (text from 100 Web documents), initialize the RL algorithm, and start tuning Postgres for TPC-H.
- Try `config/pg_tpch_onedoc.ini` for tuning using a single, TPC-H specific text document (forum discussion on TPC-H tuning).

# GUI
- To start the GUI, run `streamlit run src/run/interface.py` from the root directory.
- If accessing DB-BERT on a remote EC2 server, make sure to enable inbound traffic to port 8501.
- Enter the URL shown in the console into your Web browser to access the interface.
- You can select settings to read from configuration files in the `demo_configs` folder.
- Select a collection of tuning text documents for extraction (e.g., from the `demo_docs` folder).
- You may change parameter related to database access, learning, and tuning goals.
- Click on the `Start Tuning` button to start the tuning process.

# Ongoing Work
- Switching language models used from BERT to BART (despite the project name).
- Refined text document pre-processing: extract most relevant passage for each parameter.

# How to Cite
A video talk introducing the vision behind this project is [available online](https://youtu.be/Spa5qzKbJ4M).
```
@inproceedings{trummer2022dbbert,
  title={DB-BERT: a database tuning tool that "reads the manual"},
  author={Trummer, Immanuel},
  booktitle={SIGMOD},
  year={2022}
}

@article{trummer2021case,
  title={The case for NLP-enhanced database tuning: towards tuning tools that" read the manual"},
  author={Trummer, Immanuel},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={7},
  pages={1159--1165},
  year={2021},
  publisher={VLDB Endowment}
}
```
