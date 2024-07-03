## DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task

### 📋 Introduction

This repository contains the code for our paper [DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task](https://arxiv.org/pdf/2406.16332). 

Recently, there has been increasing interest in applying large language models (LLMs) as zero-shot passage rankers. However, **few studies have explored how to select appropriate in-context demonstrations for the passage ranking task, which is the focus of this paper**. Previous studies mainly apply a demonstration retriever to retrieve demonstrations and use top-k demonstrations for in-context learning (ICL). Although effective, this approach overlooks the dependencies between demonstrations, leading to inferior performance of few-shot ICL in the passage ranking task. In this paper, we formulate the demonstration selection as a *retrieve-then-rerank* process and introduce the DemoRank framework. In this framework, we first use LLM feedback to train a demonstration retriever and construct a novel dependency-aware training samples to train a demonstration reranker to improve few-shot ICL. The construction of such training samples not only considers demonstration dependencies but also performs in an efficient way. Extensive experiments demonstrate DemoRank's effectiveness in in-domain scenarios and strong generalization to out-of-domain scenarios.

![image-20240621121733997](https://8421bcd.oss-cn-beijing.aliyuncs.com/img/image-20240621121733997.png)

### 📝 How to reproduce the experimental results?

#### Prerequisites

##### 📦 Main packages

- pytorch 2.1.0
- pyserini 0.20.0
- transformers 4.31.0
- accelerate 0.24.1
- deepspeed 0.8.3
- faiss-gpu

##### 📥 Download Data

Please download the required data for code running to the root directory of the project (`demorank/`). The download link is: [Google Drive](https://drive.google.com/drive/folders/1oPOCMIq491pUrnHW2Ivw793ZUfZtv_3R?usp=sharing).

#### 1 Demonstration Retriever (DRetriever)

**1.1 Get Training Data**

We use Flan-T5-XL to score each demonstration candidate. Please run the following command:

```bash
bash llm_score.sh
```

Remember to replace the parameter `model_name_or_path` with your LLM's path. The code defaults to handling the MS MARCO dataset. Please modify the dataset parameters if you are processing other datasets.

**1.2 Model Training**

After get the training data, please run the following script to trained the DRetriever and build the index based on trained models. Ensure that your current path is: `demorank/train/src`. Remember to change parameter `train_data_shards_file` and `model_name_or_path` in the script to your own path.

```bash
bash train_retriever.sh
```

 After model training, the script will automatically execute the indexing script based on the model that was just saved. 

#### 2 Demonstration Reranker (DReranker)

**2.1 Get Training Data**

We use Flan-T5-XL to score demonstration candidates retrieved by DRetriever in a dependency-aware manner . Please run the following command:

```bash
bash llm_score_dependency_aware_rerank.sh
```

**2.2 Model Training**

After get the training data, please run the following script to trained the DReranker. Ensure that your current path is: `demorank/train/src`. Remember to change parameter `train_data_shards_file` and `model_name_or_path` in the script to your own path.

```bash
bash train_reranker.sh
```

#### 3 Inference

Run the following script to evaluate the trained DRetriever and DReranker. Remember to change parameter `model_name_or_path`. Note that the folder `ranking_trunc_queries_docs` used in run_kshot.py is not included in the repository due to file size issue. We provide it at `https://drive.google.com/drive/folders/1IEpFtRpAN3jflj1MXBHgNO3Be29TR2XN?usp=sharing`.

```bash
bash run_kshot.sh
```

### 📞 Contact

If you have any question or suggestion related to this project, feel free to open an issue or pull request. You also can email Wenhan Liu ([lwh@ruc.edu.cn](mailto:lwh@ruc.edu.cn)).

### ✨ Citation

If you find this repository useful, please consider giving a star ⭐ and citation

```
@article{liu2024demorank,
  title={DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task},
  author={Liu, Wenhan and Zhu, Yutao and Dou, Zhicheng},
  journal={arXiv preprint arXiv:2406.16332},
  year={2024}
}
```