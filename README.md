# AI-Generated Text Detection Competition Report

## Overview
In the landscape of ever-evolving technology, large language models (LLMs) have made significant strides, crafting text that closely mimics human writing. This project participates in a competition aimed at promoting transparent research into AI detection techniques that can be effectively applied in real-world scenarios.

The challenge for participants is to develop a machine learning model capable of distinguishing between essays written by students and those generated by LLMs. The dataset includes a variety of texts from student-written essays to outputs from several LLMs.

## Description
The goal is to create a distinction between student-authored essays and those composed by LLMs. As LLMs become more widespread, there's a concern they might supplant human efforts in areas traditionally dominated by people, like writing. Educators are particularly worried about the impact on student skill development, yet there remains hope that LLMs could become valuable tools for enhancing writing abilities.

LLMs' proficiency in generating text that parallels human writing, thanks to their training on extensive datasets, raises the issue of potential plagiarism—a pressing concern in academic settings. This project aims to identify unique characteristics of LLM-generated text to advance the current methods of AI text detection. By using moderate-length texts across various subjects and multiple, unknown generative models, we simulate typical detection scenarios and encourage the discovery of features that are effective across different models.

This initiative is a collaboration between Vanderbilt University, The Learning Agency Lab—an independent nonprofit based in Arizona, and Kaggle.

## Repository Structure

```plaintext
.
├── dataset
│   ├── concatenated.csv
│   ├── daigt_external_dataset.csv
│   ├── final_test.csv
│   ├── final_train.csv
│   ├── Mistral7B_CME_v7.csv
│   ├── sample_submission.csv
│   ├── test_essays.csv
│   ├── train_1226.csv
│   ├── train_drcat_02.csv
│   ├── train_essays.csv
│   ├── train_prompts.csv
│   └── train_v2_drcat_02.csv
├── demo.py
├── download.sh
├── models
│   ├── glove.6B.200d.txt
│   ├── mistral
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   ├── README.md
│   │   ├── score.pt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.model
│   │   └── training_args.bin
│   ├── pretrained_model.pt
│   ├── sentencepiece_constructions_model.pkl
│   ├── sentencepiece_tokenizer.pkl
│   ├── sequencepiece_glove_model.pkl
│   ├── sequencepiece_tokenizer.pkl
│   └── tfidf_vectorizer.pkl
├── README.md
├── requirements.txt
├── train_bert.py
├── train_glove.py
├── train_llm.py
└── train_tfidf.py
```

## Models Trained
We have trained three distinct model types for this competition:
- Traditional Machine Learning Models - train_tfidf.py, train_glove.py
- Transformer Pretrained Models - train_bert.py
- Large Language Models - train_llm.py

## How to Run

To run the web demo, execute the following command:

```bash
python demo.py
```

This command will launch a Gradio web interface to showcase the models' capabilities in detecting AI-generated text.

## Dependencies

Install the necessary Python packages using the following command:

```bash
pip install -r requirements.txt
```

---

After running `demo.py`, you should be able to interact with the model through a web interface, facilitating an intuitive understanding of the project's achievements.
## References

- [LLM - Detect AI Generated Text - Kaggle](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/overview)
- [AI or Not AI? Delving Into Essays with EDA - Kaggle](https://www.kaggle.com/code/pamin2222/ai-or-not-ai-delving-into-essays-with-eda)
- [DAIGT V2 train dataset - Kaggle](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)
