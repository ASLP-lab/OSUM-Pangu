<p align="center">
   <h1>I-OSUM-Pangu: Intent-Aware Open-Source Speech Understanding Framework</h1>
<p>

Yujie Liao, Xuelong Geng, Shuiyuan Wang, Lei Xie

<p align="center">
    <img src="images/I-OSUM-Pangu.png" width="400"/>
<p>

<p align="center">
    <a href="https://huggingface.co/ASLP-lab/I-OSUM-Pangu"> Ckpt</a>
</p>

In recent years, the development of large-scale audio-language models has enabled multi-dimensional speech understanding. However, most existing open-source models rely on fixed templates or task tags, while more powerful systems are often closed-source or require massive amounts of training data.

We propose **I-OSUM-Pangu**, an efficient, controllable, and fully open-source speech understanding framework.

The model is built upon:

- Whisper-medium speech encoder (from the Whisper series developed by :contentReference[oaicite:0]{index=0})
- :contentReference[oaicite:1]{index=1} 7B large language model backbone

The core objective of our framework is to enable the model to:

- Understand user instructions expressed in natural language  
- Automatically identify user intent  
- Route the request to the corresponding speech understanding task  
- Work without relying on fixed prompt templates  

Experimental results show that:

- The Instruction Following Rate (IFR) exceeds **90%**
- While maintaining comparable task performance with traditional fixed-tag approaches

This project releases both code and model weights, aiming to provide a **reproducible and extensible open-source framework** for speech understanding research.

---

## Architecture

The overall architecture of I-OSUM-Pangu is shown below:

<p align="center">
    <img src="images/structure.png" width="80%"/>
<p>

The model mainly consists of three components:

### 1. Speech Encoder
Whisper-medium  
Responsible for extracting speech representations.

### 2. Adapter
Transforms acoustic features into tokens compatible with the LLM input space.

### 3. Intent-aware LLM
OpenPangu-7B

Responsible for:
- Parsing natural language instructions
- Identifying user intent
- Determining which speech task to execute

---

## Training Strategy

We propose a **Decoupled-then-Integrated Training Strategy**, illustrated below:

<p align="center">
    <img src="images/Strategy.png" width="80%"/>
<p>

### Stage 1: Speech Understanding Alignment

Goal: Equip the model with multi-task speech understanding capability.

Characteristics:

- Only speech-related modules are trained
- Establish strong acoustic representation ability

---

### Stage 2: Intent Understanding

Goal: Enable the model to understand natural language user instructions.

Examples:

Please transcribe this audio.  
Analyze the speaker's emotion.  
Identify what event happens in the audio.

The model learns:

- Instruction semantic understanding
- Task mapping capability

---

### Stage 3: Joint Instruction Tuning

In the final stage, joint training allows the model to:

- Automatically parse user instructions
- Identify task types
- Execute the corresponding speech understanding tasks

Without requiring fixed templates, such as:

What is the emotion of this speech?  
Can you transcribe this audio?  
What event happens in the audio?

The model can correctly understand and execute all of them.

---

## Inference Results

### Dataset Configuration

The model is trained on **47,000 hours** of multi-task speech data, covering seven core speech tasks. Additionally, a dedicated dataset is constructed to enhance instruction-following ability.

<p align="center">
    <img src="images/table1.png" width="65%"/>
</p>

---

### Instruction Following Performance (IFR)

Instruction Following Rate (IFR) measures the ability of the model to parse natural language instructions and execute the corresponding tasks.

The metric is defined as:

$$
IFR = \left( \frac{N_{correct}}{N_{total}} \right) \times 100\%
$$

where:

- $N_{correct}$ represents the number of correctly executed instructions  
- $N_{total}$ represents the total number of evaluation samples
Compared with mainstream open-source models, **I-OSUM-Pangu achieves significantly better performance**:

<p align="center">
    <img src="images/table2.png" width="65%"/>
</p>

---

### Flexibility vs Accuracy

We evaluate whether natural language instructions (NL) degrade performance compared to fixed instructions (FI).

Results show that the model maintains strong flexibility while preserving task accuracy.

<p align="center">
    <img src="images/table3.png" width="65%"/>
</p>

Conclusion:

Only minor performance drops appear in relatively niche tasks such as:

- Style recognition
- Event detection

Core tasks such as:

- ASR
- SER
- SAP

remain almost unchanged, validating the effectiveness of the **Decoupled-then-Integrated strategy**.

---

### Multi-task Speech Understanding Performance

On public benchmarks, the model demonstrates competitive performance across multiple tasks, particularly in:

- Age prediction
- Emotion recognition (MER2023)

<p align="center">
    <img src="images/table4.png" width="65%"/>
</p>

---

### Speech-to-Text Chat (STTC) Capability

We further evaluate the model in conversational reasoning scenarios.

I-OSUM-Pangu outperforms GLM-4-Voice on the TriviaQA and WebQ benchmarks.

<p align="center">
    <img src="images/table5.png" width="65%"/>
</p>

---

### Ablation Study: Importance of the Decoupled Training Strategy

We compare direct joint training with our decoupled-then-integrated strategy to verify the effectiveness of our core design.

<p align="center">
    <img src="images/table6.png" width="65%"/>
</p>

Conclusion:

Text-domain intent pretraining (Stage 2) establishes a strong semantic prior for the model and is crucial for improving instruction-following stability.

---

## How to Use the I-OSUM-Pangu Framework for Training and Inference

### Environment Setup

Before starting, please ensure that your device supports **NPU** and the Python environment is properly configured.

We recommend running the code on a Linux system.

If Conda is not installed, please refer to:
https://blog.csdn.net/qq_41636123/article/details/130266232

```bash
# Create a new conda environment
conda create -n iosum python=3.10
conda activate iosum

# Clone the repository
git clone https://github.com/ASLP-lab/I-OSUM-Pangu.git
cd I-OSUM-Pangu

# Install dependencies
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Model Download
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ASLP-lab/I-OSUM-Pangu",
    local_dir="path",
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com"
)
```
### Inference
This project provides batch inference scripts for all tasks under in ：I-OSUM-Pangu/infer_code:

```shell
python infer_ASR.py
```
### Training
To ensure a smooth training process, please follow the steps below.
#### 1. Data Preparation
Data can be prepared in three formats:

raw、shard、combine

Recommended: shard format

After preparing the dataset, write the generated data index into the following configuration file:
```yaml
I-OSUM-Pangu/conf/data_s2t_tmp.yaml
```
#### 2. Start Training

Run the main training script:
```bash
I-OSUM-Pangu/train.sh
```