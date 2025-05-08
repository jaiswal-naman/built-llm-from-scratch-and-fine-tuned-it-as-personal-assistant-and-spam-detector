**From Zeros to GPT: Building a Large Language Model from Scratch with pytorch** 

We all use ChatGPT and other AI models every day — but have you ever wondered how these powerful Large Language Models (LLMs) are actually built?

In this repository, I’ve gone under the hood to build a ChatGPT-like LLM from scratch, assembling every component from the ground up. From the smallest math operations to the complete working model, everything is explained step-by-step in clean, well-commented Jupyter notebooks.

You won’t just run the model — you'll understand it.

🔩 Fixing every nut and bolt
⚙️ Building the transformer engine
🚀 Training and running a full LLM

Due to the large codebase and complexity, I’ve included a high-level walkthrough right here in the README to help you stay on track and understand the overall architecture as you go through the notebooks.

Let’s dive in and demystify how ChatGPT-like models actually work — one layer at a time.

🔁 Complete Workflow of the ChatGPT Model
The complete flow of how a ChatGPT-like model works is illustrated in the figure below. This diagram breaks down the step-by-step process — from input tokens all the way to generating meaningful, context-aware responses.

![image](https://github.com/user-attachments/assets/f1dc0bb4-16a3-4e94-81cf-e14a7814381d)

----------------------------------------------------------------------------------------------
_During the pretraining stage, LLMs process text one word at a time. Training
LLMs with millions to billions of parameters using a next-word prediction task
yields models with impressive capabilities. These models can then be further fine-
tuned to follow general instructions or perform specific target tasks. But before we
can implement and train LLMs, we need to prepare the training dataset, as illus-
trated in figure 2.1._
![image](https://github.com/user-attachments/assets/abe32846-f4c9-40d1-8a22-7986002015ba)

## 🧹 Stage One: Data Preparation & Sampling

Building a Large Language Model (LLM) from scratch begins with transforming raw text into a format suitable for model consumption. This stage involves several crucial steps:

---

### 1. 🧩 Tokenization: Breaking Text into Units

**What it is:**  
Tokenization involves splitting raw text into smaller units called *tokens*. These tokens can be words, subwords, or characters, depending on the tokenizer used.

**Why it's needed:**  
Models operate on numerical data. Tokenization converts text into manageable pieces that can be mapped to numbers.

**Example:**  
Input: "The cat sat on the mat."  
Tokens: ["The", "cat", "sat", "on", "the", "mat", "."]

---

### 2. 📚 Vocabulary Creation: Mapping Tokens to IDs

**What it is:**  
Creating a vocabulary involves assigning a unique integer ID to each token identified during tokenization.

**Why it's needed:**  
Neural networks require numerical input. Mapping tokens to IDs facilitates this requirement.

**Example:**  
Vocabulary:  
"The": 1, "cat": 2, "sat": 3, "on": 4, "the": 5, "mat": 6, ".": 7

---

### 3. 📏 Positional Encoding: Adding Order Information

**What it is:**  
Positional encoding injects information about the position of each token in the sequence, enabling the model to understand word order.

**Why it's needed:**  
Transformers lack inherent sequence awareness. Positional encodings provide this context.

---

### 4. 🔢 Embedding: Converting Tokens to Vectors

**What it is:**  
Embedding transforms token IDs into dense vectors that capture semantic meaning.

**Why it's needed:**  
Embeddings allow the model to understand relationships between words beyond their IDs.

---

### 5. 🚀 Feeding to the Transformer

**What it is:**  
The embedded vectors, enriched with positional information, are fed into the transformer model for processing.

**Why it's needed:**  
This step initiates the model's learning process, allowing it to analyze and generate language.

---
below diagram describes the complete process visually 
![image](https://github.com/user-attachments/assets/d3bfe9b7-ad81-4f12-9515-1392ff6a6343)
now lets go into the details of each step of above diagram in details:
![image](https://github.com/user-attachments/assets/c4652034-dd7e-4f6e-b5b2-58394493675c)
![image](https://github.com/user-attachments/assets/593255f0-441f-4a95-b138-90ab7059c177)
![image](https://github.com/user-attachments/assets/5b915a49-dabd-4160-96cd-b80dacb3355a)
![image](https://github.com/user-attachments/assets/04982515-3e8a-4742-b8e0-cef7f4bb3c82)
![image](https://github.com/user-attachments/assets/a95d1539-543e-493d-930c-430b32e0236d)
![image](https://github.com/user-attachments/assets/a42f68a2-4026-407f-b2f6-96773e2bc64d)
![image](https://github.com/user-attachments/assets/46fb9920-640b-4655-b8d9-7dfc64e6cc3f)
**In Project , we are using the BPE tokenizer from OpenAI's open-source tiktoken library.**
**Data sampling with a sliding window**__
The next step in creating the embeddings for the LLM is to generate the input–target
pairs required for training an LLM. What do these input–target pairs look like? As we
already learned, LLMs are pretrained by predicting the next word in a text, as depicted
in figure 2.12.
![image](https://github.com/user-attachments/assets/840439f5-4f63-4e53-86b2-4c35b9de35ef)
There’s only one more task before we can turn the tokens into embeddings: implementing an efficient data loader that iterates over the input dataset and returns the inputs and targets as PyTorch tensors, which can be thought of as multidimensional
arrays. In particular, we are interested in returning two tensors: an input tensor con-
taining the text that the LLM sees and a target tensor that includes the targets for the
LLM to predict, as depicted in figure 2.13. While the figure shows the tokens in string
format for illustration purposes, the code implementation will operate on token IDs
directly since the encode method of the BPE tokenizer performs both tokenization
and conversion into token IDs as a single step.
![image](https://github.com/user-attachments/assets/604e25bb-c102-4607-8150-0bd2cdbf9a4e)

At this point,we know how to prepare the input text for training LLMs by splitting
text into individual word and subword tokens, which can be encoded into vector rep-
resentations, embeddings, for the LLM.
Now, we will look at an integral part of the LLM architecture itself, attention
mechanisms, as illustrated in figure 3.1. We will largely look at attention mechanisms
in isolation and focus on them at a mechanistic level.
![image](https://github.com/user-attachments/assets/a81fbb39-3fed-4f63-82fc-4934009ceec4)
We will implement four different variants of attention mechanisms, as illustrated in
figure 3.2. These different attention variants build on each other, and the goal is to arrive at a compact and efficient implementation of multi-head attention that we can
then plug into the LLM architecture
**Self-attention is a mechanism that allows each position in the input sequence to
consider the relevancy of, or “attend to,” all other positions in the same sequence
when computing the representation of a sequence. Self-attention is a key component
of contemporary LLMs based on the transformer architecture, such as the GPT series.
**
![image](https://github.com/user-attachments/assets/2d1ecc89-38db-471c-9aeb-7ca8c8c1bcc2)

A simple self-attention mechanism without trainable weights
Let’s begin by implementing a simplified variant of self-attention, free from any train-
able weights, as summarized in figure 3.7. The goal is to illustrate a few key concepts
in self-attention before adding trainable weights.
![image](https://github.com/user-attachments/assets/988c5641-966e-4cba-a73c-9f810e5b99e5)




