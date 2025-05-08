![ChatGPT Image May 8, 2025, 06_22_57 PM](https://github.com/user-attachments/assets/e692b029-f0e2-45cb-9f01-22f4c6e43be1)



**Building a GPT Based Large Language Model from Scratch with pytorch** 

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


**Data sampling with a sliding window**


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
in self-attention before adding trainable weights.Figure 3.7 shows an input sequence, denoted as x, consisting of T elements repre-
sented as x(1) to x(T). This sequence typically represents text, such as a sentence, that
has already been transformed into token embeddings.


For example, consider an input text like “Your journey starts with one step.” In this
case, each element of the sequence, such as x(1), corresponds to a d-dimensional
embedding vector representing a specific token, like “Your.” Figure 3.7 shows these
input vectors as three-dimensional embeddings.


In self-attention, our goal is to calculate context vectors z(i) for each element x(i)
in the input sequence. A context vector can be interpreted as an enriched embedding
vector.


To illustrate this concept, let’s focus on the embedding vector of the second input
element, x(2) (which corresponds to the token “journey”), and the corresponding con-
text vector, z(2), shown at the bottom of figure 3.7. This enhanced context vector, z(2),
is an embedding that contains information about x(2) and all other input elements,
x(1) to x(T).


Context vectors play a crucial role in self-attention. Their purpose is to create
enriched representations of each element in an input sequence (like a sentence)
by incorporating information from all other elements in the sequence (figure 3.7).
This is essential in LLMs, which need to understand the relationship and relevance
of words in a sentence to each other.


![image](https://github.com/user-attachments/assets/8cd5acf1-4a67-4e8b-86ca-c5d531a92b4b)

Figure 3.8 illustrates how we calculate the intermediate attention scores between the
query token and each input token. We determine these scores by computing the dot
product of the query, x(2), with every other input token:

![image](https://github.com/user-attachments/assets/988c5641-966e-4cba-a73c-9f810e5b99e5)

In the next step, as shown in figure 3.9, we normalize each of the attention scores we
computed previously. The main goal behind the normalization is to obtain attention
weights that sum up to 1. This normalization is a convention that is useful for interpre-
tation and maintaining training stability in an LLM.

![image](https://github.com/user-attachments/assets/aa6fb283-fd36-4806-95cb-878b93ea4846)

Now that we have computed the normalized attention weights, we are ready for the
final step, as shown in figure 3.10: calculating the context vector z(2) by multiplying the
embedded input tokens, x(i), with the corresponding attention weights and then sum-
ming the resulting vectors. Thus, context vector z(2) is the weighted sum of all input vec-
tors, obtained by multiplying each input vector by its corresponding attention weight

![image](https://github.com/user-attachments/assets/c593e753-9cdb-49d1-b092-55951d4016e4)

Next, we will generalize this procedure for computing context vectors to calculate all
context vectors simultaneously.


**Computing attention weights for all input tokens**
![image](https://github.com/user-attachments/assets/ef90c39a-73b0-47bf-adb1-8b1fc1317b8d)

So far, we have computed attention weights and the context vector for input 2, as
shown in the highlighted row in figure 3.11. Now let’s extend this computation to cal-
culate attention weights and context vectors for all inputs.

_We follow the same three steps as before (see figure 3.12)_

![image](https://github.com/user-attachments/assets/810dae14-898b-4a89-ad5c-0f5481256ebd)

**Implementing self-attention with trainable weights**
Our next step will be to implement the self-attention mechanism used in the origi-
nal transformer architecture, the GPT models, and most other popular LLMs. This
self-attention mechanism is also called scaled dot-product attention. Figure 3.13 shows
how this self-attention mechanism fits into the broader context of implementing
an LLM.

![image](https://github.com/user-attachments/assets/8ac48c8a-a2c2-46db-8c1f-b76588de338d)

![image](https://github.com/user-attachments/assets/c9e1c8a6-3ce9-47ea-a5cb-b24106a5db64)
![image](https://github.com/user-attachments/assets/985d2fe0-909b-4d74-8b11-c8ff8e855b4c)
![image](https://github.com/user-attachments/assets/c49c0102-a4f2-4771-8c05-ecb40d639b45)
![image](https://github.com/user-attachments/assets/9a9da394-63f1-4d39-a459-2bcaf99aba83)

**Hiding future words with causal attention**

For many LLM tasks, you will want the self-attention mechanism to consider only the
tokens that appear prior to the current position when predicting the next token in a
sequence. Causal attention, also known as masked attention, is a specialized form of self-
attention. It restricts a model to only consider previous and current inputs in a sequence
when processing any given token when computing attention scores. This is in contrast
to the standard self-attention mechanism, which allows access to the entire input
sequence at once.
Now, we will modify the standard self-attention mechanism to create a causal
attention mechanism, which is essential for developing an LLM in the subsequent
chapters. To achieve this in GPT-like LLMs, for each token processed, we mask out
the future tokens, which come after the current token in the input text, as illus-
trated in figure 3.19. We mask out the attention weights above the diagonal, and we normalize the nonmasked attention weights such that the attention weights sum to 1 in
each row.


![image](https://github.com/user-attachments/assets/932c99f2-31a1-4fe6-a722-f473023cf3b9)


**Applying a causal attention mask**

Our next step is to implement the causal attention mask in code. To implement the
steps to apply a causal attention mask to obtain the masked attention weights, as sum-
marized in figure 3.20

![image](https://github.com/user-attachments/assets/313d4615-9001-483d-98c7-23400de981ae)


![image](https://github.com/user-attachments/assets/9d2a5af0-dbe5-44e9-8595-2a428e1b45a9)

**Masking additional attention weights with dropout**

Dropout in deep learning is a technique where randomly selected hidden layer units
are ignored during training, effectively “dropping” them out. This method helps pre-
vent overfitting by ensuring that a model does not become overly reliant on any spe-
cific set of hidden layer units. It’s important to emphasize that dropout is only used
during training and is disabled afterward.
In the transformer architecture, including models like GPT, dropout in the atten-
tion mechanism is typically applied at two specific times: after calculating the atten-
tion weights or after applying the attention weights to the value vectors. Here we will
apply the dropout mask after computing the attention weights, as illustrated in fig-
ure 3.22, because it’s the more common variant in practice.
![image](https://github.com/user-attachments/assets/4152ddb2-ee2e-4cd1-893d-c31dc558d0f2)


Figure 3.23 summarizes what we have accomplished so far. We have focused on the
concept and implementation of causal attention in neural networks. Next, we will
expand on this concept and implement a multi-head attention module that imple-
ments several causal attention mechanisms in parallel.

![image](https://github.com/user-attachments/assets/f75f66af-5c1d-4eb2-b0e9-bab3deef20ff)

**Extending single-head attention to multi-head
attention**

Our final step will be to extend the previously implemented causal attention class over
multiple heads. This is also called multi-head attention.

The term “multi-head” refers to dividing the attention mechanism into multiple
“heads,” each operating independently. In this context, a single causal attention mod-
ule can be considered single-head attention, where there is only one set of attention
weights processing the input sequentially.

We will tackle this expansion from causal attention to multi-head attention. First,
we will intuitively build a multi-head attention module by stacking multiple Causal-
Attention modules. Then we will then implement the same multi-head attention
module in a more complicated but more computationally efficient way.

**Stacking multiple single-head attention layers**

In practical terms, implementing multi-head attention involves creating multiple
instances of the self-attention mechanism (see figure 3.18), each with its own weights,
and then combining their outputs. Using multiple instances of the self-attention
mechanism can be computationally intensive, but it’s crucial for the kind of complex
pattern recognition that models like transformer-based LLMs are known for.

Figure 3.24 illustrates the structure of a multi-head attention module, which con-
sists of multiple single-head attention modules, as previously depicted in figure 3.18,
stacked on top of each other.

![image](https://github.com/user-attachments/assets/67324dd0-bb1b-44e8-a864-09900d0c68ca)
![image](https://github.com/user-attachments/assets/9c5a5541-fd33-4f6b-9d61-9e059cc3629b)

**Implementing multi-head attention with weight splits**

So far, we have created a MultiHeadAttentionWrapper to implement multi-head
attention by stacking multiple single-head attention modules. This was done by instan-
tiating and combining several CausalAttention objects.

Instead of maintaining two separate classes, MultiHeadAttentionWrapper and
CausalAttention, we can combine these concepts into a single MultiHeadAttention
class. Also, in addition to merging the MultiHeadAttentionWrapper with the Causal-
Attention code, we will make some other modifications to implement multi-head
attention more efficiently.

In the MultiHeadAttentionWrapper, multiple heads are implemented by creating
a list of CausalAttention objects (self.heads), each representing a separate atten-
tion head. The CausalAttention class independently performs the attention mecha-
nism, and the results from each head are concatenated. In contrast, the following
MultiHeadAttention class integrates the multi-head functionality within a single class.

It splits the input into multiple heads by reshaping the projected query, key, and value
tensors and then combines the results from these heads after computing attention.

![image](https://github.com/user-attachments/assets/d7c7c4f1-9c8e-4d25-b2ee-43a42a463e0c)

**Coding an LLM architecture**


![image](https://github.com/user-attachments/assets/30f49dbd-06aa-43c4-8973-759eca19c712)


![image](https://github.com/user-attachments/assets/dcb429e8-60ae-4684-8e1b-27414fa05f95)

LLMs, such as GPT (which stands for generative pretrained transformer), are large deep
neural network architectures designed to generate new text one word (or token) at a
time. However, despite their size, the model architecture is less complicated than you
might think, since many of its components are repeated, as we will see later. Figure 4.2
provides a top-down view of a GPT-like LLM, with its main components highlighted.


![image](https://github.com/user-attachments/assets/d1504d8e-cb45-4f5d-a0b0-8997a4e70b56)

We have already covered several aspects of the LLM architecture, such as input
tokenization and embedding and the masked multi-head attention module. Now, we
will implement the core structure of the GPT model, including its transformer blocks,
which we will later train to generate human-like text.
Previously, we used smaller embedding dimensions for simplicity, ensuring that the
concepts and examples could comfortably fit on a single page. Now, we are scaling up
to the size of a small GPT-2 model, specifically the smallest version with 124 million
parameters, as described in “Language Models Are Unsupervised Multitask Learners,”
by Radford et al. (https://mng.bz/yoBq).

In the context of deep learning and LLMs like GPT, the term “parameters” refers
to the trainable weights of the model. These weights are essentially the internal vari-
ables of the model that are adjusted and optimized during the training process to
minimize a specific loss function. This optimization allows the model to learn from
the training data.

For example, in a neural network layer that is represented by a 2,048 × 2,048–dimensional
matrix (or tensor) of weights, each element of this matrix is a parameter. Since there
are 2,048 rows and 2,048 columns, the total number of parameters in this layer is 2,048
multiplied by 2,048, which equals 4,194,304 parameters.


![image](https://github.com/user-attachments/assets/b573ee24-92a4-4887-b565-8a2c0653e460)

![image](https://github.com/user-attachments/assets/6beccb61-f85a-4028-a4a2-38bc083f1719)



**Normalizing activations with layer normalization** 

Training deep neural networks with many layers can sometimes prove challenging
due to problems like vanishing or exploding gradients. These problems lead to unsta-
ble training dynamics and make it difficult for the network to effectively adjust its
weights, which means the learning process struggles to find a set of parameters
(weights) for the neural network that minimizes the loss function. In other words, the
network has difficulty learning the underlying patterns in the data to a degree that
would allow it to make accurate predictions or decisions.

Let’s now implement layer normalization to improve the stability and efficiency of neu-
ral network training. The main idea behind layer normalization is to adjust the activa-
tions (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also
known as unit variance. This adjustment speeds up the convergence to effective
weights and ensures consistent, reliable training.

Figure 4.5 provides a visual overview of how layer normalization
functions.

![image](https://github.com/user-attachments/assets/39b54e0b-9b40-468f-804b-eb3ac002c7d4)

![image](https://github.com/user-attachments/assets/ba967774-abab-4471-a88d-67f1a4398ecb)

**Implementing a feed forward network
with GELU activations**
Next, we will implement a small neural network submodule used as part of the trans-
former block in LLMs. We begin by implementing the GELU activation function,
which plays a crucial role in this neural network submodule.

Historically, the ReLU activation function has been commonly used in deep learning
due to its simplicity and effectiveness across various neural network architectures.
However, in LLMs, several other activation functions are employed beyond the tradi-
tional ReLU. Two notable examples are GELU (Gaussian error linear unit) and SwiGLU
(Swish-gated linear unit).

GELU and SwiGLU are more complex and smooth activation functions incorpo-
rating Gaussian and sigmoid-gated linear units, respectively. They offer improved per-
formance for deep learning models, unlike the simpler ReLU.

The GELU activation function can be implemented in several ways; the exact ver-
sion is defined as GELU(x) = x⋅Φ(x), where Φ(x) is the cumulative distribution func-
tion of the standard Gaussian distribution. In practice, however, it’s common to
implement a computationally cheaper approximation (the original GPT-2 model was
also trained with this approximation, which was found via curve fitting):

![image](https://github.com/user-attachments/assets/2c98ab04-2106-47b7-aeb7-0c946df794e9)

Next, to get an idea of what this GELU function looks like and how it compares to the
ReLU function, let’s plot these functions side by side:

![image](https://github.com/user-attachments/assets/7ad71ffc-9b52-4f34-a2d6-39049d0e7aa4)
As we can see in the resulting plot in figure 4.8, ReLU (right) is a piecewise linear
function that outputs the input directly if it is positive; otherwise, it outputs zero.
GELU (left) is a smooth, nonlinear function that approximates ReLU but with a non-
zero gradient for almost all negative values (except at approximately x = –0.75).

The smoothness of GELU can lead to better optimization properties during training,
as it allows for more nuanced adjustments to the model’s parameters. In contrast,
ReLU has a sharp corner at zero (figure 4.18, right), which can sometimes make opti-
mization harder, especially in networks that are very deep or have complex architec-
tures. 
Moreover, unlike ReLU, which outputs zero for any negative input, GELU
allows for a small, non-zero output for negative values. This characteristic means that
during the training process, neurons that receive negative input can still contribute to
the learning process, albeit to a lesser extent than positive inputs.

Next, let’s use the GELU function to implement the small neural network module,
FeedForward, that we will be using in the LLM’s transformer block later.

As we can see, the FeedForward module is a small neural network consisting of two
Linear layers and a GELU activation function. In the 124-million-parameter GPT
model, it receives the input batches with tokens that have an embedding size of 768
each via the GPT_CONFIG_124M dictionary where GPT_CONFIG_ 124M["emb_dim"] = 768.
Figure 4.9 shows how the embedding size is manipulated inside this small feed for-
ward neural network when we pass it some inputs.

![image](https://github.com/user-attachments/assets/933394f1-e1e6-43bf-b5c8-77b71b1ba5d6)

The FeedForward module plays a crucial role in enhancing the model’s ability to learn
from and generalize the data. Although the input and output dimensions of this
module are the same, it internally expands the embedding dimension into a higher-
dimensional space through the first linear layer, as illustrated in figure 4.10. This expan-
sion is followed by a nonlinear GELU activation and then a contraction back to the orig-
inal dimension with the second linear transformation. Such a design allows for the
exploration of a richer representation space.

![image](https://github.com/user-attachments/assets/8cc866ef-9f8a-4b44-a81b-18e8425eaf18)



![image](https://github.com/user-attachments/assets/9058481d-b0a4-427e-a956-b9b0669b2f54)
As figure 4.11 shows, we have now implemented most of the LLM’s building blocks.
Next, we will go over the concept of shortcut connections that we insert between dif-
ferent layers of a neural network, which are important for improving the training
performance in deep neural network architectures.

**Adding shortcut connections**

Let’s discuss the concept behind shortcut connections, also known as skip or residual
connections. Originally, shortcut connections were proposed for deep networks in
computer vision (specifically, in residual networks) to mitigate the challenge of van-
ishing gradients. The vanishing gradient problem refers to the issue where gradients
(which guide weight updates during training) become progressively smaller as they
propagate backward through the layers, making it difficult to effectively train earlier
layers.

Figure 4.12 shows that a shortcut connection creates an alternative, shorter path
for the gradient to flow through the network by skipping one or more layers, which is
achieved by adding the output of one layer to the output of a later layer. This is why
these connections are also known as skip connections. They play a crucial role in pre-
serving the flow of gradients during the backward pass in training.
In the following list, we implement the neural network in figure 4.12 to see how
we can add shortcut connections in the forward method.

![image](https://github.com/user-attachments/assets/db0830db-8175-4a69-8013-75264a2b90bf)

**Next, we’ll connect all of the previously covered concepts (layer normalization,
GELU activations, feed forward module, and shortcut connections) in a transformer
block, which is the final building block we need to code the GPT architecture.**

Connecting attention and linear layers
in a transformer block

Now, let’s implement the transformer block, a fundamental building block of GPT and
other LLM architectures. This block, which is repeated a dozen times in the 124-million-
parameter GPT-2 architecture, combines several concepts we have previously covered:
multi-head attention, layer normalization, dropout, feed forward layers, and GELU
activations. Later, we will connect this transformer block to the remaining parts of the
GPT architecture. 

Figure 4.13 shows a transformer block that combines several components, includ-
ing the masked multi-head attention module (see chapter 3) and the FeedForward
module we previously implemented (see section 4.3). When a transformer block pro-
cesses an input sequence, each element in the sequence (for example, a word or sub-
word token) is represented by a fixed-size vector (in this case, 768 dimensions). The
operations within the transformer block, including multi-head attention and feed for-
ward layers, are designed to transform these vectors in a way that preserves their
dimensionality.

The idea is that the self-attention mechanism in the multi-head attention block iden-
tifies and analyzes relationships between elements in the input sequence. In contrast,
the feed forward network modifies the data individually at each position. This combina-
tion not only enables a more nuanced understanding and processing of the input but
also enhances the model’s overall capacity for handling complex data patterns.


![image](https://github.com/user-attachments/assets/f39e9516-7365-4190-828e-3106dc85cd6f)

![image](https://github.com/user-attachments/assets/cf0e5ef4-e444-443b-a8ac-5f82a5c9f770)

**Coding the GPT model**
![image](https://github.com/user-attachments/assets/ce96c411-3a02-4e58-96f3-f21a275e39fb)

![image](https://github.com/user-attachments/assets/b5222878-1968-4ddf-8072-8a54711b71bb)
![image](https://github.com/user-attachments/assets/092f0489-7849-42f9-bbdc-9de94072a0ed)

![image](https://github.com/user-attachments/assets/0c590c3c-bef7-49c3-8a97-990b5cd7d9d5)

![image](https://github.com/user-attachments/assets/35710c8a-3110-4993-b0c4-5bcb13d965a3)

![image](https://github.com/user-attachments/assets/5ac9e846-4ab3-4006-99ef-242bdf41f12c)
![image](https://github.com/user-attachments/assets/07ce7f1e-4dda-4de4-adea-acd71d2e10e7)
![image](https://github.com/user-attachments/assets/a0072c99-18e0-42eb-9121-7401bf9b89c8)
![image](https://github.com/user-attachments/assets/045bca94-dac7-4bf0-a6bb-3a4f9161f1ec)
![image](https://github.com/user-attachments/assets/3c07e1d7-0dd1-4aaf-8f2c-3bbfd13cd35b)
![image](https://github.com/user-attachments/assets/3ed692c3-5c5d-4a19-b3a8-1ed5df758238)
![image](https://github.com/user-attachments/assets/104f1730-09f4-4a7f-aa74-f92a8824bc91)
![image](https://github.com/user-attachments/assets/b92ff87b-bd6d-4c95-882c-468635d95bde)
![image](https://github.com/user-attachments/assets/43ec1a00-be99-45c7-a090-818ad7cee172)
![image](https://github.com/user-attachments/assets/8ebd0a21-e98d-4759-882a-f7987845c3a1)
![image](https://github.com/user-attachments/assets/0e0b3b5e-9897-4785-b18c-756ed05d9e4f)
![image](https://github.com/user-attachments/assets/6b7af7f6-a5c2-4cf0-9a42-b2ec49280ba1)
![image](https://github.com/user-attachments/assets/5139b867-cf11-4ad7-a4c4-87fef403fe11)
![image](https://github.com/user-attachments/assets/ee9046fb-db34-433b-a841-a48b765e93a6)
![image](https://github.com/user-attachments/assets/f38c9fb4-accc-4fdc-acf4-5c842bd727a2)
![image](https://github.com/user-attachments/assets/fecbcf6f-f0a6-4cc2-b3e1-c2c18305752a)
![image](https://github.com/user-attachments/assets/b566df22-a996-4220-8f69-d49298b4879e)
![image](https://github.com/user-attachments/assets/2db1d776-085b-4f04-a248-372e51239799)
![image](https://github.com/user-attachments/assets/080d8a3c-ade2-4439-be35-1db6a32d22c1)
![image](https://github.com/user-attachments/assets/94eb1d5d-7a96-4010-beb6-27a601b27478)
![image](https://github.com/user-attachments/assets/e86d2e9f-0294-452c-a2f8-d6c99f4863f0)
![image](https://github.com/user-attachments/assets/36fdb7ca-9c93-46aa-8002-c5b251a2bed8)
![image](https://github.com/user-attachments/assets/e96faa32-0c4f-4d7c-998a-f9d4117314ed)
![image](https://github.com/user-attachments/assets/f39131cc-bd44-4939-9bce-d302a8931103)
![image](https://github.com/user-attachments/assets/220475c6-7b41-43ff-a742-857e7a1bb8cb)
![image](https://github.com/user-attachments/assets/3146928f-39b6-402a-af7e-7e93db6daa33)
![image](https://github.com/user-attachments/assets/11e43fc4-56f5-49b6-98b2-d93fbbd7c777)
![image](https://github.com/user-attachments/assets/faeda0e5-7000-4add-b919-58103bb5d22b)
![image](https://github.com/user-attachments/assets/1e390c45-0c3c-4123-b14f-26f5ab63c4b9)
![image](https://github.com/user-attachments/assets/d7524506-1c70-4bb3-8620-e66cd5c2906a)
![image](https://github.com/user-attachments/assets/97424cec-5ead-4673-bd2a-1a36d62ba586)
![image](https://github.com/user-attachments/assets/ef8cc4a6-907d-495e-8839-0c2272c36ecf)
![image](https://github.com/user-attachments/assets/13d78efe-9ae8-4bb8-be87-7dfe02c2437f)
![image](https://github.com/user-attachments/assets/f9801f01-2bb1-4dd8-8f28-6e1e7f6caa8d)
![image](https://github.com/user-attachments/assets/1c8a2d05-17ff-4385-8091-e0b3a1c0d584)
![image](https://github.com/user-attachments/assets/605bb11f-9660-46d0-b565-5d2a0e217388)
![image](https://github.com/user-attachments/assets/00e61468-9bc2-4bda-833e-c65c6578ed8b)
![image](https://github.com/user-attachments/assets/09c5f1e3-ae00-44e3-947e-83c0ae19a08b)
![image](https://github.com/user-attachments/assets/060da0f2-a832-4ac9-a125-fb13038dfec4)
![image](https://github.com/user-attachments/assets/60052bc2-0b42-4fb9-a3e6-2b788f2e7911)
![image](https://github.com/user-attachments/assets/b048a2fd-e9c2-40a3-b349-76c6af96a3a2)
![image](https://github.com/user-attachments/assets/c5e515fb-2d82-4abf-bd4c-e2ce70d78bfc)
![image](https://github.com/user-attachments/assets/3cd2a18c-99c6-4f17-a80a-0bea3b7b434d)
![image](https://github.com/user-attachments/assets/e5e6b37b-b1fa-461e-a720-5470d9d15213)
![image](https://github.com/user-attachments/assets/084bd634-43bc-4d6c-a549-3349405b2f3e)
![image](https://github.com/user-attachments/assets/57f98a46-212d-43af-af3c-7301f522ca98)
![image](https://github.com/user-attachments/assets/f5db5756-6c9c-4d22-a364-862b177a91b3)
![image](https://github.com/user-attachments/assets/6e73e9f3-7c03-4490-b0b8-60afbb2f8839)
![image](https://github.com/user-attachments/assets/14c58a09-99e1-4a08-a128-ffe8378ed7ac)
![image](https://github.com/user-attachments/assets/2aea8398-c8f6-4ce5-8f6a-55932434b2af)
![image](https://github.com/user-attachments/assets/d27c656b-82e6-44bf-bc84-2cdc791e8efa)
![image](https://github.com/user-attachments/assets/af2c4794-afe7-488e-a87a-0b760386875d)

THANKS FOR READING!









