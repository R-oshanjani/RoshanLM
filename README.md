RoshanLM
A small decoder-only language model built from scratch using Python
and PyTorch.
🚀 Project Overview
RoshanLM is an educational AI/ML project that implements the core
components of a compact GPT-style language model without using a
pretrained LLM for the core model.
Pipeline
Raw Text → Custom BPE Tokenizer → Token IDs
→ Token + Positional Embeddings
→ Multi-Head Self-Attention
→ Transformer Blocks
→ Language Model Head
→ Next-Token Prediction
→ Generated Text
The training corpus focuses on Python, Machine Learning, NLP,
Transformers, LLMs, RAG, Data Science, PyTorch, SQL, Spark, MLOps, and
Generative AI.
🧠 Architecture
- Custom BPE tokenizer
- Token embeddings
- Learned positional embeddings
- Multi-head causal self-attention
- Causal masking
- Pre-LayerNorm
- Residual connections
- Feed-forward networks with GELU
- Final LayerNorm
- Linear language-model head
Current Configuration
  Component                   Value
  Parameters                142,592
  Vocabulary size               200
  Context length          64 tokens
  Embedding dimension            64
  Attention heads                 2
  Transformer layers              2
  Dropout                       0.3
  Batch size                     32
  Learning rate                1e-4
  Optimizer                   AdamW
  Training device               CPU
🔤 Custom BPE Tokenizer
The project includes a custom Byte Pair Encoding-style tokenizer.
It learns frequent symbol pairs from the training corpus, merges them
into subword tokens, and converts text to token IDs and back to text.
The tokenizer is trained on the training corpus and then used for
validation and generation.
🏋️ Training
RoshanLM includes:
- Training/validation split
- Mini-batch training
- Cross-entropy language-model loss
- AdamW optimization
- Validation monitoring
- Early stopping
- Best-model checkpointing
Training objective
For:
t1 t2 t3 t4 ... tn
the model learns to predict:
t2 t3 t4 t5 ... tn+1
This is next-token prediction.
📊 Current Results
Best recorded result:
Validation Loss: 2.4788
Perplexity: ~11.93
Perplexity is calculated as:
PPL = exp(validation_loss)
Lower validation loss and perplexity indicate better next-token
prediction on the validation corpus.
✍️ Text Generation
Run:
python generate.py
Example:
Prompt:
Python is

Generated text:
Python is a subword of set of sequence self attention helps the model...
The model has learned technical vocabulary and associations from its
training corpus.
Generation quality is still limited: the model can produce malformed
words, weak grammar, incomplete sentences, and topic transitions. This
is expected for a very small model trained on a relatively small corpus.
📁 Project Structure
RoshanLM/
├── data/
│   ├── domain_corpus.txt
│   ├── train.txt
│   ├── train_corpus.txt
│   └── validation_corpus.txt
├── checkpoints/
├── config.py
├── tokenizer.py
├── dataset.py
├── prepare_data.py
├── model.py
├── train.py
├── generate.py
├── requirements.txt
├── README.md
└── .gitignore
⚙️ Installation
Clone
git clone https://github.com/R-oshanjani/RoshanLM.git
cd RoshanLM
Virtual environment
Windows:
python -m venv venv
venv\Scripts\activate
Dependencies
pip install -r requirements.txt
🗂️ Prepare the Dataset
python prepare_data.py
This creates:
data/train_corpus.txt
data/validation_corpus.txt
The tokenizer is trained on the training corpus and reused for
validation.
🔎 Test the Model
python model.py
Expected output is similar to:
Input shape:
torch.Size([2, 8])

Output shape:
torch.Size([2, 8, 200])

Total parameters:
142592
🏃 Train
python train.py
The best checkpoint is saved to:
checkpoints/roshanlm.pt
💬 Generate
python generate.py
Current generation settings:
TEMPERATURE = 0.7
TOP_K = 10
🧪 Experiments
A 300-token vocabulary was also tested.
300-token experiment:
Best Validation Loss: 3.0613
Perplexity: ~21.36
Current 200-token configuration:
Best Validation Loss: 2.4788
Perplexity: ~11.93
The current project therefore uses the 200-token vocabulary.
⚠️ Current Limitations
RoshanLM is an educational/experimental language model, not a production
LLM.
Current limitations:
- Small model size
- Small training corpus
- Limited vocabulary
- Short context length
- CPU-based training
- Limited grammatical coherence
- Occasional malformed generated words
- Limited question-answering ability
- Limited long-context reasoning
- No external knowledge retrieval during generation
🔮 Future Improvements
- Expand and improve the training corpus
- Add high-quality question-answer examples
- Improve tokenizer vocabulary and tokenization
- Increase model capacity
- Experiment with larger context windows
- Add automated generation benchmarks
- Improve sampling strategies
- Add RAG as a separate layer
- Experiment with GPU training
- Improve model/tokenizer export
- Improve reproducibility and documentation
🛠️ Technologies
- Python
- PyTorch
- NumPy
- tqdm
- Custom BPE Tokenization
- Transformer Architecture
- Self-Attention
- NLP
- Generative AI
🎯 Learning Objectives
This project provides practical experience with:
- Tokenization
- Embeddings
- Self-attention
- Causal masking
- Transformer blocks
- Language-model training
- Validation loss and perplexity
- Next-token generation
- Limitations of small language models
👨‍💻 Author
Syed Roshan Jani
AI/ML Engineer | Python Developer | Generative AI Enthusiast
LinkedIn: https://www.linkedin.com/in/syed-roshan-jani-b31a53256/
GitHub: https://github.com/R-oshanjani
📌 Repository
https://github.com/R-oshanjani/RoshanLM
