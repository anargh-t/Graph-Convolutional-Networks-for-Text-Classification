# Mathematical Explanation: Topic Creation and Connections in TopicGCN

This document provides an in-depth mathematical explanation of how topics are created and how connections are established in the TopicGCN model.

---

## Project Overview: TopicGCN (Non-Mathematical Introduction)

### What is TopicGCN?

TopicGCN (Topic Graph Convolutional Network) is a text classification system that uses topics as an intermediary layer between documents. Instead of directly connecting documents to words (like traditional TextGCN), TopicGCN creates a **document-topic-topic graph** where:

- **Documents** connect to **Topics** (based on what topics the document discusses)
- **Topics** connect to **Topics** (based on semantic similarity)

This approach makes the graph more compact, interpretable, and semantically meaningful.

### Key Innovation

**Traditional Approach (TextGCN):**
```
Documents ←→ Words ←→ Words
```
- Documents directly connect to words
- Words connect to other words
- Graph is large (documents + vocabulary size)
- Hard to interpret

**TopicGCN Approach:**
```
Documents ←→ Topics ←→ Topics
```
- Documents connect to topics
- Topics connect to similar topics
- Graph is compact (documents + ~50 topics)
- Highly interpretable (topics are human-readable)

### How It Works: Conceptual Overview

#### Step 1: Discover Topics (LDA)
- **Input:** Collection of documents
- **Process:** Analyze which words appear together frequently
- **Output:** K topics (e.g., 50 topics)
  - Each topic is a collection of related words
  - Example Topic: "Sports" = {football, team, game, win, championship, ...}
  - Each document gets a "membership score" for each topic

**Simple Example:**
- Document: "The football team won the championship game"
- Topic Scores:
  - Sports topic: 75% (high - document is mostly about sports)
  - Politics topic: 5% (low - barely related)
  - Other topics: 20% (distributed among other topics)

#### Step 2: Create Topic Embeddings (Word2Vec)
- **Input:** Documents and discovered topics
- **Process:** 
  - Train Word2Vec to learn word meanings from context
  - For each topic, take its most important words
  - Combine their word embeddings into a single topic embedding
- **Output:** Each topic becomes a vector (list of numbers) representing its meaning

**Simple Example:**
- Topic "Sports" has top words: football, team, game, win
- Word embeddings:
  - football = [0.8, 0.6, 0.4, ...]
  - team = [0.7, 0.5, 0.5, ...]
  - game = [0.6, 0.7, 0.3, ...]
- Topic embedding = average of these word vectors
- Result: Sports topic = [0.7, 0.6, 0.4, ...] (captures "sports" meaning)

#### Step 3: Build Document-Topic Connections
- **Input:** Document-topic membership scores from LDA
- **Process:**
  - For each document and topic pair
  - If document's membership score ≥ threshold (e.g., 2%)
  - Create an edge with weight = membership score
- **Output:** Graph edges connecting documents to their relevant topics

**Simple Example:**
- Document 10 has 75% membership in Sports topic
- Document 10 has 5% membership in Politics topic
- Threshold = 2%
- Result:
  - Edge: Document 10 → Sports topic (weight = 0.75) ✅
  - Edge: Document 10 → Politics topic (weight = 0.05) ✅
  - No edge for topics with < 2% membership ❌

#### Step 4: Build Topic-Topic Connections
- **Input:** Topic embeddings from Step 2
- **Process:**
  - For each pair of topics, compute similarity
  - Similarity measures how "close" their meanings are
  - If similarity ≥ threshold (e.g., 30%)
  - Create an edge with weight = similarity
- **Output:** Graph edges connecting semantically similar topics

**Simple Example:**
- Sports topic embedding: [0.8, 0.6, 0.4, ...]
- Athletics topic embedding: [0.75, 0.65, 0.45, ...]
- Similarity = 92% (very similar - both about physical activities)
- Politics topic embedding: [0.2, 0.1, 0.3, ...]
- Similarity with Sports = 25% (different - below 30% threshold)
- Result:
  - Edge: Sports ↔ Athletics (weight = 0.92) ✅
  - No edge: Sports ↔ Politics ❌

#### Step 5: Train Graph Convolutional Network
- **Input:** The document-topic-topic graph
- **Process:**
  - Each document node gets features = its topic membership scores
  - Each topic node gets features = its embedding vector
  - GCN learns to propagate information through the graph
  - Documents receive information from their topics
  - Topics receive information from connected documents and similar topics
- **Output:** Trained model that can classify documents

**Simple Example:**
- Document 10 connects to Sports topic (75% weight)
- Sports topic connects to Athletics topic (92% similarity)
- When GCN processes Document 10:
  - It sees Document 10 is strongly about Sports
  - It sees Sports is similar to Athletics
  - It learns Document 10 is likely about sports/athletics category
  - Model predicts: "Document 10 belongs to Sports class"

### Key Components

#### 1. Topic Model (`topic_model.py`)
- **Purpose:** Discover topics and create embeddings
- **Technologies:** LDA (topic discovery) + Word2Vec (embeddings)
- **Output:** Topic distributions and topic embeddings

#### 2. Graph Builder (`build_graph.py`)
- **Purpose:** Construct the document-topic-topic graph
- **Process:**
  - Creates document-topic edges (from LDA scores)
  - Creates topic-topic edges (from similarity)
  - Filters edges using thresholds
- **Output:** Graph file ready for training

#### 3. GCN Model (`layer.py`)
- **Purpose:** Neural network that learns from the graph
- **Architecture:** Two-layer Graph Convolutional Network
- **Process:**
  - Layer 1: Aggregates information from neighbors
  - Layer 2: Produces classification predictions

#### 4. Trainer (`trainer.py`)
- **Purpose:** Train the GCN model
- **Process:**
  - Loads graph and features
  - Trains model with backpropagation
  - Evaluates on test set
- **Output:** Trained model and performance metrics

### Workflow Summary

```
1. Preprocess Documents
   └─> Clean text, remove stopwords, tokenize

2. Discover Topics (LDA)
   └─> Find K topics (e.g., 50 topics)
   └─> Get document-topic membership scores

3. Create Topic Embeddings (Word2Vec)
   └─> Train word embeddings
   └─> Combine into topic embeddings

4. Build Graph
   ├─> Document-Topic edges (from membership scores)
   └─> Topic-Topic edges (from similarity)

5. Train GCN
   ├─> Load graph and features
   ├─> Train neural network
   └─> Evaluate performance

6. Classify Documents
   └─> Use trained model to predict document categories
```

### Advantages of TopicGCN

1. **Compact Graph:**
   - Traditional: Documents + 10,000+ words = huge graph
   - TopicGCN: Documents + 50 topics = small graph
   - Faster training, less memory

2. **Interpretable:**
   - Topics are human-readable (e.g., "Sports: football, team, game...")
   - Can inspect which topics a document belongs to
   - Can visualize topic relationships

3. **Semantic Structure:**
   - Topic-topic edges capture meaning relationships
   - Documents connect through shared topics
   - Better captures document similarity

4. **Better Generalization:**
   - Topics capture high-level concepts
   - Less sensitive to word-level variations
   - More robust to noise

### Example: Real-World Scenario

**Dataset:** News articles about sports, politics, and technology

**Step 1 - Topic Discovery:**
- Topic 1: Sports (words: football, team, game, win, championship)
- Topic 2: Politics (words: election, vote, policy, government)
- Topic 3: Technology (words: computer, software, algorithm, data)

**Step 2 - Document Assignment:**
- Article A: "The football team won the championship"
  - Sports: 80%, Politics: 5%, Technology: 15%
- Article B: "Election results show policy changes"
  - Sports: 5%, Politics: 85%, Technology: 10%

**Step 3 - Graph Construction:**
- Article A → Sports topic (weight: 0.80)
- Article B → Politics topic (weight: 0.85)
- Sports ↔ Athletics topic (similarity: 0.90)
- Politics ↔ Government topic (similarity: 0.88)

**Step 4 - Classification:**
- GCN learns: Articles connected to Sports topic → "Sports" category
- GCN learns: Articles connected to Politics topic → "Politics" category
- Model can classify new articles based on their topic connections

---

## Mathematical Details

The following sections provide detailed mathematical explanations of each component.

---

## Table of Contents

1. [Topic Creation: Latent Dirichlet Allocation (LDA)](#1-topic-creation-latent-dirichlet-allocation-lda)
2. [Document-Topic Connections (θ_dk)](#2-document-topic-connections-θ_dk)
3. [Topic Embeddings (Word2Vec)](#3-topic-embeddings-word2vec)
4. [Topic-Topic Connections (Cosine Similarity)](#4-topic-topic-connections-cosine-similarity)
5. [Complete Mathematical Pipeline](#5-complete-mathematical-pipeline)

---

## 1. Topic Creation: Latent Dirichlet Allocation (LDA)

### 1.1 The LDA Probabilistic Model

LDA assumes that documents are generated from a mixture of topics. The mathematical model is:

**Generative Process:**
```
For each document d:
  1. Choose a topic distribution: θ_d ~ Dirichlet(α)
  2. For each word w in document d:
     a. Choose a topic: z ~ Multinomial(θ_d)
     b. Choose a word: w ~ Multinomial(φ_z)
```

### 1.2 Mathematical Notation

| Symbol | Meaning | Simple Explanation |
|--------|---------|-------------------|
| **D** | Number of documents | Total count of documents in your corpus |
| **K** | Number of topics | How many topics we want to discover (e.g., 50) |
| **V** | Vocabulary size | Total number of unique words in all documents |
| **d** | Document index | Which document we're talking about (0 to D-1) |
| **k** | Topic index | Which topic we're talking about (0 to K-1) |
| **w** | Word index | Which word we're talking about (0 to V-1) |
| **θ_d** | Document-topic distribution | Probability vector showing how much document d belongs to each topic |
| **θ_dk** | Document-topic probability | Probability that document d belongs to topic k |
| **φ_k** | Topic-word distribution | Probability vector showing which words appear in topic k |
| **φ_kw** | Topic-word probability | Probability that word w appears in topic k |
| **α** | Dirichlet prior for documents | Controls how "mixed" documents are (higher = more topics per doc) |
| **β** | Dirichlet prior for topics | Controls how "focused" topics are (higher = more words per topic) |
| **z_dn** | Topic assignment | Which topic generated word n in document d |

### 1.3 The Document-Term Matrix

First, we convert documents into a numerical representation:

**X** = Document-Term Matrix (also called Bag-of-Words)

```
        word_1  word_2  word_3  ...  word_V
doc_1  [  x_11   x_12   x_13   ...   x_1V  ]
doc_2  [  x_21   x_22   x_23   ...   x_2V  ]
doc_3  [  x_31   x_32   x_33   ...   x_3V  ]
  ...  [  ...    ...    ...    ...   ...   ]
doc_D  [  x_D1   x_D2   x_D3   ...   x_DV  ]
```

**X_dw** = Count of word w in document d

**Simple Explanation:**
- Each row is a document
- Each column is a word
- Each cell counts how many times that word appears in that document
- Example: If document 1 contains "cat" 3 times, then X_1,cat = 3

### 1.4 LDA Training (Expectation-Maximization)

LDA learns two key distributions:

#### **A. Topic-Word Distribution: φ_k**

For each topic k, we learn which words are most likely:

**φ_k** = [φ_k1, φ_k2, ..., φ_kV]

Where:
```
φ_kw = P(word = w | topic = k)
```

**Constraints:**
```
∑(w=1 to V) φ_kw = 1    (Each topic's word probabilities sum to 1)
φ_kw ≥ 0                (All probabilities are non-negative)
```

**Simple Explanation:**
- φ_k is a probability distribution over all words
- High φ_kw means word w is very characteristic of topic k
- Example: If topic 5 is about "sports", then φ_5,"football" might be 0.15 (high), while φ_5,"quantum" might be 0.0001 (low)

**In Code:**
```python
# After LDA training:
self.topic_word_distribution = self.lda_model.components_  
# Shape: (K, V) - K topics × V vocabulary words

# Normalize to probabilities:
self.topic_word_distribution = self.topic_word_distribution / \
    self.topic_word_distribution.sum(axis=1, keepdims=True)
```

#### **B. Document-Topic Distribution: θ_d**

For each document d, we learn which topics it belongs to:

**θ_d** = [θ_d1, θ_d2, ..., θ_dK]

Where:
```
θ_dk = P(topic = k | document = d)
```

**Constraints:**
```
∑(k=1 to K) θ_dk = 1    (Each document's topic probabilities sum to 1)
θ_dk ≥ 0                (All probabilities are non-negative)
```

**Simple Explanation:**
- θ_d is a probability distribution over all topics
- High θ_dk means document d is strongly about topic k
- Example: If document 10 is 60% about "sports" and 30% about "politics", then:
  - θ_10,sports = 0.60
  - θ_10,politics = 0.30
  - θ_10,other_topics = 0.10 (distributed among remaining topics)

**In Code:**
```python
# Get document-topic distribution:
doc_topic_dist = self.lda_model.transform(doc_term_matrix)
# Shape: (D, K) - D documents × K topics

# Each row is θ_d, each column is a topic
# Example: doc_topic_dist[10, 5] = θ_10,5 = probability doc 10 belongs to topic 5
```

### 1.5 The Complete LDA Matrix Formulation

**Topic-Word Matrix: Φ**
```
        word_1    word_2    word_3    ...    word_V
topic_1 [ φ_11     φ_12     φ_13     ...     φ_1V  ]
topic_2 [ φ_21     φ_22     φ_23     ...     φ_2V  ]
topic_3 [ φ_31     φ_32     φ_33     ...     φ_3V  ]
  ...   [ ...      ...      ...      ...     ...   ]
topic_K [ φ_K1     φ_K2     φ_K3     ...     φ_KV  ]
```

**Document-Topic Matrix: Θ**
```
        topic_1   topic_2   topic_3   ...   topic_K
doc_1  [ θ_11     θ_12     θ_13     ...     θ_1K  ]
doc_2  [ θ_21     θ_22     θ_23     ...     θ_2K  ]
doc_3  [ θ_31     θ_32     θ_33     ...     θ_3K  ]
  ...  [ ...      ...      ...      ...     ...   ]
doc_D  [ θ_D1     θ_D2     θ_D3     ...     θ_DK  ]
```

**Relationship:**
```
X ≈ Θ × Φ
```

Where:
- **X** (D × V): Document-term matrix (observed data)
- **Θ** (D × K): Document-topic matrix (learned)
- **Φ** (K × V): Topic-word matrix (learned)

**Simple Explanation:**
- We observe which words appear in which documents (X)
- LDA finds hidden topics that explain this pattern
- Each document is a mixture of topics (Θ)
- Each topic is a mixture of words (Φ)
- Multiplying Θ × Φ reconstructs approximately what words appear in documents

---

## 2. Document-Topic Connections (θ_dk)

### 2.1 Edge Weight Definition

For each document d and topic k, we create an edge with weight:

**Edge Weight:**
```
w(d, k) = θ_dk
```

**Simple Explanation:**
- The edge weight is exactly the probability that document d belongs to topic k
- Higher probability = stronger connection
- Lower probability = weaker connection (or no edge if below threshold)

### 2.2 Threshold Filtering

Not all document-topic pairs get edges. We only keep edges where:

```
θ_dk ≥ τ_doc_topic
```

Where:
- **τ_doc_topic** = Document-topic threshold (default: 0.02)

**Simple Explanation:**
- If a document has less than 2% probability of belonging to a topic, we don't create an edge
- This keeps the graph sparse and focused on meaningful connections
- Example: If θ_10,5 = 0.015 (1.5%), and threshold = 0.02, then NO edge is created
- If θ_10,5 = 0.05 (5%), then an edge is created with weight 0.05

### 2.3 Mathematical Formulation

**Edge Set:**
```
E_doc_topic = {(d, k) | θ_dk ≥ τ_doc_topic, d ∈ [0, D-1], k ∈ [0, K-1]}
```

**Edge Weight Function:**
```
w: E_doc_topic → [0, 1]
w(d, k) = θ_dk
```

**In Code:**
```python
for doc_idx in range(self.num_docs):
    for topic_idx in range(self.num_topics):
        weight = self.doc_topic_dist[doc_idx, topic_idx]  # θ_dk
        if weight < self.doc_topic_threshold:  # τ_doc_topic
            continue  # Skip this edge
        
        topic_node_idx = self.num_docs + topic_idx
        self.graph.add_edge(doc_idx, topic_node_idx, weight=float(weight))
```

### 2.4 Example Calculation

**Given:**
- Document 10: "The football team won the championship game"
- Topic 5: Sports topic (high probability for words: football, team, game, win, etc.)
- Topic 7: Politics topic (high probability for words: election, vote, policy, etc.)

**LDA Output:**
```
θ_10,5 = 0.75  (Document 10 is 75% about sports)
θ_10,7 = 0.05  (Document 10 is 5% about politics)
θ_10,others = 0.20  (Document 10 is 20% about other topics)
```

**Edge Creation (with threshold = 0.02):**
- Edge (doc_10, topic_5): weight = 0.75 ✅ (above threshold)
- Edge (doc_10, topic_7): weight = 0.05 ✅ (above threshold)
- Edge (doc_10, topic_others): weight = 0.20 ✅ (above threshold, if any)

**Graph Structure:**
```
doc_10 ──0.75──> topic_5 (sports)
doc_10 ──0.05──> topic_7 (politics)
doc_10 ──0.20──> topic_X (other topics)
```

---

## 3. Topic Embeddings (Word2Vec)

### 3.1 Word2Vec Overview

Word2Vec learns dense vector representations of words by predicting words from their context.

**Input:** Sentences/documents
**Output:** For each word w, a vector **v_w** ∈ ℝ^E

Where:
- **E** = Embedding dimension (default: 100)
- **v_w** = Word embedding vector for word w

**Simple Explanation:**
- Each word becomes a list of 100 numbers (a vector)
- Words with similar meanings have similar vectors
- Example: "cat" and "dog" vectors are closer than "cat" and "quantum"

### 3.2 Word2Vec Training

**Objective Function (CBOW - Continuous Bag of Words):**

For each word w_t at position t in a document:
```
P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})
```

Where:
- **c** = Context window size (default: 5)
- We predict word w_t from its surrounding words

**Simple Explanation:**
- Given words around a position, predict the word at that position
- Example: Given ["the", "cat", "sat", "on"], predict "the" (if window=2)

### 3.3 Topic Embedding Calculation

For each topic k, we compute its embedding as a weighted average of its top words:

**Step 1: Get Top Words for Topic k**

For topic k, get top N words with highest φ_kw:
```
TopWords_k = {(w, φ_kw) | w in top N words by φ_kw}
```

**Step 2: Weighted Average of Word Embeddings**

**Topic Embedding:**
```
e_k = (1 / |TopWords_k|) × Σ(w, prob) ∈ TopWords_k  [v_w × prob]
```

**Expanded Form:**
```
e_k = (1/N) × [v_{w1} × φ_kw1 + v_{w2} × φ_kw2 + ... + v_{wN} × φ_kwN]
```

Where:
- **e_k** ∈ ℝ^E = Topic embedding vector for topic k
- **v_w** ∈ ℝ^E = Word embedding for word w
- **φ_kw** = Probability of word w in topic k (from LDA)
- **N** = Number of top words used (default: 20)

**Simple Explanation:**
1. Find the top 20 words that best represent topic k
2. For each word, multiply its embedding vector by how important it is to the topic (φ_kw)
3. Add all these weighted vectors together
4. Divide by the number of words to get the average
5. Result: A single vector that represents the topic

**In Code:**
```python
for topic_id in range(self.num_topics):
    top_words = topic_words[topic_id]  # List of (word, prob) tuples
    
    word_vectors = []
    for word, prob in top_words:
        if word in self.word2vec_model.wv:
            # Weight by probability: v_w × φ_kw
            word_vectors.append(self.word2vec_model.wv[word] * prob)
    
    # Average: e_k = mean of weighted vectors
    topic_emb = np.mean(word_vectors, axis=0)
```

### 3.4 Example Calculation

**Given:**
- Topic 5: Sports topic
- Top words for topic 5:
  - "football": φ_5,football = 0.15
  - "team": φ_5,team = 0.12
  - "game": φ_5,game = 0.10
  - ... (17 more words)

**Word Embeddings (simplified, E=3 for illustration):**
```
v_football = [0.8, 0.6, 0.4]
v_team     = [0.7, 0.5, 0.5]
v_game     = [0.6, 0.7, 0.3]
```

**Weighted Vectors:**
```
v_football × 0.15 = [0.8×0.15, 0.6×0.15, 0.4×0.15] = [0.12, 0.09, 0.06]
v_team     × 0.12 = [0.7×0.12, 0.5×0.12, 0.5×0.12] = [0.084, 0.06, 0.06]
v_game     × 0.10 = [0.6×0.10, 0.7×0.10, 0.3×0.10] = [0.06, 0.07, 0.03]
```

**Topic Embedding (average of top 20 words):**
```
e_5 = mean([weighted vectors for all top 20 words])
   ≈ [0.65, 0.55, 0.45]  (after averaging all 20 words)
```

**Result:**
- Topic 5 is represented as a 100-dimensional vector (in real implementation)
- This vector captures the semantic meaning of the "sports" topic

---

## 4. Topic-Topic Connections (Cosine Similarity)

### 4.1 Cosine Similarity Definition

To measure how similar two topics are, we compute the cosine similarity between their embeddings.

**Cosine Similarity Formula:**
```
sim(e_i, e_j) = (e_i · e_j) / (||e_i|| × ||e_j||)
```

**Expanded Form:**
```
sim(e_i, e_j) = Σ(n=1 to E) [e_i[n] × e_j[n]] / (√(Σ e_i²) × √(Σ e_j²))
```

Where:
- **e_i** = Embedding vector for topic i (dimension E)
- **e_j** = Embedding vector for topic j (dimension E)
- **e_i · e_j** = Dot product (sum of element-wise products)
- **||e_i||** = L2 norm (Euclidean length) of vector e_i
- **||e_j||** = L2 norm (Euclidean length) of vector e_j

**Simple Explanation:**
- Dot product: Multiply corresponding elements and sum them up
- L2 norm: Square each element, sum them, take square root (like distance from origin)
- Cosine similarity: Measures the angle between two vectors
  - 1.0 = Same direction (very similar topics)
  - 0.0 = Perpendicular (unrelated topics)
  - -1.0 = Opposite direction (very different topics)

### 4.2 Step-by-Step Calculation

**Given:**
- Topic 5 embedding: e_5 = [0.8, 0.6, 0.4]
- Topic 7 embedding: e_7 = [0.7, 0.5, 0.3]

**Step 1: Dot Product**
```
e_5 · e_7 = (0.8 × 0.7) + (0.6 × 0.5) + (0.4 × 0.3)
          = 0.56 + 0.30 + 0.12
          = 0.98
```

**Step 2: L2 Norms**
```
||e_5|| = √(0.8² + 0.6² + 0.4²)
        = √(0.64 + 0.36 + 0.16)
        = √1.16
        ≈ 1.077

||e_7|| = √(0.7² + 0.5² + 0.3²)
        = √(0.49 + 0.25 + 0.09)
        = √0.83
        ≈ 0.911
```

**Step 3: Cosine Similarity**
```
sim(e_5, e_7) = 0.98 / (1.077 × 0.911)
              = 0.98 / 0.981
              ≈ 0.999
```

**Interpretation:**
- Topics 5 and 7 are very similar (similarity ≈ 1.0)
- They likely share many common words and semantic meaning

### 4.3 Similarity Matrix

We compute similarity for all pairs of topics:

**Similarity Matrix: S**
```
        topic_1   topic_2   topic_3   ...   topic_K
topic_1 [  1.0     sim_12   sim_13   ...    sim_1K  ]
topic_2 [ sim_21    1.0     sim_23   ...    sim_2K  ]
topic_3 [ sim_31   sim_32    1.0     ...    sim_3K  ]
  ...   [  ...      ...      ...     ...     ...   ]
topic_K [ sim_K1   sim_K2   sim_K3   ...     1.0    ]
```

Where:
- **S_ij** = sim(e_i, e_j) = Cosine similarity between topics i and j
- **S_ii** = 1.0 (topic is identical to itself)
- **S_ij** = **S_ji** (symmetric matrix)

**In Code:**
```python
# Compute similarity matrix for all topic pairs
similarity_matrix = cosine_similarity(self.topic_embeddings)
# Shape: (K, K) - K topics × K topics
```

### 4.4 Threshold Filtering

We only create edges between topics that are sufficiently similar:

```
sim(e_i, e_j) ≥ τ_topic_topic
```

Where:
- **τ_topic_topic** = Topic-topic threshold (default: 0.30)

**Simple Explanation:**
- If two topics have similarity less than 0.30, we don't create an edge
- This keeps the graph focused on meaningful semantic relationships
- Example: If sim(topic_5, topic_7) = 0.25, and threshold = 0.30, then NO edge
- If sim(topic_5, topic_7) = 0.45, then an edge is created with weight 0.45

### 4.5 Edge Creation

**Edge Set:**
```
E_topic_topic = {(i, j) | sim(e_i, e_j) ≥ τ_topic_topic, i < j, i,j ∈ [0, K-1]}
```

**Edge Weight Function:**
```
w: E_topic_topic → [0, 1]
w(i, j) = sim(e_i, e_j)
```

**Constraints:**
- **i < j**: Only create edges once (avoid duplicates)
- **i ≠ j**: No self-loops (topics don't connect to themselves)

**In Code:**
```python
similarity_matrix = cosine_similarity(self.topic_embeddings)

for i in range(self.num_topics):
    for j in range(i + 1, self.num_topics):  # i < j, avoid duplicates
        similarity = similarity_matrix[i, j]  # sim(e_i, e_j)
        if similarity <= self.topic_topic_threshold:  # τ_topic_topic
            continue  # Skip this edge
        
        topic_i = self.num_docs + i
        topic_j = self.num_docs + j
        self.graph.add_edge(topic_i, topic_j, weight=float(similarity))
```

### 4.6 Example Calculation

**Given:**
- Topic 5 (Sports): e_5 = [0.8, 0.6, 0.4, ...]
- Topic 7 (Politics): e_7 = [0.2, 0.1, 0.3, ...]
- Topic 8 (Athletics): e_8 = [0.75, 0.65, 0.45, ...]

**Similarities:**
```
sim(e_5, e_7) = 0.25  (Sports and Politics are somewhat different)
sim(e_5, e_8) = 0.92  (Sports and Athletics are very similar)
sim(e_7, e_8) = 0.18  (Politics and Athletics are different)
```

**Edge Creation (with threshold = 0.30):**
- Edge (topic_5, topic_7): weight = 0.25 ❌ (below threshold, NO edge)
- Edge (topic_5, topic_8): weight = 0.92 ✅ (above threshold, edge created)
- Edge (topic_7, topic_8): weight = 0.18 ❌ (below threshold, NO edge)

**Graph Structure:**
```
topic_5 (Sports) ──0.92── topic_8 (Athletics)
topic_7 (Politics)        (isolated, no connections)
```

---

## 5. Complete Mathematical Pipeline

### 5.1 Summary of All Formulas

**1. LDA Training:**
```
Input:  X (D × V) - Document-term matrix
Output: Θ (D × K) - Document-topic matrix
        Φ (K × V) - Topic-word matrix
```

**2. Topic Embeddings:**
```
For each topic k:
  e_k = (1/N) × Σ(w, prob) ∈ TopWords_k  [v_w × prob]
```

**3. Document-Topic Edges:**
```
E_doc_topic = {(d, k) | θ_dk ≥ τ_doc_topic}
w(d, k) = θ_dk
```

**4. Topic-Topic Edges:**
```
sim(e_i, e_j) = (e_i · e_j) / (||e_i|| × ||e_j||)
E_topic_topic = {(i, j) | sim(e_i, e_j) ≥ τ_topic_topic, i < j}
w(i, j) = sim(e_i, e_j)
```

### 5.2 Complete Example

**Given:**
- D = 3 documents
- K = 2 topics
- V = 5 words
- E = 3 (embedding dimension, simplified)

**Step 1: LDA Output**

**Document-Topic Matrix Θ:**
```
        topic_0  topic_1
doc_0  [ 0.80    0.20  ]
doc_1  [ 0.30    0.70  ]
doc_2  [ 0.60    0.40  ]
```

**Topic-Word Matrix Φ:**
```
        word_0  word_1  word_2  word_3  word_4
topic_0 [ 0.30   0.25   0.20   0.15   0.10  ]
topic_1 [ 0.10   0.15   0.20   0.25   0.30  ]
```

**Step 2: Word Embeddings (Word2Vec output)**
```
v_word0 = [0.8, 0.6, 0.4]
v_word1 = [0.7, 0.5, 0.5]
v_word2 = [0.6, 0.7, 0.3]
v_word3 = [0.5, 0.4, 0.6]
v_word4 = [0.4, 0.3, 0.7]
```

**Step 3: Topic Embeddings**

**Topic 0 (top 2 words: word_0, word_1):**
```
e_0 = (1/2) × [v_word0 × 0.30 + v_word1 × 0.25]
   = (1/2) × [[0.8,0.6,0.4]×0.30 + [0.7,0.5,0.5]×0.25]
   = (1/2) × [[0.24,0.18,0.12] + [0.175,0.125,0.125]]
   = (1/2) × [0.415, 0.305, 0.245]
   = [0.2075, 0.1525, 0.1225]
```

**Topic 1 (top 2 words: word_3, word_4):**
```
e_1 = (1/2) × [v_word3 × 0.25 + v_word4 × 0.30]
   = (1/2) × [[0.5,0.4,0.6]×0.25 + [0.4,0.3,0.7]×0.30]
   = (1/2) × [[0.125,0.1,0.15] + [0.12,0.09,0.21]]
   = (1/2) × [0.245, 0.19, 0.36]
   = [0.1225, 0.095, 0.18]
```

**Step 4: Document-Topic Edges (threshold = 0.02)**

All edges created (all θ_dk ≥ 0.02):
```
doc_0 ──0.80──> topic_0
doc_0 ──0.20──> topic_1
doc_1 ──0.30──> topic_0
doc_1 ──0.70──> topic_1
doc_2 ──0.60──> topic_0
doc_2 ──0.40──> topic_1
```

**Step 5: Topic-Topic Edges (threshold = 0.30)**

**Compute similarity:**
```
e_0 = [0.2075, 0.1525, 0.1225]
e_1 = [0.1225, 0.095, 0.18]

e_0 · e_1 = 0.2075×0.1225 + 0.1525×0.095 + 0.1225×0.18
          = 0.0254 + 0.0145 + 0.0221
          = 0.062

||e_0|| = √(0.2075² + 0.1525² + 0.1225²)
        = √(0.0431 + 0.0233 + 0.0150)
        = √0.0814
        ≈ 0.285

||e_1|| = √(0.1225² + 0.095² + 0.18²)
        = √(0.0150 + 0.0090 + 0.0324)
        = √0.0564
        ≈ 0.238

sim(e_0, e_1) = 0.062 / (0.285 × 0.238)
              = 0.062 / 0.0678
              ≈ 0.914
```

**Edge Creation:**
```
sim(e_0, e_1) = 0.914 ≥ 0.30 ✅
Edge: topic_0 ──0.914── topic_1
```

**Final Graph:**
```
doc_0 ──0.80──> topic_0 ──0.914── topic_1 <──0.70── doc_1
  │                              │
  └──0.20────────────────────────┘
  
doc_2 ──0.60──> topic_0
doc_2 ──0.40──> topic_1
```

### 5.3 Key Insights

1. **Document-Topic Connections:**
   - Directly from LDA output (θ_dk)
   - Represents how much each document belongs to each topic
   - Threshold filters weak connections

2. **Topic Embeddings:**
   - Weighted average of word embeddings
   - Captures semantic meaning of topics
   - Enables similarity computation

3. **Topic-Topic Connections:**
   - Based on semantic similarity (cosine similarity)
   - Connects related topics (e.g., "sports" and "athletics")
   - Threshold filters unrelated topics

4. **Graph Structure:**
   - Documents connect to topics (content-based)
   - Topics connect to each other (semantic-based)
   - Two-layer structure enables information flow

---

## Glossary of Mathematical Terms

| Term | Symbol | Definition | Simple Explanation |
|------|--------|------------|-------------------|
| **Probability** | P(A) | Likelihood of event A | How likely something is to happen (0 to 1) |
| **Distribution** | θ, φ | Function assigning probabilities | A list of probabilities that sum to 1 |
| **Vector** | **v**, **e** | Ordered list of numbers | A list like [0.8, 0.6, 0.4] |
| **Matrix** | **X**, **Θ**, **Φ** | 2D array of numbers | A table of numbers (rows × columns) |
| **Dot Product** | **a · b** | Sum of element-wise products | Multiply corresponding elements and add |
| **L2 Norm** | \|\|**v**\|\| | Euclidean length of vector | Distance from origin (like Pythagorean theorem) |
| **Cosine Similarity** | sim(**a**, **b**) | Measure of angle between vectors | How similar two vectors are (0 to 1) |
| **Threshold** | τ | Minimum value to keep | Cutoff point (below = discard, above = keep) |
| **Summation** | Σ | Sum of values | Add up all values in a list |
| **Normalization** | - | Scaling to sum to 1 | Divide by total so probabilities add to 1 |

---

## References

1. **LDA Paper:** Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.

2. **Word2Vec Paper:** Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

3. **GCN Paper:** Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.

---

*This document explains the mathematical foundations of TopicGCN. For implementation details, see the source code in `topic_model.py` and `build_graph.py`.*

