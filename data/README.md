# Training Data Overview for DUDUX Bio-Faithful Neural Network
# Created: August 4, 2025
# Version: 1.0

## Dataset Structure

### 1. qa_dataset.txt
- **Purpose**: Basic question-answer patterns
- **Content**: 50+ knowledge queries across multiple domains
- **Format**: Single line questions and conversational pairs
- **Categories**: Science, Math, Technology, Philosophy, Problem Solving

### 2. complex_sentences.txt
- **Purpose**: Advanced syntactic and semantic patterns
- **Content**: 60+ complex sentences testing cognitive abilities
- **Format**: Multi-clause sentences with logical relationships
- **Categories**: Logic, Causality, Abstracts, Temporal, Hierarchical, Analogies

### 3. scientific_knowledge.txt
- **Purpose**: Structured scientific concepts
- **Content**: 40+ scientific statements and explanations
- **Format**: Domain-specific factual statements
- **Categories**: Physics, Biology, Chemistry, Mathematics, CS, Neuroscience, AI

### 4. dialogue_dataset.txt
- **Purpose**: Conversational interaction patterns
- **Content**: 20+ human-AI dialogue exchanges
- **Format**: Turn-by-turn conversation pairs
- **Categories**: Greetings, Questions, Explanations, Meta-discussion

## Training Strategy for Bio-Faithful Network

### Neural Network Architecture
- **Neurons**: 2000 (1000→500→200 hierarchy)
- **Learning**: STDP (Spike-Timing Dependent Plasticity)
- **Sparsity**: 3% active neurons per pattern
- **Temporal**: 0.1ms precision simulation

### Learning Process
1. **Word-level SDR Creation**: Each word generates sparse distributed representation
2. **Sentence Processing**: Sequential temporal patterns with 20ms per word
3. **STDP Learning**: Synaptic weights adapt based on spike timing
4. **Hierarchical Abstraction**: Multi-layer pattern extraction
5. **Memory Consolidation**: Episode storage and replay

### Expected Capabilities
- **Pattern Recognition**: Identify linguistic and semantic patterns
- **Temporal Sequences**: Process and predict word sequences
- **Syntactic Emergence**: Discover grammatical structures
- **Semantic Clustering**: Group related concepts
- **Contextual Understanding**: Maintain conversation context

## Usage Instructions

1. **Training**: Use `./model` to train on all datasets
2. **Inference**: Use `./inference` for interactive testing
3. **Persistence**: All learning saved in `model.bin`
4. **Expansion**: Add new data files in `/data` directory

## Bio-Faithful Features

- ✅ **Membrane Dynamics**: Realistic neuron simulation
- ✅ **Energy Metabolism**: ATP consumption and glucose regeneration
- ✅ **STDP Learning**: Biologically accurate synaptic plasticity
- ✅ **Sparse Coding**: 3% activation matches cortical sparsity
- ✅ **Hierarchical Processing**: Multi-layer abstraction
- ✅ **Temporal Precision**: Sub-millisecond timing accuracy

This dataset enables the neural network to develop genuine intelligence through bio-faithful learning mechanisms, not hardcoded rules or lookup tables.
