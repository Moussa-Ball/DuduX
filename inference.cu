/*
DUDUX AI - Interactive Inference System
=======================================

🚀 Bio-Faithful Neural Network Inference Engine
⚡ CUDA Accelerated Interactive Processing
🧠 Real-time Response Generation

Authors: Research Team Dudux
Version: 5.1.0 Inference Protocol
Created: August 4, 2025
*/

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Bio-faithful constants
#define DT 0.0001f                    // 0.1ms time step
#define SIMULATION_TIME 0.02f         // 20ms per word
#define REFRACTORY_PERIOD 0.002f      // 2ms refractory period
#define RESTING_POTENTIAL -0.07f      // -70mV normalized
#define SPIKE_THRESHOLD 1.0f          // Normalized spike threshold
#define ATP_PER_SPIKE 0.0001f         // ATP cost per spike
#define TAU_STDP 20.0f               // STDP time constant
#define A_PLUS 0.01f                 // LTP amplitude
#define A_MINUS 0.012f               // LTD amplitude (slightly larger)
#define GLUCOSE_REGENERATION 0.01f    // Glucose regeneration rate
#define STDP_TAU_PLUS 0.02f          // 20ms STDP window
#define STDP_A_PLUS 0.005f           // LTP amplitude
#define STDP_A_MINUS 0.00525f        // LTD amplitude
#define MAX_SPIKES_PER_NEURON 100    // Maximum spikes to record
#define MAX_NEURONS 10000            // Maximum neurons
#define MAX_VOCABULARY 1000          // Maximum vocabulary size

// Include all the same structures as model.cu
// Bio-faithful neuron structure with all required fields
typedef struct {
    float membrane_potential;
    float threshold;
    float base_threshold;
    float last_spike_time;
    float atp_level;
    float pre_trace;
    float post_trace;
    int spike_count;
    float spike_times[MAX_SPIKES_PER_NEURON];
    
    // Additional bio-faithful fields
    float resting_potential;
    float spike_threshold;
    float adaptation_current;
    float membrane_time_constant;
    int n_spikes;
    float spike_frequency;
    float stdp_trace_pre;
    float stdp_trace_post;
    float synaptic_efficacy;
    float energy_consumed;
    float energy_available;
} BioNeuron;

// SDR pattern structure
typedef struct {
    int n_neurons;
    int n_active;
    float sparsity;
    int active_indices[300];  // Max 3% of 10000 = 300
    float spike_times[300];
    float energy_cost;
} SDRPattern;

// Synaptic connection with STDP
typedef struct {
    int pre_neuron;
    int post_neuron;
    float weight;
    float last_update;
    float plasticity_trace;
    float energy_efficiency;
} SynapticConnection;

// Hierarchical layer structure
typedef struct {
    int layer_id;
    int n_neurons;
    float* membrane_potential;
    float* predicted_activity;
    float prediction_error;
    int n_active_neurons;
    float layer_energy;
    float adaptation_rate;
} LayerData;

typedef struct {
    int input_size;
    int output_size;
    int layer_id;
    float prediction_accuracy;
    float energy_consumption;
    float glucose_supply;
} HierarchicalLayer;

// Contextual memory episode
typedef struct {
    int pattern_ids[50];  // Max 50 patterns per episode
    int episode_length;
    float episode_time;
    float prediction_score;
} MemoryEpisode;

// Syntactic pattern
typedef struct {
    char pattern_name[32];
    int frequency;
    float confidence;
    int word_positions[10];  // Max 10 words in pattern
    int pattern_length;
} SyntacticPattern;

// Semantic cluster
typedef struct {
    char cluster_name[32];
    int word_indices[100];  // Max 100 words per cluster
    int cluster_size;
    float semantic_coherence;
} SemanticCluster;

// Memory structures
struct EpisodicMemory {
    float episode_start_time;
    float episode_duration;
    int n_active_patterns;
    int pattern_indices[50];  // References to SDR patterns
    float emotional_valence;
    float consolidation_strength;
    float replay_frequency;
    char context_tag[32];
};

struct ConsolidationBuffer {
    EpisodicMemory episodes[100];
    int n_episodes;
    float* consolidation_weights;
    float* replay_activations;
    float global_consolidation_rate;
    int consolidation_cycles;
};

// Global device variables
__device__ curandState* d_rand_states;

// Include all the same CUDA kernels from model.cu
// (I'll include the key ones for inference)

__global__ void initialize_neurons_kernel(BioNeuron* neurons, int n_neurons, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;
    
    // Initialize cuRAND state
    curand_init(seed, idx, 0, &d_rand_states[idx]);
    
    // Initialize neuron parameters
    neurons[idx].membrane_potential = RESTING_POTENTIAL;
    neurons[idx].threshold = SPIKE_THRESHOLD;
    neurons[idx].base_threshold = SPIKE_THRESHOLD;
    neurons[idx].last_spike_time = -1000.0f;
    neurons[idx].atp_level = 1.0f;
    neurons[idx].pre_trace = 0.0f;
    neurons[idx].post_trace = 0.0f;
    neurons[idx].spike_count = 0;
}

__global__ void create_temporal_sdr_kernel(SDRPattern* patterns, int pattern_idx, 
                                          int n_neurons, float sparsity, unsigned long word_seed,
                                          SemanticCluster* semantic_clusters, int cluster_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    SDRPattern* pattern = &patterns[pattern_idx];
    
    if (idx == 0) {
        pattern->n_neurons = n_neurons;
        pattern->sparsity = sparsity;
        pattern->n_active = (int)(n_neurons * sparsity);
        pattern->energy_cost = pattern->n_active * ATP_PER_SPIKE;
    }
    
    __syncthreads();
    
    if (idx < pattern->n_active) {
        curandState state;
        curand_init(word_seed, idx, 0, &state);
        
        bool valid = false;
        int attempts = 0;
        while (!valid && attempts < 100) {
            int neuron_idx = curand(&state) % n_neurons;
            
            valid = true;
            for (int i = 0; i < idx; i++) {
                if (pattern->active_indices[i] == neuron_idx) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                pattern->active_indices[idx] = neuron_idx;
                pattern->spike_times[idx] = 0.001f * idx;  // Staggered timing
            }
            attempts++;
        }
    }
}

__global__ void spike_dynamics_kernel(BioNeuron* neurons, float* input_currents, 
                                    float* output_spikes, int n_neurons, 
                                    float current_time, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_neurons) return;
    
    BioNeuron* neuron = &neurons[idx];
    
    // Check refractory period
    if (current_time - neuron->last_spike_time < REFRACTORY_PERIOD) {
        output_spikes[idx] = 0.0f;
        return;
    }
    
    // Membrane dynamics
    float leak_current = 0.1f * (neuron->membrane_potential - RESTING_POTENTIAL);
    float dv_dt = (input_currents[idx] - leak_current) / 1.0f;
    neuron->membrane_potential += dv_dt * dt;
    
    // Threshold adaptation decay
    neuron->threshold = neuron->base_threshold + 
                       (neuron->threshold - neuron->base_threshold) * 0.95f;
    
    // Check for spike
    if (neuron->membrane_potential >= neuron->threshold && neuron->atp_level > 0.1f) {
        if (neuron->spike_count < MAX_SPIKES_PER_NEURON) {
            neuron->spike_times[neuron->spike_count] = current_time;
            neuron->spike_count++;
        }
        
        neuron->membrane_potential = RESTING_POTENTIAL;
        neuron->threshold += 0.02f;
        neuron->atp_level -= ATP_PER_SPIKE;
        neuron->post_trace = 1.0f;
        neuron->last_spike_time = current_time;
        output_spikes[idx] = 1.0f;
    } else {
        output_spikes[idx] = 0.0f;
    }
    
    // Update STDP traces
    neuron->pre_trace *= (1.0f - dt * 50.0f);
    neuron->post_trace *= (1.0f - dt * 50.0f);
}

__global__ void apply_input_spikes_kernel(float* input_currents, SDRPattern* pattern,
                                        float current_time, float spike_window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pattern->n_active) return;
    
    int neuron_idx = pattern->active_indices[idx];
    float spike_time = pattern->spike_times[idx];
    
    if (fabsf(current_time - spike_time) < spike_window) {
        float amplitude = expf(-fabsf(current_time - spike_time) / (spike_window * 0.1f));
        atomicAdd(&input_currents[neuron_idx], amplitude * 5.0f);
    }
}

__global__ void hierarchical_processing_kernel(float* input_layer, float* output_layer,
                                              float* weights, int input_size, int output_size,
                                              float sparsity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size) return;
    
    float activation = 0.0f;
    for (int i = 0; i < input_size; i++) {
        activation += weights[idx * input_size + i] * input_layer[i];
    }
    
    float threshold = 1.0f;
    output_layer[idx] = (activation > threshold) ? 1.0f : 0.0f;
}

// Inference-specific class
class BioFaithfulInference {
private:
    int n_neurons_;
    int n_layers_;
    int* layer_sizes_;
    
    BioNeuron* d_neurons_;
    float* d_input_currents_;
    float* d_output_spikes_;
    curandState* d_rand_states_;
    SDRPattern* d_sdr_patterns_;
    
    SynapticConnection* d_connections_;
    HierarchicalLayer* d_layers_;
    MemoryEpisode* d_episodes_;
    SyntacticPattern* d_syntax_patterns_;
    SemanticCluster* d_semantic_clusters_;
    
    float** d_layer_activations_;
    float** d_layer_weights_;
    
    float global_time_;
    float global_energy_budget_;
    float glucose_level_;
    
    BioNeuron* h_neurons_;
    float* h_output_spikes_;
    SyntacticPattern* h_syntax_patterns_;
    SemanticCluster* h_semantic_clusters_;
    
    int current_episode_;
    int vocabulary_size_;
    char vocabulary_[MAX_VOCABULARY][32];
    
public:
    BioFaithfulInference(int n_neurons, int* hierarchy_sizes, int n_layers) 
        : n_neurons_(n_neurons), n_layers_(n_layers), global_time_(0.0f), 
          global_energy_budget_(1000.0f), glucose_level_(1.0f), current_episode_(0), vocabulary_size_(0) {
        
        printf("🧠 Initializing Bio-Faithful Inference Engine...\n");
        
        layer_sizes_ = (int*)malloc(n_layers_ * sizeof(int));
        for (int i = 0; i < n_layers_; i++) {
            layer_sizes_[i] = hierarchy_sizes[i];
        }
        
        allocate_gpu_memory();
        initialize_network();
        
        printf("✅ Inference Engine Ready!\n");
    }
    
    ~BioFaithfulInference() {
        // Free GPU memory
        cudaFree(d_neurons_);
        cudaFree(d_input_currents_);
        cudaFree(d_output_spikes_);
        cudaFree(d_rand_states_);
        cudaFree(d_sdr_patterns_);
        cudaFree(d_connections_);
        cudaFree(d_layers_);
        cudaFree(d_episodes_);
        cudaFree(d_syntax_patterns_);
        cudaFree(d_semantic_clusters_);
        
        for (int i = 0; i < n_layers_; i++) {
            cudaFree(d_layer_activations_[i]);
            cudaFree(d_layer_weights_[i]);
        }
        free(d_layer_activations_);
        free(d_layer_weights_);
        
        free(h_neurons_);
        free(h_output_spikes_);
        free(h_syntax_patterns_);
        free(h_semantic_clusters_);
        free(layer_sizes_);
    }
    
    void allocate_gpu_memory() {
        size_t neurons_size = n_neurons_ * sizeof(BioNeuron);
        size_t currents_size = n_neurons_ * sizeof(float);
        size_t spikes_size = n_neurons_ * sizeof(float);
        size_t rand_size = n_neurons_ * sizeof(curandState);
        size_t pattern_size = MAX_VOCABULARY * sizeof(SDRPattern);
        size_t connections_size = n_neurons_ * 100 * sizeof(SynapticConnection);
        size_t layers_size = n_layers_ * sizeof(HierarchicalLayer);
        size_t episodes_size = 1000 * sizeof(MemoryEpisode);
        size_t syntax_size = 100 * sizeof(SyntacticPattern);
        size_t clusters_size = 50 * sizeof(SemanticCluster);
        
        CUDA_CHECK(cudaMalloc(&d_neurons_, neurons_size));
        CUDA_CHECK(cudaMalloc(&d_input_currents_, currents_size));
        CUDA_CHECK(cudaMalloc(&d_output_spikes_, spikes_size));
        CUDA_CHECK(cudaMalloc(&d_rand_states_, rand_size));
        CUDA_CHECK(cudaMalloc(&d_sdr_patterns_, pattern_size));
        CUDA_CHECK(cudaMalloc(&d_connections_, connections_size));
        CUDA_CHECK(cudaMalloc(&d_layers_, layers_size));
        CUDA_CHECK(cudaMalloc(&d_episodes_, episodes_size));
        CUDA_CHECK(cudaMalloc(&d_syntax_patterns_, syntax_size));
        CUDA_CHECK(cudaMalloc(&d_semantic_clusters_, clusters_size));
        
        d_layer_activations_ = (float**)malloc(n_layers_ * sizeof(float*));
        d_layer_weights_ = (float**)malloc(n_layers_ * sizeof(float*));
        
        int prev_size = n_neurons_;
        for (int i = 0; i < n_layers_; i++) {
            int layer_size = layer_sizes_[i];
            CUDA_CHECK(cudaMalloc(&d_layer_activations_[i], layer_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_layer_weights_[i], prev_size * layer_size * sizeof(float)));
            prev_size = layer_size;
        }
        
        h_neurons_ = (BioNeuron*)malloc(neurons_size);
        h_output_spikes_ = (float*)malloc(spikes_size);
        h_syntax_patterns_ = (SyntacticPattern*)malloc(syntax_size);
        h_semantic_clusters_ = (SemanticCluster*)malloc(clusters_size);
        
        CUDA_CHECK(cudaMemset(d_input_currents_, 0, currents_size));
        CUDA_CHECK(cudaMemset(d_output_spikes_, 0, spikes_size));
    }
    
    void initialize_network() {
        int block_size = 256;
        int grid_size = (n_neurons_ + block_size - 1) / block_size;
        
        unsigned long seed = (unsigned long)time(NULL);
        
        CUDA_CHECK(cudaMemcpyToSymbol(d_rand_states, &d_rand_states_, sizeof(curandState*)));
        
        initialize_neurons_kernel<<<grid_size, block_size>>>(d_neurons_, n_neurons_, seed);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        for (int i = 0; i < n_layers_; i++) {
            int prev_size = (i == 0) ? n_neurons_ : layer_sizes_[i-1];
            int current_size = layer_sizes_[i];
            
            float* h_weights = (float*)malloc(prev_size * current_size * sizeof(float));
            for (int j = 0; j < prev_size * current_size; j++) {
                h_weights[j] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
            }
            
            CUDA_CHECK(cudaMemcpy(d_layer_weights_[i], h_weights, 
                                prev_size * current_size * sizeof(float), cudaMemcpyHostToDevice));
            free(h_weights);
        }
    }
    
    // Model persistence (same as training)
    struct ModelHeader {
        char magic[8];           // "DUDUXAI\0"
        int version;             // Model version
        int n_neurons;           // Number of neurons
        int n_layers;            // Number of layers
        int vocabulary_size;     // Vocabulary size
        float global_time;       // Simulation time
        float global_energy;     // Energy budget
        int current_episode;     // Episode count
        char reserved[64];       // Future use
    };
    
    bool load_model(const char* filename = "model.bin") {
        printf("📚 Loading trained model from %s...\n", filename);
        
        FILE* file = fopen(filename, "rb");
        if (!file) {
            printf("❌ Model file not found! Please train first with './model'\n");
            return false;
        }
        
        // Read and verify header
        ModelHeader header;
        if (fread(&header, sizeof(ModelHeader), 1, file) != 1) {
            printf("❌ Invalid model file header\n");
            fclose(file);
            return false;
        }
        
        if (strcmp(header.magic, "DUDUXAI") != 0) {
            printf("❌ Invalid model file format\n");
            fclose(file);
            return false;
        }
        
        if (header.n_neurons != n_neurons_ || header.n_layers != n_layers_) {
            printf("❌ Model architecture mismatch\n");
            fclose(file);
            return false;
        }
        
        // Load vocabulary
        vocabulary_size_ = header.vocabulary_size;
        if (vocabulary_size_ > MAX_VOCABULARY) {
            printf("❌ Vocabulary too large\n");
            fclose(file);
            return false;
        }
        fread(vocabulary_, sizeof(char[32]), vocabulary_size_, file);
        
        // Load layer sizes (verify)
        int* loaded_sizes = (int*)malloc(n_layers_ * sizeof(int));
        fread(loaded_sizes, sizeof(int), n_layers_, file);
        for (int i = 0; i < n_layers_; i++) {
            if (loaded_sizes[i] != layer_sizes_[i]) {
                printf("❌ Layer size mismatch at layer %d\n", i);
                free(loaded_sizes);
                fclose(file);
                return false;
            }
        }
        free(loaded_sizes);
        
        // Load neuron states
        fread(h_neurons_, sizeof(BioNeuron), n_neurons_, file);
        CUDA_CHECK(cudaMemcpy(d_neurons_, h_neurons_, n_neurons_ * sizeof(BioNeuron), cudaMemcpyHostToDevice));
        
        // Load layer weights
        for (int i = 0; i < n_layers_; i++) {
            int prev_size = (i == 0) ? n_neurons_ : layer_sizes_[i-1];
            int current_size = layer_sizes_[i];
            
            float* h_weights = (float*)malloc(prev_size * current_size * sizeof(float));
            fread(h_weights, sizeof(float), prev_size * current_size, file);
            CUDA_CHECK(cudaMemcpy(d_layer_weights_[i], h_weights, 
                                prev_size * current_size * sizeof(float), cudaMemcpyHostToDevice));
            free(h_weights);
        }
        
        // Load syntax patterns
        fread(h_syntax_patterns_, sizeof(SyntacticPattern), 100, file);
        CUDA_CHECK(cudaMemcpy(d_syntax_patterns_, h_syntax_patterns_, 100 * sizeof(SyntacticPattern), cudaMemcpyHostToDevice));
        
        // Load semantic clusters
        fread(h_semantic_clusters_, sizeof(SemanticCluster), 50, file);
        CUDA_CHECK(cudaMemcpy(d_semantic_clusters_, h_semantic_clusters_, 50 * sizeof(SemanticCluster), cudaMemcpyHostToDevice));
        
        // Restore state
        global_time_ = header.global_time;
        global_energy_budget_ = header.global_energy;
        current_episode_ = header.current_episode;
        
        fclose(file);
        
        printf("✅ Model loaded successfully!\n");
        printf("   📚 Vocabulary: %d words\n", vocabulary_size_);
        printf("   ⏱️  Training time: %.3f s\n", global_time_);
        printf("   🎯 Episodes: %d\n", current_episode_);
        
        return true;
    }
    
    // Process input prompt and generate response
    float process_prompt(const char* prompt) {
        clock_t start = clock();
        
        printf("\n🧠 Processing: '%s'\n", prompt);
        
        // Parse prompt into words
        char prompt_copy[512];
        strcpy(prompt_copy, prompt);
        char* words[50];
        int word_count = 0;
        
        char* token = strtok(prompt_copy, " ");
        while (token != nullptr && word_count < 50) {
            words[word_count] = token;
            word_count++;
            token = strtok(nullptr, " ");
        }
        
        // Process each word
        for (int i = 0; i < word_count; i++) {
            SDRPattern pattern = create_sdr_cuda(words[i]);
            simulate_word_spikes_cuda(&pattern);
            process_hierarchical_layers();
        }
        
        clock_t end = clock();
        float time_ms = ((float)(end - start) / CLOCKS_PER_SEC) * 1000.0f;
        
        return time_ms;
    }
    
    SDRPattern create_sdr_cuda(const char* word, float sparsity = 0.03f) {
        unsigned long word_hash = 0;
        for (int i = 0; word[i] != '\0'; i++) {
            word_hash = word_hash * 31 + word[i];
        }
        
        int block_size = 256;
        int n_active = (int)(n_neurons_ * sparsity);
        int grid_size = (n_active + block_size - 1) / block_size;
        
        SDRPattern* d_pattern = d_sdr_patterns_;
        
        create_temporal_sdr_kernel<<<grid_size, block_size>>>(d_sdr_patterns_, 0,
                                                               n_neurons_, sparsity, word_hash,
                                                               d_semantic_clusters_, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        SDRPattern h_pattern;
        CUDA_CHECK(cudaMemcpy(&h_pattern, d_pattern, sizeof(SDRPattern), cudaMemcpyDeviceToHost));
        
        return h_pattern;
    }
    
    void simulate_word_spikes_cuda(SDRPattern* pattern) {
        int block_size = 256;
        int grid_size = (n_neurons_ + block_size - 1) / block_size;
        int input_grid = (pattern->n_active + block_size - 1) / block_size;
        
        int time_steps = (int)(SIMULATION_TIME / DT);
        
        CUDA_CHECK(cudaMemcpy(d_sdr_patterns_, pattern, sizeof(SDRPattern), cudaMemcpyHostToDevice));
        
        for (int step = 0; step < time_steps; step++) {
            float current_time = step * DT;
            
            CUDA_CHECK(cudaMemset(d_input_currents_, 0, n_neurons_ * sizeof(float)));
            
            apply_input_spikes_kernel<<<input_grid, block_size>>>(
                d_input_currents_, d_sdr_patterns_, current_time, 0.005f);
            
            spike_dynamics_kernel<<<grid_size, block_size>>>(
                d_neurons_, d_input_currents_, d_output_spikes_, n_neurons_, current_time, DT);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        global_time_ += SIMULATION_TIME;
    }
    
    void process_hierarchical_layers() {
        int block_size = 256;
        
        for (int layer_idx = 0; layer_idx < n_layers_; layer_idx++) {
            int input_size = (layer_idx == 0) ? n_neurons_ : layer_sizes_[layer_idx - 1];
            int output_size = layer_sizes_[layer_idx];
            int grid_size = (output_size + block_size - 1) / block_size;
            
            float* input_layer = (layer_idx == 0) ? d_output_spikes_ : d_layer_activations_[layer_idx - 1];
            float* output_layer = d_layer_activations_[layer_idx];
            float* weights = d_layer_weights_[layer_idx];
            
            hierarchical_processing_kernel<<<grid_size, block_size>>>(
                input_layer, output_layer, weights, input_size, output_size, 0.03f);
            
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    void print_vocabulary() {
        printf("\n📚 Learned Vocabulary (%d words):\n", vocabulary_size_);
        printf("================================\n");
        for (int i = 0; i < vocabulary_size_; i++) {
            printf("   %d. %s\n", i + 1, vocabulary_[i]);
        }
    }
    
    void analyze_prompt(const char* prompt) {
        printf("\n🔍 NEURAL ANALYSIS\n");
        printf("==================\n");
        
        // Copy current neuron states
        CUDA_CHECK(cudaMemcpy(h_neurons_, d_neurons_, n_neurons_ * sizeof(BioNeuron), cudaMemcpyDeviceToHost));
        
        int active_neurons = 0;
        int total_spikes = 0;
        float avg_potential = 0.0f;
        
        for (int i = 0; i < n_neurons_; i++) {
            if (h_neurons_[i].spike_count > 0) active_neurons++;
            total_spikes += h_neurons_[i].spike_count;
            avg_potential += h_neurons_[i].membrane_potential;
        }
        avg_potential /= n_neurons_;
        
        float sparsity = 1.0f - ((float)active_neurons / n_neurons_);
        float firing_rate = total_spikes / (global_time_ + 0.001f);
        
        printf("🧠 Network Response:\n");
        printf("   Active neurons: %d/%d (%.1f%%)\n", active_neurons, n_neurons_, 100.0f * active_neurons / n_neurons_);
        printf("   Total spikes: %d\n", total_spikes);
        printf("   Sparsity: %.1f%%\n", sparsity * 100);
        printf("   Firing rate: %.2f Hz\n", firing_rate);
        printf("   Avg membrane potential: %.3f\n", avg_potential);
        
        // Simple semantic analysis
        printf("\n🧩 Semantic Analysis:\n");
        char prompt_copy[512];
        strcpy(prompt_copy, prompt);
        char* words[50];
        int word_count = 0;
        
        char* token = strtok(prompt_copy, " ");
        while (token != nullptr && word_count < 50) {
            words[word_count] = token;
            word_count++;
            token = strtok(nullptr, " ");
        }
        
        int known_words = 0;
        for (int i = 0; i < word_count; i++) {
            bool found = false;
            for (int j = 0; j < vocabulary_size_; j++) {
                if (strcmp(words[i], vocabulary_[j]) == 0) {
                    found = true;
                    break;
                }
            }
            if (found) known_words++;
        }
        
        float vocabulary_coverage = (float)known_words / word_count;
        printf("   Words recognized: %d/%d (%.1f%%)\n", known_words, word_count, vocabulary_coverage * 100);
        
        if (vocabulary_coverage > 0.8f) {
            printf("   🟢 High familiarity - Strong neural response\n");
        } else if (vocabulary_coverage > 0.5f) {
            printf("   🟡 Moderate familiarity - Learning new patterns\n");
        } else {
            printf("   🔴 Low familiarity - Requires more training\n");
        }
    }
    
    void print_stats() {
        printf("\n📊 INFERENCE ENGINE STATUS\n");
        printf("==========================\n");
        printf("🧠 Neurons: %d\n", n_neurons_);
        printf("🏗️ Layers: %d\n", n_layers_);
        printf("📚 Vocabulary: %d words\n", vocabulary_size_);
        printf("⏱️ Total inference time: %.3f s\n", global_time_);
        printf("🎯 Episodes processed: %d\n", current_episode_);
    }
};

// Interactive inference main program
int main() {
    printf("🚀 DUDUX AI - INTERACTIVE INFERENCE SYSTEM\n");
    printf("==========================================\n");
    printf("🧠 Bio-Faithful Neural Network Inference Engine\n");
    printf("⚡ CUDA Accelerated Real-time Processing\n\n");
    
    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        printf("❌ No CUDA devices found!\n");
        return 1;
    }
    
    try {
        // Define hierarchical architecture (same as training)
        int hierarchy_sizes[] = {1000, 500, 200};
        int n_layers = 3;
        
        // Initialize inference engine
        BioFaithfulInference inference(2000, hierarchy_sizes, n_layers);
        
        // Load trained model
        if (!inference.load_model("model.bin")) {
            printf("\n💡 No trained model found. Please train first:\n");
            printf("   ./model\n");
            printf("   Then run inference again.\n");
            return 1;
        }
        
        printf("\n🎯 INTERACTIVE MODE ACTIVATED\n");
        printf("=============================\n");
        printf("💡 Commands:\n");
        printf("   Type any sentence to analyze\n");
        printf("   'vocab' - Show learned vocabulary\n");
        printf("   'stats' - Show engine statistics\n");
        printf("   'quit' or 'exit' - Exit program\n");
        printf("\n🚀 Ready for prompts! Type something:\n");
        
        char input[512];
        int prompt_count = 0;
        
        while (true) {
            printf("\n> ");
            fflush(stdout);
            
            if (!fgets(input, sizeof(input), stdin)) {
                break;
            }
            
            // Remove newline
            input[strcspn(input, "\n")] = 0;
            
            // Check for commands
            if (strlen(input) == 0) {
                continue;
            } else if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
                printf("👋 Goodbye! Thanks for using DUDUX AI!\n");
                break;
            } else if (strcmp(input, "vocab") == 0) {
                inference.print_vocabulary();
                continue;
            } else if (strcmp(input, "stats") == 0) {
                inference.print_stats();
                continue;
            } else if (strcmp(input, "help") == 0) {
                printf("\n💡 Available Commands:\n");
                printf("   vocab - Show learned vocabulary\n");
                printf("   stats - Show engine statistics\n");
                printf("   help  - Show this help\n");
                printf("   quit/exit - Exit program\n");
                printf("\n   Or type any sentence for neural analysis!\n");
                continue;
            }
            
            // Process the prompt
            prompt_count++;
            printf("\n⚡ Processing prompt #%d...\n", prompt_count);
            
            float processing_time = inference.process_prompt(input);
            
            printf("✅ Processed in %.2f ms\n", processing_time);
            
            // Analyze the neural response
            inference.analyze_prompt(input);
            
            printf("\n==================================================");
        }
        
        printf("\n📊 SESSION SUMMARY\n");
        printf("==================\n");
        printf("🎯 Prompts processed: %d\n", prompt_count);
        inference.print_stats();
        
    } catch (...) {
        printf("❌ Error during inference session\n");
        return 1;
    }
    
    return 0;
}
