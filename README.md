![logo](logo.png)

**[English](README.md) | [简体中文](README_CN.md)**

# 🤖 Introduction

The Intern-S1 technical report mentions that the model is based on continued pretraining of Qwen3 and InternVL-ViT. However, the paper does not yet contain detailed technical introduction about Intern-S1-Pro. Therefore, this project explores the config files, modeling code, and other aspects of two multimodal MoE models: Qwen3-VL-235B-A22B-Instruct and Intern-S1-Pro, and provides comparative analysis.

**TL;DR**: Intern-S1-Pro scales from 235B to 1T parameters by expanding MoE experts from 128 to 512 (4x) while maintaining similar inference efficiency (both ~22B activated parameters). Whether and how the expanded experts reuse Qwen3-235B-A22B expert weights remains unknown.

## 📦 Project Structure

```
intern-s1-pro-vs-qwen3-235b-a22b/
├── checkpoints/
│   ├── Intern-S1-Pro/              # Weights not included
│   │   ├── config.json             # Model configuration
│   │   ├── modeling_*.py           # Model implementation
│   │   └── ...
│   └── Qwen3-VL-235B-A22B-Instruct/  # Weights not included
│       ├── config.json
│       ├── modeling_*.py
│       └── ...
├── logo.png
├── README.md                       # This file (English)
└── README_CN.md                    # Chinese version
```

## 💡 Model Comparison Overview

| Feature | Intern-S1-Pro | Qwen3-VL-235B-A22B |
|---------|--------------|-------------------|
| **Total Parameters** | **~920B (≈1T)** | **~235B** |
| **Activated Parameters** | 22B | 22B |
| **MoE Experts** | **512** | **128** |
| **Experts per Token** | 8 | 8 |
| **Hidden Layers** | 94 | 94 |
| **Hidden Size** | 4096 | 4096 |
| **Attention Heads** | 64 | 64 |
| **KV Heads** | 4 (GQA) | 4 (GQA) |
| **Context Length** | 262K | 262K |

## ⚙️ Detailed Config Comparison

### Text Model Configuration (text_config)

| Config Item | Intern-S1-Pro | Qwen3-VL-235B | Description |
|------------|--------------|---------------|-------------|
| **Base Architecture** |  |  |  |
| model_type | `interns1_pro_text` | `qwen3_vl_moe_text` | Model type identifier |
| hidden_size | 4096 | 4096 | Hidden dimension |
| num_hidden_layers | 94 | 94 | Number of transformer layers |
| intermediate_size | 12288 | 12288 | FFN intermediate dimension |
| **Attention Mechanism** |  |  |  |
| num_attention_heads | 64 | 64 | Number of attention heads |
| num_key_value_heads | 4 | 4 | Number of KV heads (GQA) |
| head_dim | 128 | 128 | Dimension per attention head |
| attention_dropout | 0.0 | 0.0 | Attention dropout rate |
| attention_bias | false | false | Whether to use attention bias |
| **MoE Configuration** ⭐ |  |  |  |
| **num_experts** | **512** | **128** | 🔥 Total experts (4x difference) |
| num_experts_per_tok | 8 | 8 | Experts activated per token |
| moe_intermediate_size | 1536 | 1536 | Intermediate size per expert |
| **router_n_groups** | **8** | ❌ | 🔥 Router groups (Intern-specific) |
| norm_topk_prob | true | true | Normalize top-k probabilities |
| decoder_sparse_step | 1 | 1 | Decoder sparse step |
| **Vocabulary & Tokens** |  |  |  |
| vocab_size | 155,008 | 151,936 | Vocabulary size |
| bos_token_id | 151643 | 151643 | Beginning of sequence token |
| eos_token_id | 151645 | 151645 | End of sequence token |
| **Position Encoding** |  |  |  |
| rope_theta | 5,000,000 | 5,000,000 | RoPE base frequency |
| rope_type | `default` | `default` | RoPE type |
| **fope_init_factor** | **0.5** | ❌ | 🔥 FoPE init factor (Intern-specific) |
| **fope_sep_head** | **true** | ❌ | 🔥 FoPE separate heads (Intern-specific) |
| **mrope_interleaved** | ❌ | **true** | 🔥 Interleaved MRoPE (Qwen-specific) |
| **mrope_section** | ❌ | **[24, 20, 20]** | 🔥 MRoPE sections (Qwen-specific) |
| max_position_embeddings | 262,144 | 262,144 | Max position embeddings (256K) |
| **Normalization** |  |  |  |
| rms_norm_eps | 1e-06 | 1e-06 | RMSNorm epsilon |
| **Other** |  |  |  |
| hidden_act | `silu` | `silu` | Activation function |
| initializer_range | 0.02 | 0.02 | Parameter initialization range |
| mlp_only_layers | [] | [] | MLP-only layers |
| use_cache | true | true | Use KV cache |
| dtype | bfloat16 | bfloat16 | Data type |

### Vision Model Configuration (vision_config)

| Config Item | Intern-S1-Pro | Qwen3-VL-235B | Description |
|------------|--------------|---------------|-------------|
| **Base Architecture** |  |  |  |
| model_type | `interns1_pro_vision` | `qwen3_vl_moe` | Vision model type |
| **depth** | **24** | **27** | Number of ViT layers |
| **hidden_size** | **1024** | **1152** | Hidden dimension |
| **intermediate_size** | **4096** | **4304** | FFN intermediate dimension |
| num_heads | 16 | 16 | Number of attention heads |
| **Patch Configuration** |  |  |  |
| patch_size | 16 | 16 | Image patch size |
| temporal_patch_size | 2 | 2 | Temporal patch size (video) |
| spatial_merge_size | 2 | 2 | Spatial merge size |
| in_channels | 3 | 3 | Input channels (RGB) |
| **Feature Fusion** |  |  |  |
| **deepstack_visual_indexes** | ❌ | **[8, 16, 24]** | 🔥 DeepStack layer indices (Qwen-specific) |
| **Output Configuration** |  |  |  |
| out_hidden_size | 4096 | 4096 | Output hidden size (aligned with text) |
| num_position_embeddings | 2304 | 2304 | Number of position embeddings |
| **Other** |  |  |  |
| hidden_act | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` | Activation function |
| initializer_range | 0.02 | 0.02 | Parameter initialization range |

### Quantization Configuration (quantization_config)

| Config Item | Intern-S1-Pro | Qwen3-VL-235B | Description |
|------------|--------------|---------------|-------------|
| **Quantization Method** | **FP8** | ❌ | Intern-S1-Pro uses FP8 quantization |
| quant_method | `fp8` | - | Quantization method |
| fmt | `e4m3` | - | FP8 format (4-bit exponent, 3-bit mantissa) |
| scale_fmt | `ue8m0` | - | Scale factor format |
| weight_block_size | [128, 128] | - | Weight block size |
| activation_scheme | `dynamic` | - | Activation quantization scheme |
| modules_to_not_convert | 698 modules | - | List of modules not to convert |

### Special Token Configuration

| Token | Intern-S1-Pro | Qwen3-VL-235B | Description |
|-------|--------------|---------------|-------------|
| image_token_id | 151655 | 151655 | Image placeholder |
| video_token_id | 151656 | 151656 | Video placeholder |
| vision_start_token_id | 151652 | 151652 | Vision start token |
| vision_end_token_id | 151653 | 151653 | Vision end token |

### 🔍 Key Configuration Differences Summary

#### 1️⃣ **MoE Expert Count** (Most Critical Difference)
- **Intern-S1-Pro**: 512 experts + 8 router groups
- **Qwen3-VL**: 128 experts, no grouping
- **Impact**: Directly causes ~4x parameter difference

#### 2️⃣ **Position Encoding Strategy**
- **Intern-S1-Pro**: FoPE (Fourier Position Encoding)
  - Optimized for scientific signals and time series
  - `fope_init_factor=0.5`, `fope_sep_head=true`
- **Qwen3-VL**: Interleaved MRoPE
  - Multi-dimensional interleaved encoding (time, width, height)
  - `mrope_section=[24, 20, 20]` three-dimensional allocation

#### 3️⃣ **Vision Encoder**
- **Intern-S1-Pro**: 24 layers, 1024-dim, more lightweight
- **Qwen3-VL**: 27 layers, 1152-dim, supports DeepStack multi-layer feature fusion

#### 4️⃣ **Quantization Support**
- **Intern-S1-Pro**: Built-in FP8 quantization (e4m3 format)
  - 698 critical modules maintain high precision
- **Qwen3-VL**: No built-in quantization config

#### 5️⃣ **Vocabulary Size**
- **Intern-S1-Pro**: 155,008 tokens
  - Includes specialized domain tokenizers (PROT, SMILES, XNA)
- **Qwen3-VL**: 151,936 tokens

## 📊 Detailed Parameter Analysis

### Intern-S1-Pro (~1T Parameters)

```
Total Parameters: 920B
├─ Text Model:      916.56B (99.97%)
│  ├─ Embedding:    0.63B
│  ├─ 94 Layers:    915.29B
│  │  ├─ Attention per layer:    0.071B
│  │  ├─ MoE Experts (512):      9.66B  ⬅️ Key Difference
│  │  └─ Router per layer:       2.10M
│  └─ LM Head:      0.63B
└─ Vision Model:    0.31B (0.03%)
   ├─ Depth: 24 layers
   ├─ Hidden Size: 1024
   └─ Intermediate Size: 4096

Activated Parameters: 22.36B (only 2.4%)
```

### Qwen3-VL-235B-A22B (~235B Parameters)

```
Total Parameters: 235.51B
├─ Text Model:      235.09B (99.82%)
│  ├─ Embedding:    0.62B
│  ├─ 94 Layers:    233.85B
│  │  ├─ Attention per layer:    0.071B
│  │  ├─ MoE Experts (128):      2.42B
│  │  └─ Router per layer:       0.52M
│  └─ LM Head:      0.62B
└─ Vision Model:    0.42B (0.18%)
   ├─ Depth: 27 layers
   ├─ Hidden Size: 1152
   └─ Intermediate Size: 4304

Activated Parameters: 22.19B (9.4%)
```

### 🔍 Why Does Intern-S1-Pro Have 1T Parameters?

**Key Reason: MoE Expert Count Difference**

- **Intern-S1-Pro**: 512 experts
- **Qwen3-VL**: 128 experts
- **Gap**: **4x**

**MoE Parameters per Layer**
- Intern-S1-Pro per layer: 9.66B (512 experts)
- Qwen3-VL per layer: 2.42B (128 experts)
- Accumulated over 94 layers: **681B vs 170B** difference

**But Similar Inference Efficiency**
- Both models activate only **8 experts**
- Activated parameters both ~**22B**
- Similar inference memory and compute cost

## ✨ Model Feature Comparison

### Intern-S1-Pro Features

#### 🔬 Scientific Reasoning Specialization
- **AI4Science Optimization**: Chemistry, materials, life sciences, earth sciences
- **Physical Signal Modeling**: Supports long time series (10^0 ~ 10^6 points)
- **Specialized Tokenizers**:
  - `tokenizer_PROT.model` - Protein sequences
  - `tokenizer_SMILES.model` - Chemical molecular formulas
  - `tokenizer_XNA.model` - Nucleic acid sequences

#### 🧠 Thinking Mode
- Thinking mode enabled by default for enhanced reasoning
- Can be disabled with `enable_thinking=False`

#### ⚙️ Technical Innovations
- **FoPE (Fourier Position Encoding)**: Better position encoding
- **STE Routing**: Dense gradient router training
- **Grouped Routing**: 8 groups of expert management (router_n_groups=8)
- **FP8 Quantization**: Reduced memory footprint

### Qwen3-VL Features

#### 🌐 General Visual Understanding
- **Visual Agent**: Operates PC/mobile GUI
- **Visual Coding Enhancement**: Generates Draw.io/HTML/CSS/JS
- **"Recognize Everything"**: Celebrities, anime, products, landmarks, flora/fauna

#### 📹 Advanced Video Understanding
- **Native 256K context**, expandable to 1M
- **Second-level video indexing**: Handles hours-long videos
- **Text-Timestamp Alignment**: Precise timestamp localization

#### 🏗️ Architecture Upgrades
- **Interleaved-MRoPE**: Multi-dimensional position encoding (time, width, height)
- **DeepStack**: Multi-layer ViT feature fusion (layers 8, 16, 24)
- **32 Language OCR**: Supports low-light, blur, tilt scenarios

## 📝 Calculation Methodology

### Parameter Calculation Formulas

#### 1️⃣ **Base Component Parameters (per layer)**

```python
# Attention parameters (GQA - Grouped Query Attention)
Attention = 4 × hidden_size² × (1 + num_kv_heads / num_q_heads)
          = 4 × 4096² × (1 + 4/64)
          = 71,303,168 ≈ 0.071B

# Router parameters (routing to experts)
Router = hidden_size × num_experts
       = 4096 × 512 (Intern-S1-Pro) or 4096 × 128 (Qwen3-VL)
       = 2.10M (Intern) or 0.52M (Qwen)

# LayerNorm parameters (2: pre-attn + post-attn)
LayerNorm = 2 × hidden_size
          = 2 × 4096
          = 8,192
```

#### 2️⃣ **MoE Expert Parameters (per layer)**

```python
# Single expert parameters (using SwiGLU)
Expert_params = 3 × hidden_size × moe_intermediate_size
              = 3 × 4096 × 1536
              = 18,874,368 ≈ 18.87M

# Total expert parameters (per layer)
MoE_total = num_experts × Expert_params
          = 512 × 18.87M (Intern) or 128 × 18.87M (Qwen)
          = 9.66B (Intern) or 2.42B (Qwen)

# Activated expert parameters (per layer, inference time)
MoE_activated = num_experts_per_tok × Expert_params
              = 8 × 18.87M
              = 150,994,944 ≈ 0.151B
```

#### 3️⃣ **Total Parameters (Complete Model)**

```python
# Embedding layer
Embedding = vocab_size × hidden_size
          = 155,008 × 4096 (Intern) or 151,936 × 4096 (Qwen)
          = 0.635B (Intern) or 0.622B (Qwen)

# All Transformer layers (including all experts)
All_layers_total = num_layers × (Attention + Router + LayerNorm + MoE_total)
                 = 94 × (0.071B + 2.10M + 8,192 + 9.66B)    [Intern]
                 = 94 × 9.737B
                 = 915.29B

# LM Head (output layer)
LM_head = hidden_size × vocab_size
        = 0.635B (Intern) or 0.622B (Qwen)

# 【Total Parameters】
Total = Embedding + All_layers_total + LM_head
      = 0.635B + 915.29B + 0.635B
      = 916.56B ≈ 0.92T (Intern-S1-Pro)

      or

      = 0.622B + 233.85B + 0.622B
      = 235.09B (Qwen3-VL)
```

#### 4️⃣ **Activated Parameters (Inference Time)**

```python
# Activated parameters for all layers (only partial experts)
All_layers_activated = num_layers × (Attention + Router + LayerNorm + MoE_activated)
                     = 94 × (0.071B + 2.10M + 8,192 + 0.151B)
                     = 94 × 0.224B
                     = 21.09B

# 【Inference-time Activated Parameters】
Activated_total = Embedding + All_layers_activated + LM_head
                = 0.635B + 21.09B + 0.635B
                = 22.36B (Intern-S1-Pro)

                or

                = 0.622B + 20.94B + 0.622B
                = 22.19B (Qwen3-VL)
```

### 📊 Calculation Results Comparison

| Metric | Intern-S1-Pro | Qwen3-VL | Notes |
|--------|--------------|----------|-------|
| **Total Parameters** | 0.92T | 235.09B | Intern is 3.9x larger |
| **Activated Parameters** | 22.36B | 22.19B | Nearly identical (< 1% diff) |
| **Activation Rate** | 2.44% | 9.44% | Intern has more experts, lower rate |
| **Expert Count** | 512 | 128 | 4x difference |
| **Activated per Token** | 8 experts | 8 experts | Same |

**Key Insights:**
- Total expert count → Determines total parameters (920B vs 235B) → Affects **deployment cost**
- Activated expert count → Determines activated parameters (22.36B vs 22.19B) → Affects **inference cost**
- Intern-S1-Pro: 4x deployment cost for stronger scientific capabilities

## 📚 References

### Intern-S1-Pro
- 🤗 [Hugging Face](https://huggingface.co/internlm/Intern-S1-Pro)
- 📄 [Technical Report](https://arxiv.org/abs/2508.15763)
- 🏠 [Official Repository](https://github.com/InternLM/Intern-S1)

### Qwen3-VL
- 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
- 📄 [Technical Report (Qwen3)](https://arxiv.org/abs/2505.09388)
- 📄 [Technical Report (Qwen3-VL)](https://arxiv.org/abs/2511.21631)

## 📄 License

- Intern-S1-Pro: Apache 2.0
- Qwen3-VL: Apache 2.0
