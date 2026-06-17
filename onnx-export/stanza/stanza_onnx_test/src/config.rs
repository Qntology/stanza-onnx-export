use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteLayerType {
    #[default]
    Attention,
    Mamba,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GraniteRopeType {
    #[default]
    Default,
    Granite,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GraniteRopeConfig {
    pub factor: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_type: GraniteRopeType,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<GraniteRopeConfig>,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub layer_types: Vec<GraniteLayerType>,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
    // Mamba configuration
    pub mamba_n_heads: Option<usize>,
    pub mamba_n_groups: usize,
    pub mamba_d_state: usize,
    pub mamba_d_head: Option<usize>,
    pub mamba_d_conv: usize,
    pub mamba_expand: usize,
    pub mamba_chunk_size: usize,
    pub mamba_conv_bias: bool,
    pub mamba_proj_bias: bool,
    // MoE configuration
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub position_embedding_type: String,
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn shared_intermediate_size(&self) -> usize {
        self.shared_intermediate_size
            .unwrap_or(self.intermediate_size)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn layer_types(&self) -> Vec<GraniteLayerType> {
        if self.layer_types.is_empty() {
            vec![GraniteLayerType::Attention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        }
    }

    pub fn mamba_intermediate_size(&self) -> usize {
        self.mamba_expand * self.hidden_size
    }

    pub fn mamba_n_heads(&self) -> usize {
        self.mamba_n_heads.unwrap_or(128)
    }

    pub fn mamba_d_head(&self) -> usize {
        self.mamba_d_head
            .unwrap_or(self.mamba_intermediate_size() / self.mamba_n_heads())
    }

    pub fn mamba_conv_dim(&self) -> usize {
        self.mamba_intermediate_size() + 2 * self.mamba_n_groups * self.mamba_d_state
    }
}
