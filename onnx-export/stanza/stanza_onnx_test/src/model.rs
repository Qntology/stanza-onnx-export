use std::path::Path;
use crate::config::Config;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::llama_batch::LlamaBatch;

pub struct GraniteModel {
    pub config: Config,
    pub model: LlamaModel,
    pub context: LlamaContext<'static>,
    pub model_path: String,
}

impl GraniteModel {
    pub fn from_file<P: AsRef<Path>>(path: P, config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let path_str = path.as_ref().to_string_lossy().into_owned();
        
        let backend = llama_cpp_2::ggml_time_init();
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &path_str, &model_params)
            .map_err(|e| format!("GGUF Load Error: {}", e))?;
            
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(std::num::NonZeroU32::new(2048).unwrap()));
        let context = model.create_context(&backend, ctx_params)
            .map_err(|e| format!("Context Error: {}", e))?;
            
        Ok(Self { config, model, context, model_path: path_str })
    }

    pub fn generate_text(&mut self, prompt: &str, max_tokens: usize) -> Result<String, Box<dyn std::error::Error>> {
        let tokens_list = self.model.str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| format!("Tokenization Error: {}", e))?;
        
        self.context.clear_kv_cache();
        
        let mut batch = LlamaBatch::new(2048, 1);
        let last_index = tokens_list.len() - 1;
        for (i, &token) in tokens_list.iter().enumerate() {
            batch.add(token, i as i32, &[0], i == last_index)
                .map_err(|e| format!("Batch Add Error: {}", e))?;
        }
        
        self.context.decode(&mut batch)
            .map_err(|e| format!("Decode Error: {}", e))?;
        
        let mut output = String::new();
        
        for _ in 0..max_tokens {
            let candidates = self.context.candidates_ith(batch.n_tokens() - 1);
            let candidates_array = LlamaTokenDataArray::from_iter(candidates, false);
            let new_token_id = self.context.sample_token_greedy(candidates_array);
            
            if new_token_id == self.model.token_eos() {
                break;
            }
            
            let token_str = self.model.token_to_str(new_token_id)
                .unwrap_or_else(|_| "".to_string());
            output.push_str(&token_str);
            
            batch.clear();
            batch.add(new_token_id, batch.n_tokens() as i32, &[0], true)
                .map_err(|e| format!("Next Token Batch Error: {}", e))?;
            self.context.decode(&mut batch)
                .map_err(|e| format!("Next Token Decode Error: {}", e))?;
        }
        
        Ok(output)
    }
}
