use ndarray::{Array2, Axis};
use crate::model::GraniteModel;

pub struct Generator<'a> {
    model: &'a mut GraniteModel,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a mut GraniteModel) -> Self {
        Self { model }
    }

    pub fn generate(&mut self, input_ids: Vec<i64>, max_new_tokens: usize) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
        let mut current_ids = input_ids.clone();
        
        for _ in 0..max_new_tokens {
            let batch_size = 1;
            let seq_len = current_ids.len();
            let input_array = Array2::from_shape_vec((batch_size, seq_len), current_ids.clone())?;
            
            let logits = self.model.forward(input_array)?;
            
            // Greedy sampling: get the last token's logits
            let last_token_logits = logits.index_axis(Axis(1), seq_len - 1);
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            
            for (i, &val) in last_token_logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            current_ids.push(max_idx as i64);
            
            // Stop if EOS token reached (assuming 2 is EOS for many models, should be config driven)
            if max_idx == 2 {
                break;
            }
        }
        
        Ok(current_ids)
    }
}
