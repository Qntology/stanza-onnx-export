use ndarray::{Array2, Axis};
use crate::model::GraniteModel;

/* [기록용] candle_core 텐서 연산을 활용한 텍스트 제너레이션 로직
use candle_core::Tensor;

pub fn generate_candle(model: &mut GraniteModel, input_ids: Vec<i64>, max_new_tokens: usize) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    let mut current_ids = input_ids.clone();
    let device = model.device.clone();

    for step in 0..max_new_tokens {
        let seq_len = current_ids.len();
        let input_u32: Vec<u32> = current_ids.iter().map(|&x| x as u32).collect();
        let input_tensor = Tensor::from_vec(input_u32, (1, seq_len), &device)?;
        let logits = model.forward(input_tensor)?;
        
        let last_token_logits = logits.i((0, seq_len - 1, ..))?;
        let logits_vec = last_token_logits.to_vec1::<f32>()?;
        
        let mut max_idx = 0;
        let mut max_val = f32::MIN;
        for (i, &val) in logits_vec.iter().enumerate() {
            if val > max_val { max_val = val; max_idx = i; }
        }
        current_ids.push(max_idx as i64);
        if max_idx == 2 { break; }
    }
    Ok(current_ids)
}
*/

// 현재 빌드를 위한 Mock Generator 구조체 (ndarray 기반)
pub struct Generator<'a> {
    model: &'a mut GraniteModel,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a mut GraniteModel) -> Self {
        Self { model }
    }

    pub fn generate(&mut self, input_ids: Vec<i64>, max_new_tokens: usize) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
        let mut current_ids = input_ids.clone();
        
        for step in 0..max_new_tokens {
            let batch_size = 1;
            let seq_len = current_ids.len();
            let input_array = Array2::from_shape_vec((batch_size, seq_len), current_ids.clone())?;
            
            let logits = self.model.forward(input_array)?;
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
            if max_idx == 2 {
                break;
            }
        }
        
        Ok(current_ids)
    }
}
