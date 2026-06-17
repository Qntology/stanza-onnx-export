use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::{Array2, ArrayD, IxDyn};
use std::path::Path;
use crate::config::Config;

pub struct GraniteModel {
    pub config: Config,
    pub session: Session,
}

impl GraniteModel {
    pub fn from_file<P: AsRef<Path>>(path: P, config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;
        
        Ok(Self {
            config,
            session,
        })
    }

    pub fn forward(&mut self, input_ids: Array2<i64>) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
        let inputs = ort::inputs![
            "input_ids" => Value::from_array(input_ids)?,
        ];
        
        let outputs = self.session.run(inputs)?;
        let logits = outputs.get("logits").ok_or("Logits not found in output")?;
        let (shape, data) = logits.try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        
        Ok(ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec())?)
    }
}
