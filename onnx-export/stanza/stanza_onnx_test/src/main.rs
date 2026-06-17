use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::{Array2, ArrayViewD, IxDyn};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Stanza ONNX Fully Dynamic High-Performance Inference ---");

    // 1. 110MB 완결 모델 로드
    let model_path = Path::new("../onnx_final/ko_pos.onnx");
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_path)?;

    println!("✅ Complete POS Model (110MB) loaded successfully!");

    // 2. 동적 입력 데이터 준비 (seq_len = 3)
    let batch_size = 1;
    let seq_len = 3;
    let max_char_len = 5;

    let word_ids = Array2::<i64>::from_elem((batch_size, seq_len), 1);
    let wordchars = Array2::<i64>::from_elem((batch_size * seq_len, max_char_len), 1);
    let pretrained = Array2::<i64>::from_elem((batch_size, seq_len), 1);

    let word_val = Value::from_array(word_ids)?;
    let wordchars_val = Value::from_array(wordchars)?;
    let pretrained_val = Value::from_array(pretrained)?;

    // 3. 추론 실행 (3개 입력 모두 전달)
    println!("Running dynamic inference (seq_len={})...", seq_len);
    let outputs = session.run(ort::inputs![
        "word" => word_val,
        "wordchars" => wordchars_val,
        "pretrained" => pretrained_val,
    ])?;

    // 4. 결과 출력
    if let Some(upos_logits) = outputs.get("upos_logits") {
        let (shape, data) = upos_logits.try_extract_tensor::<f32>()?;
        println!("✅ Inference success! Output shape: {:?}", shape);
        
        let dims: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let logits = ArrayViewD::from_shape(IxDyn(&dims), data)?;
        
        for s in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            for i in 0..dims[2] {
                let val = logits[[0, s, i]];
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            println!("Word[{}] -> Predicted UPOS Tag ID: {}", s, max_idx);
        }
    }

    println!("\n--- ALL TASKS COMPLETED SUCCESSFULLY ---");
    Ok(())
}
