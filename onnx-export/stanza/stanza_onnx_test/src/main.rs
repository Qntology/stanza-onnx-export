use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::{Array2, Array1};
use std::path::Path;
use std::fs::{self, File};
use std::io::{Write, Read};
use serde::{Serialize, Deserialize};

mod config;
mod model;
mod generate;
mod pii;
mod vocab;
mod llm_verifier;
mod pii_scanner;

use vocab::Vocab;
use pii_scanner::{PiiScanner, StanzaResult};

#[derive(Serialize)]
struct PiiReport {
    masked_text: String,
    language_info: String,
    entities: Vec<PiiEntity>,
    status: String,
}

#[derive(Serialize, Clone, Debug)]
struct PiiEntity {
    word: String,
    #[serde(rename = "type")]
    entity_type: String,
}

struct HybridPiiProcessor {
    tokenizers: std::collections::HashMap<String, MultilingualTokenizer>,
    taggers: std::collections::HashMap<String, MultilingualTagger>,
    parsers: std::collections::HashMap<String, MultilingualDepparser>,
    scanner: PiiScanner,
}

struct MultilingualTokenizer {
    session: Session,
    vocab: Vocab,
    feat_dim: usize,
}

impl MultilingualTokenizer {
    fn new(lang: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = format!("../onnx_final/{}_tokenizer.onnx", lang);
        let vocab_path = format!("../vocabs/{}_vocab_full.json", lang);
        if !Path::new(&model_path).exists() { return Err(format!("Model not found: {}", model_path).into()); }
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(Path::new(&model_path))?;
        let vocab = Vocab::from_file(&vocab_path)?;
        
        // Manual Map for feat_dim based on previous run errors
        let feat_dim = match lang {
            "ja" => 17,
            "zh-hans" => 13,
            _ => 5,
        };
        
        Ok(Self { session, vocab, feat_dim })
    }

    fn tokenize<'a>(&mut self, text: &'a str) -> Result<Vec<&'a str>, Box<dyn std::error::Error>> {
        let chars: Vec<char> = text.chars().collect();
        let seq_len = chars.len();
        if seq_len == 0 { return Ok(vec![]); }
        let x_vec: Vec<i64> = chars.iter().map(|&c| self.vocab.get_char_id(c)).collect();
        let x = Array2::from_shape_vec((1, seq_len), x_vec)?;
        
        let f = ndarray::Array3::from_elem((1, seq_len, self.feat_dim), 0f32);
        let l = Array1::from_vec(vec![seq_len as i64]);

        let outputs = self.session.run(ort::inputs![
            "x" => Value::from_array(x)?,
            "f" => Value::from_array(f)?,
            "l" => Value::from_array(l)?,
        ])?;
        
        let output_value = outputs.get("output").ok_or("Output not found")?;
        let (_shape, data) = output_value.try_extract_tensor::<f32>()?;
        
        let mut tokens = Vec::new();
        let mut current_start = 0;
        for i in 0..seq_len {
            if data[i] > 0.5 || i == seq_len - 1 {
                let byte_start = text.char_indices().nth(current_start).map(|(idx, _)| idx).unwrap_or(text.len());
                let byte_end = text.char_indices().nth(i).map(|(idx, c)| idx + c.len_utf8()).unwrap_or(text.len());
                tokens.push(&text[byte_start..byte_end]);
                current_start = i + 1;
            }
        }
        Ok(tokens)
    }
}

struct MultilingualTagger {
    session: Session,
    vocab: Vocab,
}

impl MultilingualTagger {
    fn new(lang: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = format!("../onnx_final/{}_pos.onnx", lang);
        let vocab_path = format!("../vocabs/{}_vocab_full.json", lang);
        let session = Session::builder()?.commit_from_file(Path::new(&model_path))?;
        let vocab = Vocab::from_file(&vocab_path)?;
        Ok(Self { session, vocab })
    }
}

struct MultilingualDepparser {
    session: Session,
    vocab: Vocab,
}

impl MultilingualDepparser {
    fn new(lang: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = format!("../onnx_final/{}_depparse.onnx", lang);
        let vocab_path = format!("../vocabs/{}_vocab_full.json", lang);
        let session = Session::builder()?.commit_from_file(Path::new(&model_path))?;
        let vocab = Vocab::from_file(&vocab_path)?;
        Ok(Self { session, vocab })
    }
}

impl HybridPiiProcessor {
    fn new() -> Self {
        let mut tokenizers = std::collections::HashMap::new();
        let mut taggers = std::collections::HashMap::new();
        let mut parsers = std::collections::HashMap::new();
        for lang in vec!["ko", "en", "ja", "zh-hans"] {
            if let Ok(tok) = MultilingualTokenizer::new(lang) { tokenizers.insert(lang.to_string(), tok); }
            if let Ok(tag) = MultilingualTagger::new(lang) { taggers.insert(lang.to_string(), tag); }
            if let Ok(par) = MultilingualDepparser::new(lang) { parsers.insert(lang.to_string(), par); }
        }
        Self { tokenizers, taggers, parsers, scanner: PiiScanner::new() }
    }

    fn process(&mut self, text: &str, lang: &str) -> Result<PiiReport, Box<dyn std::error::Error>> {
        println!("  [Hybrid] Processing: {}", lang);
        let tokens: Vec<&str> = if let Some(tok) = self.tokenizers.get_mut(lang) {
            match tok.tokenize(text) {
                Ok(t) => t,
                Err(e) => {
                    println!("    ⚠️ Tokenizer error for {}: {}", lang, e);
                    text.split_whitespace().collect()
                }
            }
        } else {
            text.split_whitespace().collect()
        };

        let mut stanza_results = Vec::new();
        if let (Some(tagger), Some(parser)) = (self.taggers.get_mut(lang), self.parsers.get_mut(lang)) {
            let seq_len = tokens.len();
            if seq_len > 0 {
                let word_ids: Vec<i64> = tokens.iter().map(|&w| tagger.vocab.get_word_id(w)).collect();
                let word_ids_arr = Array2::from_shape_vec((1, seq_len), word_ids)?;
                let pre_ids: Vec<i64> = tokens.iter().map(|&w| tagger.vocab.get_pre_id(w)).collect();
                let pre_ids_arr = Array2::from_shape_vec((1, seq_len), pre_ids)?;

                let pos_outputs = tagger.session.run(ort::inputs![
                    "word" => Value::from_array(word_ids_arr.clone())?,
                    "pre" => Value::from_array(pre_ids_arr.clone())?,
                ])?;

                let upos_logits = pos_outputs.get("upos").ok_or("upos not found")?;
                let (u_shape, u_data) = upos_logits.try_extract_tensor::<f32>()?;
                let upos_dim = u_shape[2] as usize;

                let upos_indices: Vec<i64> = (0..seq_len).map(|i| {
                    let mut max_idx = 0;
                    let mut max_val = f32::MIN;
                    for j in 0..upos_dim {
                        let val = u_data[i * upos_dim + j];
                        if val > max_val { max_val = val; max_idx = j; }
                    }
                    max_idx as i64
                }).collect();
                let upos_arr = Array2::from_shape_vec((1, seq_len), upos_indices)?;
                // FIX: Use zeros for lemma to avoid out-of-bounds in arc embedding if it expects specific lemma IDs
                let lemma_arr = Array2::from_elem((1, seq_len), 0i64);

                let dep_outputs = parser.session.run(ort::inputs![
                    "word" => Value::from_array(word_ids_arr)?,
                    "upos" => Value::from_array(upos_arr)?,
                    "pre" => Value::from_array(pre_ids_arr)?,
                    "lemma" => Value::from_array(lemma_arr)?,
                ])?;

                let deprel_logits = dep_outputs.get("deprel").ok_or("deprel not found")?;
                let (d_shape, d_data) = deprel_logits.try_extract_tensor::<f32>()?;
                let deprel_dim = d_shape[2] as usize;

                for i in 0..seq_len {
                    let mut max_u_idx = 0;
                    let mut max_u_val = f32::MIN;
                    for j in 0..upos_dim {
                        let val = u_data[i * upos_dim + j];
                        if val > max_u_val { max_u_val = val; max_u_idx = j; }
                    }
                    let upos = tagger.vocab.get_upos(max_u_idx);

                    let mut max_d_idx = 0;
                    let mut max_d_val = f32::MIN;
                    for j in 0..deprel_dim {
                        let val = d_data[i * deprel_dim + j];
                        if val > max_d_val { max_d_val = val; max_d_idx = j; }
                    }
                    let deprel = parser.vocab.get_deprel(max_d_idx);

                    stanza_results.push(StanzaResult {
                        text: tokens[i].to_string(),
                        upos,
                        head: 0,
                        deprel,
                    });
                }
            }
        }

        let masked_text = self.scanner.apply_tiered_masking(text, &stanza_results, lang);
        
        let mut entities = Vec::new();
        if masked_text != text {
            entities.push(PiiEntity {
                word: "Detected PII Block".to_string(),
                entity_type: format!("{}_PII", lang.to_uppercase()),
            });
        }

        Ok(PiiReport {
            masked_text,
            language_info: format!("Lang: {}", lang),
            entities,
            status: "PII Tiered Processing Complete".to_string(),
        })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Initializing Hybrid PII Processor...");
    let mut processor = HybridPiiProcessor::new();
    let input_dir = "../pii/test_docs";
    let output_dir = "test_results_json";
    fs::create_dir_all(output_dir)?;

    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let file_name = entry.file_name().into_string().unwrap();
        if file_name.ends_with(".txt") {
            let lang = if file_name.starts_with("ko_") { "ko" } 
                       else if file_name.starts_with("ja_") { "ja" }
                       else if file_name.starts_with("zh_") { "zh-hans" }
                       else { "en" };
            
            println!("📄 Reading: {}", file_name);
            let mut content = String::new();
            File::open(entry.path())?.read_to_string(&mut content)?;
            
            match processor.process(&content, lang) {
                Ok(report) => {
                    let out_path = Path::new(output_dir).join(format!("masked_{}.json", file_name));
                    println!("  💾 Saving result to: {:?}", out_path);
                    let mut out = File::create(out_path)?;
                    let json = serde_json::to_string_pretty(&report)?;
                    out.write_all(json.as_bytes())?;
                },
                Err(e) => {
                    println!("  ❌ Error processing {}: {}", file_name, e);
                }
            }
        }
    }
    println!("✅ Multilingual ONNX PII Processing complete.");
    Ok(())
}
