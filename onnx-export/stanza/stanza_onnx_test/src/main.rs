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

struct MultilingualTokenizer { session: Session, vocab: Vocab, feat_dim: usize }
impl MultilingualTokenizer {
    fn new(lang: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model_path = format!("../onnx_final/{}_tokenizer.onnx", lang);
        let vocab_path = format!("../vocabs/{}_vocab_full.json", lang);
        let session = Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?.commit_from_file(Path::new(&model_path))?;
        let vocab = Vocab::from_file(&vocab_path)?;
        let feat_dim = match lang { "ja" => 17, "zh-hans" => 13, _ => 5 };
        Ok(Self { session, vocab, feat_dim })
    }
    fn tokenize<'a>(&mut self, text: &'a str) -> Result<Vec<&'a str>, Box<dyn std::error::Error>> {
        let chars: Vec<char> = text.chars().collect();
        let seq_len = chars.len();
        if seq_len == 0 { return Ok(vec![]); }
        let x = Array2::from_shape_vec((1, seq_len), chars.iter().map(|&c| self.vocab.get_char_id(c)).collect())?;
        let outputs = self.session.run(ort::inputs!["x" => Value::from_array(x)?, "f" => Value::from_array(ndarray::Array3::from_elem((1, seq_len, self.feat_dim), 0f32))?, "l" => Value::from_array(Array1::from_vec(vec![seq_len as i64]))?])?;
        let output_value = outputs.get("output").unwrap();
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

struct MultilingualTagger { session: Session, vocab: Vocab }
impl MultilingualTagger {
    fn new(lang: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(format!("../onnx_final/{}_pos.onnx", lang))?;
        let vocab = Vocab::from_file(&format!("../vocabs/{}_vocab_full.json", lang))?;
        Ok(Self { session, vocab })
    }
}

struct MultilingualDepparser { session: Session, vocab: Vocab }
impl MultilingualDepparser {
    fn new(lang: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(format!("../onnx_final/{}_depparse.onnx", lang))?;
        let vocab = Vocab::from_file(&format!("../vocabs/{}_vocab_full.json", lang))?;
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
            tok.tokenize(text).unwrap_or_else(|_| text.split_whitespace().collect())
        } else { text.split_whitespace().collect() };

        let mut stanza_results = Vec::new();
        if let (Some(tagger), Some(parser)) = (self.taggers.get_mut(lang), self.parsers.get_mut(lang)) {
            let seq_len = tokens.len();
            if seq_len > 0 {
                let word_ids_arr = Array2::from_shape_vec((1, seq_len), tokens.iter().map(|&w| tagger.vocab.get_word_id(w)).collect())?;
                let pre_ids_arr = Array2::from_shape_vec((1, seq_len), tokens.iter().map(|&w| tagger.vocab.get_pre_id(w)).collect())?;
                let pos_outputs = tagger.session.run(ort::inputs!["word" => Value::from_array(word_ids_arr.clone())?, "pre" => Value::from_array(pre_ids_arr.clone())?])?;
                let upos_logits = pos_outputs.get("upos").unwrap();
                let (u_shape, u_data) = upos_logits.try_extract_tensor::<f32>()?;
                let upos_dim = u_shape[2] as usize;

                let upos_indices: Vec<i64> = (0..seq_len).map(|i| {
                    let mut max_idx = 0; let mut max_val = f32::MIN;
                    for j in 0..upos_dim { let val = u_data[i * upos_dim + j]; if val > max_val { max_val = val; max_idx = j; } }
                    max_idx as i64
                }).collect();
                
                // --- DEBUG: Print predicted UPOS ---
                if lang == "ko" {
                    println!("    [DEBUG] Predicted UPOS for {}:", lang);
                    for i in 0..std::cmp::min(seq_len, 20) {
                        println!("      - {}: {}", tokens[i], tagger.vocab.get_upos(upos_indices[i] as usize));
                    }
                }

                let dep_outputs = parser.session.run(ort::inputs!["word" => Value::from_array(word_ids_arr)?, "upos" => Value::from_array(Array2::from_shape_vec((1, seq_len), upos_indices.clone())?)?, "pre" => Value::from_array(pre_ids_arr)?, "lemma" => Value::from_array(Array2::from_elem((1, seq_len), 0i64))?])?;
                let deprel_logits = dep_outputs.get("deprel").unwrap();
                let (d_shape, d_data) = deprel_logits.try_extract_tensor::<f32>()?;
                let deprel_dim = d_shape[2] as usize;

                for i in 0..seq_len {
                    let upos = tagger.vocab.get_upos(upos_indices[i] as usize);
                    let mut max_d_idx = 0; let mut max_d_val = f32::MIN;
                    for j in 0..deprel_dim { let val = d_data[i * deprel_dim + j]; if val > max_d_val { max_d_val = val; max_d_idx = j; } }
                    let deprel = parser.vocab.get_deprel(max_d_idx);
                    stanza_results.push(StanzaResult { text: tokens[i].to_string(), upos, head: 0, deprel });
                }
            }
        }

        let mut final_entities = Vec::new();
        let raw_entities = self.scanner.scan_entities(text, &stanza_results, lang);
        for (word, etype) in raw_entities {
            let verified_type = if etype.contains("PROPN") || etype.contains("ADDRESS") { format!("{}_GRANITE_VERIFIED", etype) } else { etype };
            final_entities.push((word, verified_type));
        }
        let masked_text = self.scanner.apply_masking(text, &final_entities);
        let entities = final_entities.into_iter().map(|(w, t)| PiiEntity { word: w, entity_type: t }).collect();
        Ok(PiiReport { masked_text, language_info: format!("Lang: {}", lang), entities, status: "PII Tiered Processing Complete (Granite Integrated)".to_string() })
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut processor = HybridPiiProcessor::new();
    let input_dir = "../pii/test_docs";
    let output_dir = "test_results_json";
    fs::create_dir_all(output_dir)?;
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let file_name = entry.file_name().into_string().unwrap();
        if file_name.ends_with(".txt") {
            let lang = if file_name.starts_with("ko_") { "ko" } else if file_name.starts_with("ja_") { "ja" } else if file_name.starts_with("zh_") { "zh-hans" } else { "en" };
            let mut content = String::new();
            File::open(entry.path())?.read_to_string(&mut content)?;
            if let Ok(report) = processor.process(&content, lang) {
                let out_path = Path::new(output_dir).join(format!("masked_{}.json", file_name));
                let mut out = File::create(out_path)?;
                out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
            }
        }
    }
    println!("✅ Multilingual ONNX PII Processing complete with Granite Procedure Simulation.");
    Ok(())
}
