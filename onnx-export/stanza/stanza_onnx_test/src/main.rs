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
        let (shape, data) = output_value.try_extract_tensor::<f32>()?;
        
        let mut tokens = Vec::new();
        let mut current_start = 0;
        let channels = if shape.len() == 3 { shape[2] as usize } else { 1 };

        for i in 0..seq_len {
            let mut max_idx = 0;
            let mut max_val = f32::MIN;
            if channels >= 2 {
                for j in 0..channels {
                    let val = data[i * channels + j];
                    if val > max_val { max_val = val; max_idx = j; }
                }
            }
            
            // Channel 1 or 2 often represent boundaries (Space or No-Space boundary)
            let is_boundary = if channels >= 3 {
                max_idx == 1 || max_idx == 2
            } else if channels == 2 {
                max_idx == 1
            } else {
                data[i] > 0.0
            };

            if is_boundary || i == seq_len - 1 {
                let byte_start = text.char_indices().nth(current_start).map(|(idx, _)| idx).unwrap_or(text.len());
                let byte_end = text.char_indices().nth(i).map(|(idx, c)| idx + c.len_utf8()).unwrap_or(text.len());
                if byte_end > byte_start {
                    tokens.push(&text[byte_start..byte_end]);
                }
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
            match MultilingualTokenizer::new(lang) {
                Ok(tok) => { tokenizers.insert(lang.to_string(), tok); },
                Err(e) => println!("    [Warning] Tokenizer for {} failed to load: {}", lang, e),
            }
            match MultilingualTagger::new(lang) {
                Ok(tag) => { taggers.insert(lang.to_string(), tag); },
                Err(e) => println!("    [Warning] Tagger for {} failed to load: {}", lang, e),
            }
            match MultilingualDepparser::new(lang) {
                Ok(par) => { parsers.insert(lang.to_string(), par); },
                Err(e) => println!("    [Warning] Parser for {} failed to load: {}", lang, e),
            }
        }
        Self { tokenizers, taggers, parsers, scanner: PiiScanner::new() }
    }

    fn process(&mut self, text: &str, lang: &str, granite_model: Option<&mut crate::model::GraniteModel>) -> Result<PiiReport, Box<dyn std::error::Error>> {
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
                let dep_outputs = parser.session.run(ort::inputs!["word" => Value::from_array(word_ids_arr)?, "upos" => Value::from_array(Array2::from_shape_vec((1, seq_len), upos_indices.clone())?)?, "pre" => Value::from_array(pre_ids_arr)?, "lemma" => Value::from_array(Array2::from_elem((1, seq_len), 0i64))?])?;
                let deprel_logits = dep_outputs.get("deprel").unwrap();
                let (d_shape, d_data) = deprel_logits.try_extract_tensor::<f32>()?;
                let deprel_dim = d_shape[2] as usize;

                println!("    [Log] === Stanza Model Output ===");
                for i in 0..seq_len {
                    let upos = tagger.vocab.get_upos(upos_indices[i] as usize);
                    let mut max_d_idx = 0; let mut max_d_val = f32::MIN;
                    for j in 0..deprel_dim { let val = d_data[i * deprel_dim + j]; if val > max_d_val { max_d_val = val; max_d_idx = j; } }
                    let deprel = parser.vocab.get_deprel(max_d_idx);
                    
                    println!("    [Log] Token: '{}' | UPOS: {} (Index: {}) | DEPREL: {} (Max Val: {})", tokens[i], upos, upos_indices[i], deprel, max_d_val);
                    stanza_results.push(StanzaResult { text: tokens[i].to_string(), upos, head: 0, deprel });
                }
                println!("    [Log] ===========================");
            }
        }

        let mut final_entities = Vec::new();
        let raw_entities = self.scanner.scan_entities(text, &stanza_results, lang);
        
        if let Some(model) = granite_model {
            let mut verifier = crate::llm_verifier::PiiVerifier::new(model);
            for (word, etype) in raw_entities {
                let context_snippet = text.chars().take(250).collect::<String>();
                if verifier.verify(&word, &context_snippet) {
                    let class = verifier.classify(&word, &context_snippet);
                    final_entities.push((word, format!("{}_GRANITE_{}", lang.to_uppercase(), class)));
                } else {
                    println!("    [Log] Granite rejected '{}' as PII.", word);
                }
            }
        } else {
            for (word, etype) in raw_entities {
                println!("    [Log] Fake Granite Verification bypass applied for: '{}' [{}]", word, etype);
                final_entities.push((word, etype));
            }
        }
        
        let masked_text = self.scanner.apply_masking(text, &final_entities);
        let entities = final_entities.into_iter().map(|(w, t)| PiiEntity { word: w, entity_type: t }).collect();
        Ok(PiiReport { masked_text, language_info: format!("Lang: {}", lang), entities, status: "PII Tiered Processing Complete (Granite Integrated)".to_string() })
    }
}

fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        println!("  [Log] File already exists: {:?}", path);
        return Ok(());
    }
    println!("  [Log] Downloading {} to {:?}...", url, path);
    let response = reqwest::blocking::get(url)?;
    let mut dest = File::create(path)?;
    let content = response.bytes()?;
    std::io::copy(&mut content.as_ref(), &mut dest)?;
    println!("  [Log] Download complete: {:?}", path);
    Ok(())
}

fn prepare_granite_gguf() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let appdata = std::env::var("APPDATA").expect("APPDATA environment variable not found");
    let xtanza_dir = Path::new(&appdata).join("xtanza");
    fs::create_dir_all(&xtanza_dir)?;
    
    let model_path = xtanza_dir.join("granite-4.0-h-350m-Q8_0.gguf");
    download_file("https://huggingface.co/ibm-granite/granite-4.0-h-350m-GGUF/resolve/main/granite-4.0-h-350m-Q8_0.gguf", &model_path)?;
    
    let meta_files = vec![
        ("config.json", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/config.json"),
        ("tokenizer.json", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/tokenizer.json"),
        ("tokenizer_config.json", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/tokenizer_config.json"),
        ("vocab.json", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/vocab.json"),
        ("chat_template.jinja", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/chat_template.jinja"),
        ("generation_config.json", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/generation_config.json"),
        ("model.sig", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/model.sig"),
        ("special_tokens_map.json", "https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/special_tokens_map.json"),
    ];

    for (file_name, url) in meta_files {
        let path = xtanza_dir.join(file_name);
        download_file(url, &path).unwrap_or_else(|e| println!("Failed to download {}: {}", file_name, e));
    }
    
    Ok(model_path)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("  [Log] Checking and preparing Granite GGUF model in APPDATA/xtanza...");
    let gguf_model_path = prepare_granite_gguf()?;
    println!("  [Log] GGUF Model ready at: {:?}", gguf_model_path);
    
    let mut granite_model = crate::model::GraniteModel::from_file(&gguf_model_path, crate::config::Config::default())
        .expect("Failed to initialize Granite Model via llama-cpp-rs");

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
            
            if let Ok(report) = processor.process(&content, lang, Some(&mut granite_model)) {
                let out_path = Path::new(output_dir).join(format!("masked_{}.json", file_name));
                let mut out = File::create(out_path)?;
                out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
            }
        }
    }
    println!("✅ Multilingual ONNX & Granite GGUF PII Processing complete.");
    Ok(())
}
