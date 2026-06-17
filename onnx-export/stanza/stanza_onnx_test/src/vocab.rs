use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Vocab {
    #[serde(default)]
    pub word2id: HashMap<String, i64>,
    #[serde(default)]
    pub upos: Vec<String>,
    #[serde(default)]
    pub deprel: Vec<String>,
    #[serde(default)]
    pub pre2id: Option<HashMap<String, i64>>,
    #[serde(default)]
    pub char2id: HashMap<String, i64>,
}

impl Vocab {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let vocab: Vocab = serde_json::from_reader(reader)?;
        Ok(vocab)
    }

    pub fn get_word_id(&self, word: &str) -> i64 {
        if let Some(&id) = self.word2id.get(word) {
            return id;
        }
        let lower = word.to_lowercase();
        if let Some(&id) = self.word2id.get(&lower) {
            return id;
        }
        1 // <UNK>
    }

    pub fn get_char_id(&self, c: char) -> i64 {
        if let Some(&id) = self.char2id.get(&c.to_string()) {
            return id;
        }
        1 // <UNK>
    }

    pub fn get_pre_id(&self, word: &str) -> i64 {
        if let Some(ref pre2id) = self.pre2id {
            if let Some(&id) = pre2id.get(word) {
                return id;
            }
            let lower = word.to_lowercase();
            if let Some(&id) = pre2id.get(&lower) {
                return id;
            }
        }
        1 // <UNK>
    }

    pub fn get_upos(&self, id: usize) -> String {
        self.upos.get(id).cloned().unwrap_or_else(|| "X".to_string())
    }

    pub fn get_deprel(&self, id: usize) -> String {
        self.deprel.get(id).cloned().unwrap_or_else(|| "root".to_string())
    }
}
