use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref EMAIL_REGEX: Regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
    pub static ref PHONE_REGEX: Regex = Regex::new(r"\b(\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})\b").unwrap();
}

#[derive(Debug, Clone)]
pub struct StanzaResult {
    pub text: String,
    pub upos: String,
    pub head: usize,
    pub deprel: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddressInfo {
    pub admin: Vec<String>,
    pub street: Vec<String>,
}

pub struct PiiScanner {
    pub address_dict: HashMap<String, AddressInfo>,
}

impl PiiScanner {
    pub fn new() -> Self {
        let mut address_dict = HashMap::new();
        address_dict.insert("ko".to_string(), AddressInfo {
            admin: vec!["서울특별시".into(), "특별시".into(), "광역시".into(), "도".into(), "시".into(), "군".into(), "구".into(), "동".into(), "리".into(), "본사".into(), "대한민국".into()],
            street: vec!["대로".into(), "로".into(), "길".into(), "번길".into()],
        });
        address_dict.insert("en".to_string(), AddressInfo {
            admin: vec!["State".into(), "City".into(), "County".into(), "USA".into(), "UK".into()],
            street: vec!["Street".into(), "St".into(), "Road".into(), "Rd".into(), "Avenue".into(), "Ave".into(), "Blvd".into()],
        });
        Self { address_dict }
    }

    pub fn scan_entities(&self, text: &str, stanza: &[StanzaResult], lang: &str) -> Vec<(String, String)> {
        let mut entities = Vec::new();
        let tag_prefix = lang.to_uppercase();

        // 1. Regex
        for mat in EMAIL_REGEX.find_iter(text) {
            entities.push((mat.as_str().to_string(), format!("{}_EMAIL", tag_prefix)));
        }
        for mat in PHONE_REGEX.find_iter(text) {
            entities.push((mat.as_str().to_string(), format!("{}_PHONE", tag_prefix)));
        }

        // 2. Linguistic (Stanza)
        let mut i = 0;
        while i < stanza.len() {
            let res = &stanza[i];
            let is_cand = res.upos == "PROPN" || self.is_address_part(&res.text, lang);

            if is_cand {
                let mut combined = res.text.clone();
                let mut j = i + 1;
                while j < stanza.len() {
                    let next = &stanza[j];
                    // More conservative merging
                    let should_merge = (next.deprel == "flat" || next.deprel == "compound" || next.deprel == "flat:name")
                                    && (next.upos == "PROPN" || next.upos == "NOUN" || next.upos == "NUM" || self.is_address_part(&next.text, lang));
                    
                    if should_merge {
                        // Try with space first, then without
                        let with_space = format!("{} {}", combined, next.text);
                        let no_space = format!("{}{}", combined, next.text);
                        if text.contains(&with_space) { combined = with_space; }
                        else if text.contains(&no_space) { combined = no_space; }
                        else { break; }
                        j += 1;
                    } else { break; }
                }
                
                if combined.chars().count() >= 2 {
                    let ent_type = if self.is_address_part(&combined, lang) { "ADDRESS" } else { "PROPN" };
                    entities.push((combined, format!("{}_{}", tag_prefix, ent_type)));
                }
                i = j;
            } else {
                i += 1;
            }
        }
        
        entities.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        entities.dedup();
        entities
    }

    pub fn apply_masking(&self, text: &str, entities: &[(String, String)]) -> String {
        let mut masked = text.to_string();
        for (word, _) in entities {
            if word.trim().is_empty() || word.len() < 2 { continue; }
            let mask = "*".repeat(word.chars().count());
            masked = masked.replace(word, &mask);
        }
        masked
    }

    fn is_address_part(&self, text: &str, lang: &str) -> bool {
        if let Some(info) = self.address_dict.get(lang) {
            info.admin.iter().any(|a| text == a || text.ends_with(a)) || info.street.iter().any(|s| text == s || text.ends_with(s))
        } else { false }
    }
}
