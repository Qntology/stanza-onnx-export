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

    pub fn apply_tiered_masking(&self, text: &str, stanza: &[StanzaResult], lang: &str) -> String {
        let mut masked_text = text.to_string();
        let mut entities_to_mask = Vec::new();

        // 1. Tier: Stanza-based Entity Merging (Matches Python refine_entities_with_hint)
        let mut i = 0;
        while i < stanza.len() {
            let res = &stanza[i];
            
            // Check for potential PII start (Linguistic context)
            let is_candidate = res.upos == "PROPN" 
                             || res.upos == "NUM"
                             || self.is_address_part(&res.text, lang);

            if is_candidate {
                let mut block_tokens = vec![res.text.clone()];
                let mut j = i + 1;
                while j < stanza.len() {
                    let next = &stanza[j];
                    // Python logic: combine if deprel in ['flat', 'compound', 'flat:name', 'nummod']
                    let is_linked = next.deprel == "flat" 
                                 || next.deprel == "compound" 
                                 || next.deprel == "flat:name" 
                                 || next.deprel == "nummod"
                                 || (next.upos == "PROPN" && res.upos == "PROPN") // Heuristic: consecutive names
                                 || self.is_address_part(&next.text, lang);
                    
                    if is_linked {
                        block_tokens.push(next.text.clone());
                        j += 1;
                    } else { break; }
                }
                
                let combined = block_tokens.join(" ");
                if combined.chars().count() >= 2 {
                    entities_to_mask.push(combined);
                }
                i = j;
                continue;
            }
            i += 1;
        }

        // 2. Tier: Regex
        for mat in EMAIL_REGEX.find_iter(text) { entities_to_mask.push(mat.as_str().to_string()); }
        for mat in PHONE_REGEX.find_iter(text) { entities_to_mask.push(mat.as_str().to_string()); }

        // Sort by length DESC and mask
        entities_to_mask.sort_by(|a, b| b.len().cmp(&a.len()));
        entities_to_mask.dedup();

        for ent in entities_to_mask {
            if ent.trim().is_empty() || ent.len() < 2 { continue; }
            let mask = "*".repeat(ent.chars().count());
            // Exact character count replacement to match Python's visual output
            masked_text = masked_text.replace(&ent, &mask);
        }

        masked_text
    }

    fn is_address_part(&self, text: &str, lang: &str) -> bool {
        if let Some(info) = self.address_dict.get(lang) {
            info.admin.iter().any(|a| text.contains(a)) || info.street.iter().any(|s| text.contains(s))
        } else { false }
    }
}
