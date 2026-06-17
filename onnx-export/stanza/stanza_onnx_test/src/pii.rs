use regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref EMAIL_REGEX: Regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
    pub static ref PHONE_KO: Regex = Regex::new(r"(\+82|0)[1-9]\d{0,2}[-.\s]?\d{3,4}[-.\s]?\d{4}").unwrap();
    pub static ref PHONE_EN: Regex = Regex::new(r"(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|(\+44[-.\s]?)?\(?0?\d{2,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}").unwrap();
    pub static ref PHONE_JA: Regex = Regex::new(r"(\+81|0)\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{4}").unwrap();
    pub static ref PCC_REGEX: Regex = Regex::new(r"(?i)\bP\d{12}\b").unwrap();
}

pub enum PiiType {
    Email,
    Phone,
    Person,
    Location,
    Organization,
    Pcc, // Personal Custom Code
}

pub struct PiiMasker {
    mask_char: char,
}

impl PiiMasker {
    pub fn new(mask_char: char) -> Self {
        Self { mask_char }
    }

    pub fn mask_regex(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        // Mask Emails
        for mat in EMAIL_REGEX.find_iter(text) {
            let mask = self.mask_char.to_string().repeat(mat.as_str().len());
            result = result.replace(mat.as_str(), &mask);
        }

        // Mask Pcc
        for mat in PCC_REGEX.find_iter(text) {
            let mask = self.mask_char.to_string().repeat(mat.as_str().len());
            result = result.replace(mat.as_str(), &mask);
        }

        // Mask Phone (Global)
        for mat in PHONE_KO.find_iter(text) {
            let mask = self.mask_char.to_string().repeat(mat.as_str().len());
            result = result.replace(mat.as_str(), &mask);
        }
        for mat in PHONE_EN.find_iter(text) {
            let mask = self.mask_char.to_string().repeat(mat.as_str().len());
            result = result.replace(mat.as_str(), &mask);
        }
        for mat in PHONE_JA.find_iter(text) {
            let mask = self.mask_char.to_string().repeat(mat.as_str().len());
            result = result.replace(mat.as_str(), &mask);
        }

        result
    }

    pub fn mask_token(&self, token: &str, upos: &str) -> String {
        // Stanza UPOS tags that are candidates for PII
        // PROPN: Proper Noun (Names, Places)
        // X: Unknown/Foreign (Often names in mixed contexts)
        // NUM: Numbers (Can be parts of addresses or IDs)
        if upos == "PROPN" || upos == "X" {
            self.mask_char.to_string().repeat(token.len())
        } else {
            token.to_string()
        }
    }
}
