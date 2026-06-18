use crate::model::GraniteModel;

pub struct PiiVerifier<'a> {
    model: &'a mut GraniteModel,
}

impl<'a> PiiVerifier<'a> {
    pub fn new(model: &'a mut GraniteModel) -> Self {
        Self { model }
    }

    pub fn verify(&mut self, word: &str, context: &str) -> bool {
        let prompt = format!(
            "<|start_of_role|>system<|end_of_role|>Task: PII Verification\n\
            Context: {}\n\
            Question: Is '{}' a private personal information (like Name, Address, etc.)?\n\
            Answer ONLY 'YES' or 'NO'.\n\
            <|start_of_role|>user<|end_of_role|>Word: {}\n\
            <|start_of_role|>assistant<|end_of_role|>Answer:",
            context, word, word
        );

        println!("    [Log] LLM Verifier called. Validating '{}'...", word);
        
        match self.model.generate_text(&prompt, 5) {
            Ok(response) => {
                let answer = response.trim().to_uppercase();
                println!("    [Log] Granite LLM Answered: {}", answer);
                return answer.contains("YES");
            },
            Err(e) => {
                println!("    [Error] LLM Verification failed: {}. Defaulting to true.", e);
            }
        }
        true
    }

    pub fn classify(&mut self, word: &str, context: &str) -> String {
        let prompt = format!(
            "<|start_of_role|>system<|end_of_role|>Task: PII Classification\n\
            Context: {}\n\
            Labels: PERSON, ORGANIZATION, LOCATION, ADDRESS, OTHER\n\
            Classify '{}'.\n\
            <|start_of_role|>user<|end_of_role|>Result:",
            context, word
        );

        match self.model.generate_text(&prompt, 10) {
            Ok(response) => {
                let answer = response.trim().to_uppercase();
                println!("    [Log] Granite LLM Classified as: {}", answer);
                return answer;
            },
            Err(_) => return "PERSON".to_string(),
        }
    }
}
