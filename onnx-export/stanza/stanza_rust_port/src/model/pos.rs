use tch::{nn, Tensor};

pub struct Tagger {
    pub word_emb: nn::Embedding,
    pub pretrained_emb: nn::Embedding,
    pub taggerlstm: nn::LSTM,
    pub upos_clf: nn::Linear,
    pub xpos_clf: nn::Linear,
    pub ufeats_clf: Vec<nn::Linear>,
}

impl Tagger {
    pub fn new(vs: &nn::Path, vocab_size: i64, pretrained_size: i64, hidden_dim: i64, num_feats: usize) -> Self {
        let config = nn::lstm::LSTMConfig {
            ..Default::default()
        };

        Self {
            word_emb: nn::embedding(vs / "word_emb", vocab_size, 75, Default::default()),
            pretrained_emb: nn::embedding(vs / "pretrained_emb", pretrained_size, 100, Default::default()),
            taggerlstm: nn::lstm(vs / "taggerlstm", 175, hidden_dim, config),
            upos_clf: nn::linear(vs / "upos_clf", hidden_dim * 2, 50, Default::default()),
            xpos_clf: nn::linear(vs / "xpos_clf", hidden_dim * 2, 50, Default::default()),
            ufeats_clf: (0..num_feats)
                .map(|i| nn::linear(vs / format!("ufeats_clf_{}", i), hidden_dim * 2, 10, Default::default()))
                .collect(),
        }
    }
}
