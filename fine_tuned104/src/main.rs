use anyhow::{Result, Context};
use csv::ReaderBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use tch::no_grad;
use rand::{thread_rng, seq::SliceRandom}; // <-- FIX: import rand utils

// ======================== DATA & CONFIG ========================

#[derive(Debug, Deserialize, Clone)]
struct SummarizationData {
    text: String,
    summary: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct IndoBERTConfig {
    vocab_size: i64,
    hidden_size: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    intermediate_size: i64,
    max_position_embeddings: i64,
    type_vocab_size: i64,
    hidden_dropout_prob: f64,
    attention_probs_dropout_prob: f64,
}

// ======================== ROUGE METRICS ========================

#[derive(Debug, Clone)]
struct RougeScores {
    rouge_1_f: f64,
    rouge_2_f: f64,
    rouge_l_f: f64,
}

fn tokenize_for_metrics(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn get_ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n { return vec![]; }
    tokens.windows(n).map(|w| w.to_vec()).collect()
}

fn calculate_rouge_n(reference: &[String], hypothesis: &[String], n: usize) -> f64 {
    let ref_ngrams = get_ngrams(reference, n);
    let hyp_ngrams = get_ngrams(hypothesis, n);
    if ref_ngrams.is_empty() || hyp_ngrams.is_empty() { return 0.0; }
    let ref_set: HashSet<_> = ref_ngrams.iter().collect();
    let hyp_set: HashSet<_> = hyp_ngrams.iter().collect();
    let overlap = ref_set.intersection(&hyp_set).count() as f64;
    let precision = overlap / hyp_ngrams.len() as f64;
    let recall = overlap / ref_ngrams.len() as f64;
    if precision + recall > 0.0 { 2.0 * (precision * recall) / (precision + recall) } else { 0.0 }
}

fn lcs_length(s1: &[String], s2: &[String]) -> usize {
    if s1.is_empty() || s2.is_empty() { return 0; }
    let (m, n) = (s1.len(), s2.len());
    let mut dp = vec![vec![0; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if s1[i - 1] == s2[j - 1] { dp[i][j] = dp[i - 1][j - 1] + 1; }
            else { dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]); }
        }
    }
    dp[m][n]
}

fn calculate_rouge_l(reference: &[String], hypothesis: &[String]) -> f64 {
    if reference.is_empty() || hypothesis.is_empty() { return 0.0; }
    let lcs = lcs_length(reference, hypothesis) as f64;
    let precision = lcs / hypothesis.len() as f64;
    let recall = lcs / reference.len() as f64;
    if precision + recall > 0.0 { 2.0 * (precision * recall) / (precision + recall) } else { 0.0 }
}

fn calculate_rouge(reference: &str, hypothesis: &str) -> RougeScores {
    let ref_tokens = tokenize_for_metrics(reference);
    let hyp_tokens = tokenize_for_metrics(hypothesis);
    RougeScores {
        rouge_1_f: calculate_rouge_n(&ref_tokens, &hyp_tokens, 1),
        rouge_2_f: calculate_rouge_n(&ref_tokens, &hyp_tokens, 2),
        rouge_l_f: calculate_rouge_l(&ref_tokens, &hyp_tokens),
    }
}

// ======================== IMPROVED TOKENIZER ========================

#[derive(Clone)]
struct IndoBERTTokenizer {
    vocab: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    max_length: i64,
    pad_token_id: i64,
    cls_token_id: i64,
    sep_token_id: i64,
    unk_token_id: i64,
    vocab_size: i64,
}

impl IndoBERTTokenizer {
    fn new(vocab: HashMap<String, i64>, max_length: i64, config_vocab_size: i64) -> Self {
        let pad_token_id = *vocab.get("[PAD]").unwrap_or(&0);
        let cls_token_id = *vocab.get("[CLS]").unwrap_or(&101);
        let sep_token_id = *vocab.get("[SEP]").unwrap_or(&102);
        let unk_token_id = *vocab.get("[UNK]").unwrap_or(&100);
        let mut id_to_token = HashMap::new();
        for (token, id) in vocab.iter() {
            id_to_token.insert(*id, token.clone());
        }
        IndoBERTTokenizer {
            vocab,
            id_to_token,
            max_length,
            pad_token_id,
            cls_token_id,
            sep_token_id,
            unk_token_id,
            vocab_size: config_vocab_size,
        }
    }

    fn wordpiece_tokenize(&self, word: &str) -> Vec<i64> {
        let w = word.to_lowercase();
        if w.is_empty() { return vec![]; }

        let chars: Vec<char> = w.chars().collect();
        let mut start = 0usize;
        let mut sub_tokens: Vec<i64> = Vec::new(); // <-- FIX: gunakan ini, tidak ada var tokens yang tak terpakai

        while start < chars.len() {
            let mut end = chars.len();
            let mut matched: Option<i64> = None;
            let mut matched_end = start;

            while end > start {
                let piece: String = chars[start..end].iter().collect();
                let candidate = if start == 0 { piece.clone() } else { format!("##{}", piece) };
                if let Some(&id) = self.vocab.get(&candidate) {
                    if id < self.vocab_size {
                        matched = Some(id);
                        matched_end = end;
                        break;
                    }
                }
                end -= 1;
            }

            if let Some(id) = matched {
                sub_tokens.push(id);
                start = matched_end;
            } else {
                // tidak ada subword yg cocok
                sub_tokens.push(self.unk_token_id);
                break;
            }
        }
        sub_tokens
    }

    fn tokenize_word(&self, word: &str) -> Vec<i64> {
        let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
        if clean_word.is_empty() { return vec![]; }

        if let Some(&id) = self.vocab.get(clean_word) {
            if id < self.vocab_size { return vec![id]; }
        }
        let lower = clean_word.to_lowercase();
        if let Some(&id) = self.vocab.get(&lower) {
            if id < self.vocab_size { return vec![id]; }
        }
        self.wordpiece_tokenize(clean_word)
    }

    fn encode(&self, text: &str) -> Vec<i64> {
        let mut tokens = vec![self.cls_token_id];
        for word in text.split_whitespace() {
            let mut word_tokens = self.tokenize_word(word);
            tokens.append(&mut word_tokens);
            if tokens.len() >= (self.max_length - 1) as usize { break; }
        }
        tokens.push(self.sep_token_id);
        while tokens.len() < self.max_length as usize { tokens.push(self.pad_token_id); }
        tokens.truncate(self.max_length as usize);
        tokens
    }

    fn decode(&self, ids: &[i64]) -> String {
        let mut out = Vec::new();
        let mut cur = String::new();

        for &id in ids {
            if id == self.pad_token_id || id == self.cls_token_id || id == self.sep_token_id { continue; }
            if let Some(tok) = self.id_to_token.get(&id) {
                if tok.starts_with("##") {
                    cur.push_str(&tok[2..]);
                } else {
                    if !cur.is_empty() { out.push(cur.clone()); cur.clear(); }
                    cur = tok.clone();
                }
            }
        }
        if !cur.is_empty() { out.push(cur); }
        out.join(" ")
    }

    fn vocab_size(&self) -> i64 { self.vocab_size }

    fn extend_vocab_with_dataset(&mut self, data: &[SummarizationData]) {
        println!("Extending vocabulary with dataset words... (WARNING: requires embedding resize if used)");
        let mut new_tokens = HashSet::new();
        let mut total_words = 0;
        let mut oov_words = 0;

        for item in data {
            for word in item.text.split_whitespace().chain(item.summary.split_whitespace()) {
                total_words += 1;
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                if clean.is_empty() { continue; }
                let tokens = self.tokenize_word(clean);
                if tokens.contains(&self.unk_token_id) {
                    oov_words += 1;
                    new_tokens.insert(clean.to_string());
                    new_tokens.insert(clean.to_lowercase());
                }
            }
        }

        println!(
            "Found {} OOV words out of {} total words ({:.1}%)",
            oov_words, total_words, (oov_words as f64 / total_words as f64) * 100.0
        );

        let start_id = *self.vocab.values().max().unwrap_or(&0) + 1;
        let new_tokens_count = new_tokens.len(); // <-- FIX: simpan jumlah sebelum consume

        for (idx, token) in new_tokens.into_iter().enumerate() {
            let new_id = start_id + idx as i64;
            self.vocab.insert(token.clone(), new_id);
            self.id_to_token.insert(new_id, token);
        }

        println!("Added {} new tokens to vocabulary", new_tokens_count);
        println!("NOTE: embeddings NOT resized. Do not call this unless you also resize model embeddings.");
    }
}

fn check_vocab_coverage(data: &[SummarizationData], tokenizer: &IndoBERTTokenizer) {
    let mut covered = 0;
    let mut total = 0;
    let mut missing_words = HashSet::new();

    for item in data {
        for word in item.text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.is_empty() { continue; }
            total += 1;
            let toks = tokenizer.tokenize_word(clean);
            if !toks.contains(&tokenizer.unk_token_id) && !toks.is_empty() { covered += 1; }
            else { missing_words.insert(clean.to_string()); }
        }
    }

    println!("Vocabulary Coverage: {}/{} ({:.1}%)", covered, total, (covered as f64 / total as f64) * 100.0);
    if !missing_words.is_empty() {
        println!("Sample missing words: {:?}", missing_words.iter().take(10).collect::<Vec<_>>());
    }
}

fn load_indobert_vocab(p: &str) -> Result<HashMap<String, i64>> {
    let file = File::open(p).context(format!("Failed to open vocab file: {}", p))?;
    let reader = BufReader::new(file);
    use std::io::BufRead;
    let mut vocab = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let token = line?;
        vocab.insert(token, idx as i64);
    }
    println!("✓ Loaded vocabulary: {} tokens", vocab.len());
    Ok(vocab)
}

fn load_indobert_config(p: &str) -> Result<IndoBERTConfig> {
    let file = File::open(p).context(format!("Failed to open config file: {}", p))?;
    let config: IndoBERTConfig = serde_json::from_reader(file).context("Failed to parse config JSON")?;
    println!("✓ Loaded IndoBERT config: vocab={}, hidden={}, heads={}",
             config.vocab_size, config.hidden_size, config.num_attention_heads);
    Ok(config)
}

// ======================== TRANSFORMER PARTS ========================

struct FeedForward { linear1: nn::Linear, linear2: nn::Linear, dropout: f64 }
impl FeedForward {
    fn new(vs: &nn::Path, d_model: i64, d_ff: i64, dropout: f64) -> Self {
        let linear1 = nn::linear(vs / "linear1", d_model, d_ff, Default::default());
        let linear2 = nn::linear(vs / "linear2", d_ff, d_model, Default::default());
        Self { linear1, linear2, dropout }
    }
    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply(&self.linear1).gelu("none").dropout(self.dropout, train).apply(&self.linear2)
    }
}

struct MultiHeadAttention {
    query: nn::Linear, key: nn::Linear, value: nn::Linear, out: nn::Linear,
    n_heads: i64, d_k: i64, dropout: f64,
}
impl MultiHeadAttention {
    fn new(vs: &nn::Path, d_model: i64, n_heads: i64, dropout: f64) -> Self {
        let d_k = d_model / n_heads;
        let query = nn::linear(vs / "query", d_model, d_model, Default::default());
        let key = nn::linear(vs / "key", d_model, d_model, Default::default());
        let value = nn::linear(vs / "value", d_model, d_model, Default::default());
        let out = nn::linear(vs / "out", d_model, d_model, Default::default());
        Self { query, key, value, out, n_heads, d_k, dropout }
    }
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        let bsz = q.size()[0];
        let q_len = q.size()[1];
        let k_len = k.size()[1];

        let q = q.apply(&self.query).view([bsz, q_len, self.n_heads, self.d_k]).transpose(1, 2);
        let k = k.apply(&self.key).view([bsz, k_len, self.n_heads, self.d_k]).transpose(1, 2);
        let v = v.apply(&self.value).view([bsz, k_len, self.n_heads, self.d_k]).transpose(1, 2);

        let mut scores = q.matmul(&k.transpose(-2, -1)) / (self.d_k as f64).sqrt();
        if let Some(m) = mask { scores = scores + m; } // additive mask
        let attn = scores.softmax(-1, Kind::Float).dropout(self.dropout, train);
        let out = attn.matmul(&v);
        out.transpose(1, 2).contiguous().view([bsz, q_len, self.n_heads * self.d_k]).apply(&self.out)
    }
}

struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention, feed_forward: FeedForward,
    norm1: nn::LayerNorm, norm2: nn::LayerNorm, dropout: f64
}
impl TransformerEncoderLayer {
    fn new(vs: &nn::Path, d_model: i64, n_heads: i64, d_ff: i64, dropout: f64) -> Self {
        let self_attn = MultiHeadAttention::new(&(vs / "self_attn"), d_model, n_heads, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), d_model, d_ff, dropout);
        let cfg = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let norm1 = nn::layer_norm(vs / "norm1", vec![d_model], cfg);
        let norm2 = nn::layer_norm(vs / "norm2", vec![d_model], cfg);
        Self { self_attn, feed_forward, norm1, norm2, dropout }
    }
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        let a = self.self_attn.forward(x, x, x, mask, train);
        let x = (x + a.dropout(self.dropout, train)).apply(&self.norm1);
        let f = self.feed_forward.forward(&x, train);
        (x + f.dropout(self.dropout, train)).apply(&self.norm2)
    }
}

struct BERTEncoder {
    embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    encoder_layers: Vec<TransformerEncoderLayer>,
    layer_norm: nn::LayerNorm,
    dropout: f64,
    pad_token_id: i64,
}
impl BERTEncoder {
    fn new(vs: &nn::Path, config: &IndoBERTConfig, pad_token_id: i64) -> Self {
        let embedding = nn::embedding(vs / "embedding", config.vocab_size, config.hidden_size, Default::default());
        let position_embedding = nn::embedding(vs / "position_embedding", config.max_position_embeddings, config.hidden_size, Default::default());
        let mut encoder_layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            encoder_layers.push(TransformerEncoderLayer::new(
                &(vs / format!("encoder_{}", i)),
                config.hidden_size, config.num_attention_heads, config.intermediate_size, config.hidden_dropout_prob,
            ));
        }
        let cfg = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let layer_norm = nn::layer_norm(vs / "layer_norm", vec![config.hidden_size], cfg);
        Self { embedding, position_embedding, encoder_layers, layer_norm, dropout: config.hidden_dropout_prob, pad_token_id }
    }

    fn forward(&self, input_ids: &Tensor, train: bool) -> (Tensor, Tensor) {
        let (bsz, seq_len) = (input_ids.size()[0], input_ids.size()[1]);
        let positions = Tensor::arange(seq_len, (Kind::Int64, input_ids.device()))
            .unsqueeze(0).expand([bsz, seq_len], false);

        let mut x = input_ids.apply(&self.embedding) + positions.apply(&self.position_embedding);
        x = x.apply(&self.layer_norm).dropout(self.dropout, train);

        // pad mask: [bsz,1,1,seqlen] dengan -1e9 pada PAD
        let pad_mask = input_ids.eq(self.pad_token_id).to_kind(Kind::Float).unsqueeze(1).unsqueeze(2) * -1e9;

        for layer in &self.encoder_layers {
            x = layer.forward(&x, Some(&pad_mask), train);
        }
        (x, pad_mask)
    }
}

struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention, cross_attn: MultiHeadAttention, feed_forward: FeedForward,
    norm1: nn::LayerNorm, norm2: nn::LayerNorm, norm3: nn::LayerNorm, dropout: f64,
}
impl TransformerDecoderLayer {
    fn new(vs: &nn::Path, d_model: i64, n_heads: i64, d_ff: i64, dropout: f64) -> Self {
        let self_attn = MultiHeadAttention::new(&(vs / "self_attn"), d_model, n_heads, dropout);
        let cross_attn = MultiHeadAttention::new(&(vs / "cross_attn"), d_model, n_heads, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), d_model, d_ff, dropout);
        let cfg = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let norm1 = nn::layer_norm(vs / "norm1", vec![d_model], cfg);
        let norm2 = nn::layer_norm(vs / "norm2", vec![d_model], cfg);
        let norm3 = nn::layer_norm(vs / "norm3", vec![d_model], cfg);
        Self { self_attn, cross_attn, feed_forward, norm1, norm2, norm3, dropout }
    }
    fn forward(&self, x: &Tensor, encoder_output: &Tensor, causal_mask: &Tensor, enc_pad_mask: &Tensor, train: bool) -> Tensor {
        let a = self.self_attn.forward(x, x, x, Some(causal_mask), train);
        let x = (x + a.dropout(self.dropout, train)).apply(&self.norm1);
        let c = self.cross_attn.forward(&x, encoder_output, encoder_output, Some(enc_pad_mask), train);
        let x = (x + c.dropout(self.dropout, train)).apply(&self.norm2);
        let f = self.feed_forward.forward(&x, train);
        (x + f.dropout(self.dropout, train)).apply(&self.norm3)
    }
}

struct SummaryDecoder {
    embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    decoder_layers: Vec<TransformerDecoderLayer>,
    layer_norm: nn::LayerNorm,
    output_projection: nn::Linear,
    dropout: f64,
}
impl SummaryDecoder {
    fn new(vs: &nn::Path, config: &IndoBERTConfig, num_layers: i64) -> Self {
        let embedding = nn::embedding(vs / "embedding", config.vocab_size, config.hidden_size, Default::default());
        let position_embedding = nn::embedding(vs / "position_embedding", config.max_position_embeddings, config.hidden_size, Default::default());
        let mut decoder_layers = Vec::new();
        for i in 0..num_layers {
            decoder_layers.push(TransformerDecoderLayer::new(&(vs / format!("layer_{}", i)),
                config.hidden_size, config.num_attention_heads, config.intermediate_size, config.hidden_dropout_prob));
        }
        let cfg = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let layer_norm = nn::layer_norm(vs / "layer_norm", vec![config.hidden_size], cfg);
        let output_projection = nn::linear(vs / "output_projection", config.hidden_size, config.vocab_size, Default::default());
        Self { embedding, position_embedding, decoder_layers, layer_norm, output_projection, dropout: config.hidden_dropout_prob }
    }
    fn forward(&self, input_ids: &Tensor, encoder_output: &Tensor, enc_pad_mask: &Tensor, train: bool) -> Tensor {
        let (bsz, seq_len) = (input_ids.size()[0], input_ids.size()[1]);
        let positions = Tensor::arange(seq_len, (Kind::Int64, input_ids.device()))
            .unsqueeze(0).expand([bsz, seq_len], false);

        let mut x = input_ids.apply(&self.embedding) + positions.apply(&self.position_embedding);
        x = x.apply(&self.layer_norm).dropout(self.dropout, train);

        let causal = Tensor::ones([seq_len, seq_len], (Kind::Float, input_ids.device()))
            .tril(0).view([1, 1, seq_len, seq_len]);
        let causal = (causal - 1.0) * 1e9;

        for layer in &self.decoder_layers {
            x = layer.forward(&x, encoder_output, &causal, enc_pad_mask, train);
        }
        x.apply(&self.output_projection)
    }
}

// ======================== SEQ2SEQ ========================

struct Seq2SeqSummarizer {
    encoder: BERTEncoder,
    decoder: SummaryDecoder,
}
impl Seq2SeqSummarizer {
    fn new(vs: &nn::Path, config: &IndoBERTConfig, pad_token_id: i64) -> Self {
        let encoder = BERTEncoder::new(&(vs / "encoder"), config, pad_token_id);
        let decoder = SummaryDecoder::new(&(vs / "decoder"), config, 4);
        Self { encoder, decoder }
    }

    fn forward(&self, src: &Tensor, tgt: &Tensor, train: bool) -> (Tensor, Tensor) {
        let (enc_out, enc_mask) = self.encoder.forward(src, train);
        let logits = self.decoder.forward(tgt, &enc_out, &enc_mask, train);
        (logits, enc_mask)
    }

    fn generate(
        &self,
        src: &Tensor,
        max_len: i64,
        start_token: i64,
        end_token: i64,
        pad_token: i64,
        min_new_tokens: i64,
    ) -> Tensor {
        let (enc_out, enc_mask) = self.encoder.forward(src, false);
        let bsz = src.size()[0];
        let device = src.device();
        let mut generated = Tensor::full([bsz, 1], start_token, (Kind::Int64, device));

        for t in 0..max_len {
            let logits = self.decoder.forward(&generated, &enc_out, &enc_mask, false);
            let mut step = logits.select(1, -1).squeeze_dim(1);

            if t < min_new_tokens {
                // ban PAD & SEP di awal
                let banned = Tensor::from_slice(&[pad_token, end_token]).to_device(device);
                let mut mask_row = Tensor::zeros([step.size()[1]], (Kind::Float, device));
                mask_row = mask_row.index_fill(-1, &banned, -1e9);
                step = step + mask_row.unsqueeze(0);
            }

            let next_token = step.argmax(-1, false).unsqueeze(-1);
            generated = Tensor::cat(&[generated, next_token.copy()], 1);

            let last = generated.select(1, -1);
            if bool::try_from(&last.eq(end_token).all()).unwrap_or(false) { break; }
        }
        generated
    }
}

// ======================== TRAINING ========================

struct TrainingConfig { learning_rate: f64, batch_size: i64, epochs: i64, device: Device }

fn load_data(file_path: &str) -> Result<Vec<SummarizationData>> {
    let file = File::open(file_path).context(format!("Failed to open file: {}", file_path))?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut data = Vec::new();
    for result in reader.deserialize() {
        let record: SummarizationData = result?;
        data.push(record);
    }
    Ok(data)
}

fn prepare_batch(data: &[SummarizationData], tokenizer: &IndoBERTTokenizer, device: Device) -> (Tensor, Tensor, Tensor) {
    let batch_size = data.len() as i64;
    let max_len = tokenizer.max_length as usize;
    let mut src_data: Vec<i64> = Vec::with_capacity((batch_size as usize) * max_len);
    let mut tgt_input_data: Vec<i64> = Vec::with_capacity((batch_size as usize) * max_len);
    let mut tgt_output_data: Vec<i64> = Vec::with_capacity((batch_size as usize) * max_len);

    for item in data {
        let src_tokens = tokenizer.encode(&item.text);
        let tgt_tokens = tokenizer.encode(&item.summary);

        src_data.extend(&src_tokens);

        let mut tgt_input = vec![tokenizer.cls_token_id];
        tgt_input.extend(&tgt_tokens[1..max_len.min(tgt_tokens.len())]);
        while tgt_input.len() < max_len { tgt_input.push(tokenizer.pad_token_id); }
        tgt_input.truncate(max_len);
        tgt_input_data.extend(&tgt_input);

        let mut tgt_output = tgt_tokens.clone();
        tgt_output.truncate(max_len);
        while tgt_output.len() < max_len { tgt_output.push(tokenizer.pad_token_id); }
        tgt_output_data.extend(&tgt_output);
    }

    let src = Tensor::from_slice(&src_data).view([batch_size, tokenizer.max_length]).to(device);
    let tgt_input = Tensor::from_slice(&tgt_input_data).view([batch_size, tokenizer.max_length]).to(device);
    let tgt_output = Tensor::from_slice(&tgt_output_data).view([batch_size, tokenizer.max_length]).to(device);

    (src, tgt_input, tgt_output)
}

fn train_epoch(
    model: &Seq2SeqSummarizer,
    optimizer: &mut nn::Optimizer,
    data: &mut [SummarizationData],
    tokenizer: &IndoBERTTokenizer,
    config: &TrainingConfig,
) -> f64 {
    let mut total_loss = 0.0;
    let mut batch_count = 0;

    data.shuffle(&mut thread_rng()); // <-- FIX: trait SliceRandom + thread_rng

    let batches: Vec<_> = data.chunks(config.batch_size as usize).collect();

    let pb = ProgressBar::new(batches.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Loss: {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for batch in batches {
        let (src, tgt_input, tgt_output) = prepare_batch(batch, tokenizer, config.device);
        let (logits, _enc_mask) = model.forward(&src, &tgt_input, true);

        let bsz = logits.size()[0];
        let seq_len = logits.size()[1];
        let vocab = logits.size()[2];

        let logits_flat = logits.view([bsz * seq_len, vocab]);
        let targets_flat = tgt_output.view([bsz * seq_len]);

        // Masked CE: ignore PAD
        let log_probs = logits_flat.log_softmax(-1, Kind::Float);
        let nll = -log_probs.gather(1, &targets_flat.unsqueeze(1), false).squeeze_dim(1);
        let mask = targets_flat.ne(tokenizer.pad_token_id);
        let loss = nll.masked_select(&mask).mean(Kind::Float);

        optimizer.zero_grad();
        loss.backward();

        for var in optimizer.trainable_variables() {
            let mut grad = var.grad();
            let _ = grad.clamp_(-1.0, 1.0);
        }
        optimizer.step();

        total_loss += f64::try_from(loss).unwrap();
        batch_count += 1;
        pb.inc(1);
        pb.set_message(format!("{:.4}", total_loss / batch_count as f64));
    }

    pb.finish_with_message("Epoch completed");
    total_loss / batch_count as f64
}

// ======================== EVALUATION ========================

fn evaluate_model(
    model: &Seq2SeqSummarizer,
    test_data: &[SummarizationData],
    tokenizer: &IndoBERTTokenizer,
    device: Device,
) -> RougeScores {
    println!("\nEvaluating on test set...");
    let pb = ProgressBar::new(test_data.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Evaluating...")
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut all_rouge_1 = Vec::new();
    let mut all_rouge_2 = Vec::new();
    let mut all_rouge_l = Vec::new();

    for sample in test_data {
        let src_tokens = tokenizer.encode(&sample.text);
        let src = Tensor::from_slice(&src_tokens)
            .view([1, tokenizer.max_length])
            .to(device);

        let generated = no_grad(|| {
            model.generate(
                &src,
                64,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                10,
            )
        });

        let generated_ids: Vec<i64> = generated.view([-1]).try_into().unwrap();
        let generated_text = tokenizer.decode(&generated_ids);

        let rouge = calculate_rouge(&sample.summary, &generated_text);
        all_rouge_1.push(rouge.rouge_1_f);
        all_rouge_2.push(rouge.rouge_2_f);
        all_rouge_l.push(rouge.rouge_l_f);

        pb.inc(1);
    }

    pb.finish_with_message("Evaluation completed");

    let avg = |v: &Vec<f64>| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    RougeScores {
        rouge_1_f: avg(&all_rouge_1),
        rouge_2_f: avg(&all_rouge_2),
        rouge_l_f: avg(&all_rouge_l),
    }
}

// ======================== MAIN FUNCTION ========================

fn main() -> Result<()> {
    println!(" Starting IndoBERT Summarization Training...");

    // Load configuration and vocabulary
    let config = load_indobert_config("indobert_config.json")
        .context("Failed to load IndoBERT config")?;
    let vocab = load_indobert_vocab("/Users/mraffyzeidan/Learning/TransKI/fine_tuned104/indobert_vocab.txt ")
        .context("Failed to load IndoBERT vocabulary")?;

    // Initialize tokenizer
    let mut tokenizer = IndoBERTTokenizer::new(vocab, 512, config.vocab_size);

    // Load dataset
    println!(" Loading dataset...");
    let mut data = load_data("/Users/mraffyzeidan/Learning/TransKI/fine_tuned104/Benchmark.csv")
        .context("Failed to load summarization data")?;

    // Optional diagnostics
    println!(" Checking initial vocabulary coverage...");
    check_vocab_coverage(&data, &tokenizer);

    // ⚠️ Jangan extend vocab tanpa resize embedding model
    // tokenizer.extend_vocab_with_dataset(&data);

    // Split data into train and test
    let test_size = (data.len() as f64 * 0.1) as usize;
    let test_data = data.split_off(data.len() - test_size);
    let mut train_data = data; // <-- FIX: jangan clone, gunakan mutable asli

    println!(" Dataset split: {} training, {} test", train_data.len(), test_data.len());

    // Initialize model
    let device = Device::cuda_if_available();
    println!(" Using device: {:?}", device);

    let vs = nn::VarStore::new(device);
    let model = Seq2SeqSummarizer::new(&vs.root(), &config, tokenizer.pad_token_id);

    // Training configuration
    let train_config = TrainingConfig { learning_rate: 5e-5, batch_size: 4, epochs: 1, device };

    // Initialize optimizer
    let mut optimizer = nn::AdamW::default().build(&vs, train_config.learning_rate)?;

    // Training loop
    println!(" Starting training...");
    for epoch in 0..train_config.epochs {
        println!("\n Epoch {}/{}", epoch + 1, train_config.epochs);

        let train_loss = train_epoch(
            &model,
            &mut optimizer,
            &mut train_data,
            &tokenizer,
            &train_config,
        );

        println!(" Epoch {} - Average Loss: {:.4}", epoch + 1, train_loss);

        // Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0 {
            let rouge_scores = evaluate_model(&model, &test_data, &tokenizer, device);
            println!(
                " Epoch {} - ROUGE Scores: R1: {:.4}, R2: {:.4}, RL: {:.4}",
                epoch + 1,
                rouge_scores.rouge_1_f,
                rouge_scores.rouge_2_f,
                rouge_scores.rouge_l_f
            );
        }

        // Save model checkpoint
        if (epoch + 1) % 5 == 0 {
            let model_path = format!("indobert_summarizer_epoch_{}.pt", epoch + 1);
            vs.save(&model_path).context(format!("Failed to save model to {}", model_path))?;
            println!(" Saved model checkpoint: {}", model_path);
        }
    }

    // Final evaluation
    println!("\n Training completed! Running final evaluation...");
    let final_rouge = evaluate_model(&model, &test_data, &tokenizer, device);
    println!(
        " Final ROUGE Scores: R1: {:.4}, R2: {:.4}, RL: {:.4}",
        final_rouge.rouge_1_f, final_rouge.rouge_2_f, final_rouge.rouge_l_f
    );

    // Save final model
    vs.save("indobert_summarizer_final.pt").context("Failed to save final model")?;
    println!(" Saved final model: indobert_summarizer_final.pt");

    // Test inference on a few samples
    println!("\n Testing inference on a few samples...");
    let mut rng = thread_rng(); // <-- FIX
    let test_samples = test_data.choose_multiple(&mut rng, 3); // <-- FIX

    for (i, sample) in test_samples.into_iter().enumerate() { // <-- gunakan into_iter
        println!("\n--- Sample {} ---", i + 1);
        println!("Original Text: {}", sample.text);
        println!("Reference Summary: {}", sample.summary);

        let src_tokens = tokenizer.encode(&sample.text);
        let src = Tensor::from_slice(&src_tokens)
            .view([1, tokenizer.max_length])
            .to(device);

        let generated = no_grad(|| {
            model.generate(
                &src,
                64,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                10,
            )
        });

        let generated_ids: Vec<i64> = generated.view([-1]).try_into().unwrap();
        let generated_text = tokenizer.decode(&generated_ids);

        println!("Generated Summary: {}", generated_text);

        let rouge = calculate_rouge(&sample.summary, &generated_text);
        println!(
            "ROUGE Scores: R1: {:.4}, R2: {:.4}, RL: {:.4}",
            rouge.rouge_1_f, rouge.rouge_2_f, rouge.rouge_l_f
        );
    }

    Ok(())
}
