use anyhow::{Result, Context};
use csv::ReaderBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Cursor};
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind, TchError};
use tch::no_grad;
use std::path::Path;
use rayon::prelude::*;

#[derive(Debug, Deserialize, Clone)]
struct SummarizationData {
    text: String,
    summary: String,
}

#[derive(Debug, Deserialize, Serialize)]
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

// ============================================================================
// ROUGE & BLEU EVALUATION METRICS
// ============================================================================

#[derive(Debug, Clone)]
struct RougeScores {
    rouge_1_f: f64,
    rouge_1_p: f64,
    rouge_1_r: f64,
    rouge_2_f: f64,
    rouge_2_p: f64,
    rouge_2_r: f64,
    rouge_l_f: f64,
    rouge_l_p: f64,
    rouge_l_r: f64,
}

impl RougeScores {
    fn new() -> Self {
        RougeScores {
            rouge_1_f: 0.0,
            rouge_1_p: 0.0,
            rouge_1_r: 0.0,
            rouge_2_f: 0.0,
            rouge_2_p: 0.0,
            rouge_2_r: 0.0,
            rouge_l_f: 0.0,
            rouge_l_p: 0.0,
            rouge_l_r: 0.0,
        }
    }

    fn average(scores: &[RougeScores]) -> Self {
        let n = scores.len() as f64;
        if n == 0.0 {
            return RougeScores::new();
        }

        RougeScores {
            rouge_1_f: scores.iter().map(|s| s.rouge_1_f).sum::<f64>() / n,
            rouge_1_p: scores.iter().map(|s| s.rouge_1_p).sum::<f64>() / n,
            rouge_1_r: scores.iter().map(|s| s.rouge_1_r).sum::<f64>() / n,
            rouge_2_f: scores.iter().map(|s| s.rouge_2_f).sum::<f64>() / n,
            rouge_2_p: scores.iter().map(|s| s.rouge_2_p).sum::<f64>() / n,
            rouge_2_r: scores.iter().map(|s| s.rouge_2_r).sum::<f64>() / n,
            rouge_l_f: scores.iter().map(|s| s.rouge_l_f).sum::<f64>() / n,
            rouge_l_p: scores.iter().map(|s| s.rouge_l_p).sum::<f64>() / n,
            rouge_l_r: scores.iter().map(|s| s.rouge_l_r).sum::<f64>() / n,
        }
    }
}

fn tokenize_for_metrics(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn get_ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n {
        return vec![];
    }
    tokens.windows(n).map(|w| w.to_vec()).collect()
}

fn calculate_rouge_n(reference: &[String], hypothesis: &[String], n: usize) -> (f64, f64, f64) {
    let ref_ngrams = get_ngrams(reference, n);
    let hyp_ngrams = get_ngrams(hypothesis, n);

    if ref_ngrams.is_empty() || hyp_ngrams.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let ref_set: HashSet<_> = ref_ngrams.iter().collect();
    let hyp_set: HashSet<_> = hyp_ngrams.iter().collect();

    let overlap = ref_set.intersection(&hyp_set).count() as f64;
    
    let precision = if hyp_ngrams.len() > 0 {
        overlap / hyp_ngrams.len() as f64
    } else {
        0.0
    };

    let recall = if ref_ngrams.len() > 0 {
        overlap / ref_ngrams.len() as f64
    } else {
        0.0
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    (f1, precision, recall)
}

fn lcs_length(s1: &[String], s2: &[String]) -> usize {
    if s1.is_empty() || s2.is_empty() {
        return 0;
    }
    
    let m = s1.len();
    let n = s2.len();
    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if s1[i - 1] == s2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[m][n]
}

fn calculate_rouge_l(reference: &[String], hypothesis: &[String]) -> (f64, f64, f64) {
    if reference.is_empty() || hypothesis.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let lcs = lcs_length(reference, hypothesis) as f64;
    
    let precision = lcs / hypothesis.len() as f64;
    let recall = lcs / reference.len() as f64;
    
    let f1 = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    (f1, precision, recall)
}

fn calculate_rouge(reference: &str, hypothesis: &str) -> RougeScores {
    let ref_tokens = tokenize_for_metrics(reference);
    let hyp_tokens = tokenize_for_metrics(hypothesis);

    let (r1_f, r1_p, r1_r) = calculate_rouge_n(&ref_tokens, &hyp_tokens, 1);
    let (r2_f, r2_p, r2_r) = calculate_rouge_n(&ref_tokens, &hyp_tokens, 2);
    let (rl_f, rl_p, rl_r) = calculate_rouge_l(&ref_tokens, &hyp_tokens);

    RougeScores {
        rouge_1_f: r1_f,
        rouge_1_p: r1_p,
        rouge_1_r: r1_r,
        rouge_2_f: r2_f,
        rouge_2_p: r2_p,
        rouge_2_r: r2_r,
        rouge_l_f: rl_f,
        rouge_l_p: rl_p,
        rouge_l_r: rl_r,
    }
}

fn calculate_bleu(reference: &str, hypothesis: &str) -> f64 {
    let ref_tokens = tokenize_for_metrics(reference);
    let hyp_tokens = tokenize_for_metrics(hypothesis);

    if hyp_tokens.is_empty() {
        return 0.0;
    }

    let mut precisions = Vec::new();

    for n in 1..=4 {
        let (_, precision, _) = calculate_rouge_n(&ref_tokens, &hyp_tokens, n);
        precisions.push(precision);
    }

    let bp = if hyp_tokens.len() < ref_tokens.len() {
        (1.0 - (ref_tokens.len() as f64 / hyp_tokens.len() as f64)).exp()
    } else {
        1.0
    };

    let geo_mean = if precisions.iter().any(|&p| p == 0.0) {
        0.0
    } else {
        let log_sum: f64 = precisions.iter().map(|p| p.ln()).sum();
        (log_sum / precisions.len() as f64).exp()
    };

    bp * geo_mean
}

// ============================================================================
// TOKENIZER & DATA PREPARATION (with parallel processing)
// ============================================================================

fn try_load_ckpt(vs: &mut nn::VarStore) -> bool {
    let mut loaded = false;

    // 1) Coba safetensors dulu (stabil & aman)
    if Path::new("clean_state_dict.safetensors").exists() {
        let ok = tch::no_grad(|| vs.read_safetensors("clean_state_dict.safetensors"));
        match ok {
            Ok(()) => {
                let total_params: i64 = vs.variables().values().map(|t| t.numel() as i64).sum();
                println!(" Weights loaded from clean_state_dict.safetensors (params: {total_params})");
                loaded = true;
            }
            Err(e) => {
                eprintln!(" read_safetensors gagal: {e}");
            }
        }
    }

    // 2) (Opsional) Fallback .pt — hanya kalau kamu benar-benar butuh
    //    Catatan: .pt yang valid harus dibuat dgn torch.save(model.state_dict(), "xxx.pt")
    if !loaded && Path::new("clean_state_dict.pt").exists() {
        let ok = tch::no_grad(|| vs.load_partial("clean_state_dict.pt"));
        match ok {
            Ok(unused) => {
                println!("✓ partial .pt loaded");
                if !unused.is_empty() {
                    println!("  unused keys in checkpoint: {}", unused.join(", "));
                }
                let total_params: i64 = vs.variables().values().map(|t| t.numel() as i64).sum();
                println!("   Total parameters: {total_params}");
                loaded = true;
            }
            Err(e) => {
                eprintln!("❌ load_partial (.pt) error: {e}");
            }
        }
    }

    if !loaded {
        println!("→ lanjut train from scratch");
    }
    loaded
}

fn load_indobert_vocab(vocab_path: &str) -> Result<HashMap<String, i64>> {
    let file = File::open(vocab_path)
        .context(format!("Failed to open vocab file: {}", vocab_path))?;
    let reader = BufReader::new(file);
    
    let mut vocab = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let token = line?;
        vocab.insert(token, idx as i64);
    }
    
    println!("✓ Loaded vocabulary: {} tokens", vocab.len());
    Ok(vocab)
}

fn load_indobert_config(config_path: &str) -> Result<IndoBERTConfig> {
    let file = File::open(config_path)
        .context(format!("Failed to open config file: {}", config_path))?;
    let config: IndoBERTConfig = serde_json::from_reader(file)
        .context("Failed to parse config JSON")?;
    
    println!("✓ Loaded IndoBERT config:");
    println!("   - Vocab size: {}", config.vocab_size);
    println!("   - Hidden size: {}", config.hidden_size);
    println!("   - Layers: {}", config.num_hidden_layers);
    println!("   - Attention heads: {}", config.num_attention_heads);
    
    Ok(config)
}

#[derive(Clone)]
struct IndoBERTTokenizer {
    vocab: HashMap<String, i64>,
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
        let cls_token_id = *vocab.get("[CLS]").unwrap_or(&2);
        let sep_token_id = *vocab.get("[SEP]").unwrap_or(&3);
        let unk_token_id = *vocab.get("[UNK]").unwrap_or(&1);
        
        let vocab_size = config_vocab_size;
        
        IndoBERTTokenizer {
            vocab,
            max_length,
            pad_token_id,
            cls_token_id,
            sep_token_id,
            unk_token_id,
            vocab_size,
        }
    }

    fn tokenize_word(&self, word: &str) -> Vec<i64> {
        if let Some(&id) = self.vocab.get(word) {
            if id < self.vocab_size {
                return vec![id];
            }
        }
        
        let word_lower = word.to_lowercase();
        if let Some(&id) = self.vocab.get(&word_lower) {
            if id < self.vocab_size {
                return vec![id];
            }
        }
        
        let subword = format!("##{}", word_lower);
        if let Some(&id) = self.vocab.get(&subword) {
            if id < self.vocab_size {
                return vec![id];
            }
        }
        
        let mut tokens = Vec::new();
        for (i, c) in word_lower.chars().enumerate() {
            let token = if i == 0 {
                c.to_string()
            } else {
                format!("##{}", c)
            };
            
            if let Some(&id) = self.vocab.get(&token) {
                if id < self.vocab_size {
                    tokens.push(id);
                } else {
                    tokens.push(self.unk_token_id);
                }
            } else {
                tokens.push(self.unk_token_id);
            }
        }
        
        if tokens.is_empty() {
            tokens.push(self.unk_token_id);
        }
        
        tokens
    }

    fn encode_sentence(&self, sentence: &str) -> Vec<i64> {
        let mut tokens = vec![self.cls_token_id];
        
        for word in sentence.split_whitespace() {
            let word_tokens = self.tokenize_word(word);
            tokens.extend(word_tokens);
            
            if tokens.len() >= (self.max_length - 1) as usize {
                break;
            }
        }
        
        tokens.push(self.sep_token_id);
        
        while tokens.len() < self.max_length as usize {
            tokens.push(self.pad_token_id);
        }
        
        tokens.truncate(self.max_length as usize);
        tokens
    }

    fn vocab_size(&self) -> i64 {
        self.vocab_size
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(&['.', '!', '?'][..])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.split_whitespace().count() > 2)
        .collect()
}

fn calculate_sentence_labels(text: &str, summary: &str) -> Vec<i64> {
    let sentences = split_sentences(text);
    let summary_lower = summary.to_lowercase();
    
    sentences.iter().map(|sent| {
        let sent_lower = sent.to_lowercase();
        let sent_words: Vec<&str> = sent_lower.split_whitespace().collect();
        
        if sent_words.is_empty() {
            return 0;
        }
        
        let overlap = sent_words.iter()
            .filter(|word| summary_lower.contains(*word))
            .count();
        
        let overlap_ratio = overlap as f64 / sent_words.len() as f64;
        
        if overlap_ratio > 0.3 { 1 } else { 0 }
    }).collect()
}

// PARALLEL DATA PREPARATION
fn prepare_training_data_parallel(
    data: &[SummarizationData],
    tokenizer: &IndoBERTTokenizer,
) -> Vec<(Vec<i64>, i64)> {
    println!("\n Preparing training data (parallel processing)...");
    
    let pb = ProgressBar::new(data.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Processing documents...")
            .unwrap()
            .progress_chars("#>-")
    );
    
    // Process documents in parallel
    let train_data: Vec<_> = data.par_iter()
        .flat_map(|item| {
            let sentences = split_sentences(&item.text);
            let labels = calculate_sentence_labels(&item.text, &item.summary);
            
            pb.inc(1);
            
            sentences.iter()
                .zip(labels.iter())
                .map(|(sentence, label)| {
                    let tokens = tokenizer.encode_sentence(sentence);
                    (tokens, *label)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    
    pb.finish_with_message("Data preparation completed");
    train_data
}

// ============================================================================
// MODEL ARCHITECTURE
// ============================================================================

struct FeedForward {
    linear1: nn::Linear,
    linear2: nn::Linear,
    dropout: f64,
}

impl FeedForward {
    fn new(vs: &nn::Path, d_model: i64, d_ff: i64, dropout: f64) -> Self {
        let linear1 = nn::linear(vs / "linear1", d_model, d_ff, Default::default());
        let linear2 = nn::linear(vs / "linear2", d_ff, d_model, Default::default());
        
        FeedForward {
            linear1,
            linear2,
            dropout,
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply(&self.linear1)
            .gelu("none")
            .dropout(self.dropout, train)
            .apply(&self.linear2)
    }
}

struct MultiHeadAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    out: nn::Linear,
    n_heads: i64,
    d_k: i64,
    dropout: f64,
}

impl MultiHeadAttention {
    fn new(vs: &nn::Path, d_model: i64, n_heads: i64, dropout: f64) -> Self {
        let d_k = d_model / n_heads;
        
        let query = nn::linear(vs / "query", d_model, d_model, Default::default());
        let key = nn::linear(vs / "key", d_model, d_model, Default::default());
        let value = nn::linear(vs / "value", d_model, d_model, Default::default());
        let out = nn::linear(vs / "out", d_model, d_model, Default::default());
        
        MultiHeadAttention {
            query,
            key,
            value,
            out,
            n_heads,
            d_k,
            dropout,
        }
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        let batch_size = q.size()[0];
        let seq_len = q.size()[1];
        
        let q = q.apply(&self.query).view([batch_size, seq_len, self.n_heads, self.d_k]).transpose(1, 2);
        let k = k.apply(&self.key).view([batch_size, seq_len, self.n_heads, self.d_k]).transpose(1, 2);
        let v = v.apply(&self.value).view([batch_size, seq_len, self.n_heads, self.d_k]).transpose(1, 2);
        
        let scores = q.matmul(&k.transpose(-2, -1)) / (self.d_k as f64).sqrt();
        
        let scores = if let Some(m) = mask {
            scores + m
        } else {
            scores
        };
        
        let attention = scores.softmax(-1, Kind::Float).dropout(self.dropout, train);
        let output = attention.matmul(&v);
        
        let output = output.transpose(1, 2).contiguous().view([batch_size, seq_len, self.n_heads * self.d_k]);
        output.apply(&self.out)
    }
}

struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    dropout: f64,
}

impl TransformerEncoderLayer {
    fn new(vs: &nn::Path, d_model: i64, n_heads: i64, d_ff: i64, dropout: f64) -> Self {
        let self_attn = MultiHeadAttention::new(&(vs / "self_attn"), d_model, n_heads, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), d_model, d_ff, dropout);
        
        let norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let norm1 = nn::layer_norm(vs / "norm1", vec![d_model], norm_config);
        let norm2 = nn::layer_norm(vs / "norm2", vec![d_model], norm_config);
        
        TransformerEncoderLayer {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            dropout,
        }
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, train: bool) -> Tensor {
        let attn_output = self.self_attn.forward(x, x, x, mask, train);
        let x = x + attn_output.dropout(self.dropout, train);
        let x = x.apply(&self.norm1);
        
        let ff_output = self.feed_forward.forward(&x, train);
        let x = x + ff_output.dropout(self.dropout, train);
        x.apply(&self.norm2)
    }
}

struct BERTSummarizer {
    embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    embedding_projection: Option<nn::Linear>,
    encoder_layers: Vec<TransformerEncoderLayer>,
    classifier: nn::Linear,
    dropout: f64,
    layer_norm: nn::LayerNorm,
    embedding_size: i64,
    hidden_size: i64,
}

impl BERTSummarizer {
    fn new(vs: &nn::Path, config: &IndoBERTConfig) -> Self {
        // DULU: let embedding_size = 128;
        let embedding_size = config.hidden_size; // 768
        let hidden_size = config.hidden_size;    // 768

        let embedding = nn::embedding(
            vs / "embedding",
            config.vocab_size,
            embedding_size,
            Default::default()
        );

        let position_embedding = nn::embedding(
            vs / "position_embedding",
            config.max_position_embeddings,
            embedding_size,
            Default::default()
        );

        // DULU: Some(linear(...))
        let embedding_projection = None; // hilangkan projection

        let mut encoder_layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            encoder_layers.push(TransformerEncoderLayer::new(
                &(vs / format!("encoder_{}", i)),
                hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.hidden_dropout_prob,
            ));
        }

        let classifier = nn::linear(vs / "classifier", hidden_size, 2, Default::default());

        // ukuran LN harus = embedding_size (sekarang 768)
        let norm_config = nn::LayerNormConfig { eps: 1e-12, ..Default::default() };
        let layer_norm = nn::layer_norm(vs / "layer_norm", vec![embedding_size], norm_config);

        BERTSummarizer {
            embedding,
            position_embedding,
            embedding_projection, // None
            encoder_layers,
            classifier,
            dropout: config.hidden_dropout_prob,
            layer_norm,
            embedding_size,
            hidden_size,
        }
    }

    fn forward(&self, input_ids: &Tensor, train: bool) -> Tensor {
        let (batch_size, seq_len) = (input_ids.size()[0], input_ids.size()[1]);

        let positions = Tensor::arange(seq_len, (Kind::Int64, input_ids.device()))
            .unsqueeze(0)
            .expand([batch_size, seq_len], false);

        let token_embeddings = input_ids.apply(&self.embedding);
        let position_embeddings = positions.apply(&self.position_embedding);

        let mut x = token_embeddings + position_embeddings;
        x = x.apply(&self.layer_norm).dropout(self.dropout, train);

        if let Some(ref proj) = self.embedding_projection {
            x = x.apply(proj);
        }

        for layer in &self.encoder_layers {
            x = layer.forward(&x, None, train);
        }

        let cls_output = x.select(1, 0);
        cls_output.apply(&self.classifier)
    }
}


// ============================================================================
// TORCHSCRIPT HANDLING
// ============================================================================

fn convert_torchscript_to_state_dict(torchscript_path: &str, output_path: &str) -> Result<()> {
    println!(" Converting TorchScript model to state_dict...");
    
    // This would typically be done with Python, but we'll handle it in Rust
    // For now, we'll create a placeholder function
    println!(" Please convert the TorchScript file using Python:");
    println!("   python -c \"import torch; m=torch.jit.load('{}'); torch.save(m.state_dict(), '{}')\"", 
             torchscript_path, output_path);
    
    Ok(())
}

// ============================================================================
// TRAINING & EVALUATION
// ============================================================================

struct TrainingConfig {
    learning_rate: f64,
    batch_size: i64,
    epochs: i64,
    warmup_epochs: i64,
    device: Device,
    num_threads: usize,
}

fn load_data(file_path: &str) -> Result<Vec<SummarizationData>> {
    let file = File::open(file_path)
        .context(format!("Failed to open file: {}", file_path))?;
    
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);
    
    let mut data = Vec::new();
    for result in reader.deserialize() {
        let record: SummarizationData = result?;
        data.push(record);
    }
    
    Ok(data)
}

fn prepare_batch(
    data: &[(Vec<i64>, i64)],
    device: Device,
) -> (Tensor, Tensor) {
    let batch_size = data.len() as i64;
    let seq_len = data[0].0.len() as i64;
    
    let mut input_data: Vec<i64> = vec![];
    let mut label_data = vec![];
    
    for (tokens, label) in data {
        input_data.extend(tokens);
        label_data.push(*label);
    }
    
    let inputs = Tensor::from_slice(&input_data)
        .view([batch_size, seq_len])
        .to(device);
    
    let labels = Tensor::from_slice(&label_data)
        .to(device)
        .to_kind(Kind::Int64);
    
    (inputs, labels)
}

fn validate_input_range(inputs: &Tensor, vocab_size: i64) -> bool {
    let min_val = inputs.min();
    let max_val = inputs.max();
    let min_val_f64 = f64::try_from(min_val).unwrap_or(0.0);
    let max_val_f64 = f64::try_from(max_val).unwrap_or(0.0);
    
    min_val_f64 >= 0.0 && max_val_f64 < vocab_size as f64
}

fn train_epoch(
    model: &BERTSummarizer,
    optimizer: &mut nn::Optimizer,
    data: &[(Vec<i64>, i64)],
    config: &TrainingConfig,
    vocab_size: i64,
) -> f64 {
    let mut total_loss = 0.0;
    let mut batch_count = 0;
    
    let batches: Vec<_> = data.chunks(config.batch_size as usize).collect();
    let pb = ProgressBar::new(batches.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) Loss: {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    
    for batch in batches {
        let (inputs, labels) = prepare_batch(batch, config.device);
        
        if !validate_input_range(&inputs, vocab_size) {
            pb.inc(1);
            continue;
        }
        
        let logits = model.forward(&inputs, true);
        let loss = logits.cross_entropy_for_logits(&labels);
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        total_loss += f64::try_from(loss).unwrap();
        batch_count += 1;
        
        pb.inc(1);
        pb.set_message(format!("{:.4}", total_loss / batch_count as f64));
    }
    
    pb.finish_with_message("Epoch completed");
    if batch_count > 0 {
        total_loss / batch_count as f64
    } else {
        0.0
    }
}

fn evaluate(
    model: &BERTSummarizer,
    data: &[(Vec<i64>, i64)],
    config: &TrainingConfig,
    vocab_size: i64,
) -> (f64, f64, f64, f64) {
    let mut correct = 0;
    let mut total = 0;
    let mut total_loss = 0.0;
    let mut batch_count = 0;
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    
    let pb = ProgressBar::new((data.len() as f64 / config.batch_size as f64).ceil() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Evaluating...")
            .unwrap()
            .progress_chars("#>-")
    );
    
    tch::no_grad(|| {
        for batch in data.chunks(config.batch_size as usize) {
            let (inputs, labels) = prepare_batch(batch, config.device);
            
            if !validate_input_range(&inputs, vocab_size) {
                pb.inc(1);
                continue;
            }
            
            let logits = model.forward(&inputs, false);
            let loss = logits.cross_entropy_for_logits(&labels);
            total_loss += f64::try_from(loss).unwrap();
            batch_count += 1;
            
            let predictions = logits.argmax(-1, false);
            let labels_vec = Vec::<i64>::try_from(labels).unwrap();
            let preds_vec = Vec::<i64>::try_from(predictions).unwrap();
            
            for (pred, label) in preds_vec.iter().zip(labels_vec.iter()) {
                if pred == label {
                    correct += 1;
                }
                if *pred == 1 && *label == 1 {
                    true_positives += 1;
                }
                if *pred == 1 && *label == 0 {
                    false_positives += 1;
                }
                if *pred == 0 && *label == 1 {
                    false_negatives += 1;
                }
                total += 1;
            }
            pb.inc(1);
        }
    });
    
    pb.finish_with_message("Evaluation completed");
    
    let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    let avg_loss = if batch_count > 0 { total_loss / batch_count as f64 } else { 0.0 };
    
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };
    
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };
    
    (accuracy, avg_loss, precision, recall)
}

fn count_parameters(vs: &nn::VarStore) -> i64 {
    let mut total = 0;
    for variable in vs.variables() {
        total += variable.1.size().iter().product::<i64>();
    }
    total
}


// NEW: Improved weight loading with TorchScript support

// Generate summary from model predictions
fn generate_summary_from_model(
    model: &BERTSummarizer,
    tokenizer: &IndoBERTTokenizer,
    text: &str,
    max_seq_len: i64,
    device: Device,
) -> String {
    let sentences = split_sentences(text);
    let mut selected_sentences = Vec::new();
    
    tch::no_grad(|| {
        for sentence in &sentences {
            let tokens = tokenizer.encode_sentence(sentence);
            
            if tokens.iter().all(|&token_id| token_id >= 0 && token_id < tokenizer.vocab_size()) {
                let input = Tensor::from_slice(&tokens)
                    .view([1, max_seq_len])
                    .to(device);
                
                let logits = model.forward(&input, false);
                let pred = logits.argmax(-1, false);
                let pred_val = i64::try_from(pred).unwrap();
                
                if pred_val == 1 {
                    selected_sentences.push(sentence.clone());
                }
            }
        }
    });
    
    selected_sentences.join(" ")
}

// Batch inference for better performance
fn generate_summaries_batch(
    model: &BERTSummarizer,
    tokenizer: &IndoBERTTokenizer,
    texts: &[String],
    max_seq_len: i64,
    device: Device,
) -> Vec<String> {
    let mut all_summaries = Vec::new();
    
    for text in texts {
        let summary = generate_summary_from_model(model, tokenizer, text, max_seq_len, device);
        all_summaries.push(summary);
    }
    
    all_summaries
}

// PARALLEL EVALUATION for ROUGE/BLEU
fn evaluate_summarization(
    model: &BERTSummarizer,
    tokenizer: &IndoBERTTokenizer,
    data: &[SummarizationData],
    max_seq_len: i64,
    device: Device,
) -> (RougeScores, f64) {
    println!("\n Evaluating Summarization Quality...");
    
    let pb = ProgressBar::new(data.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Generating summaries...")
            .unwrap()
            .progress_chars("#>-")
    );
    
    let mut rouge_scores = Vec::new();
    let mut bleu_scores = Vec::new();
    
    // Use batch processing for better performance
    let batch_size = 8;
    let batches: Vec<_> = data.chunks(batch_size).collect();
    
    for batch in batches {
        let texts: Vec<String> = batch.iter().map(|d| d.text.clone()).collect();
        let references: Vec<String> = batch.iter().map(|d| d.summary.clone()).collect();
        
        let generated_summaries = generate_summaries_batch(
            model, tokenizer, &texts, max_seq_len, device
        );
        
        for (generated, reference) in generated_summaries.iter().zip(references.iter()) {
            if !generated.is_empty() {
                let rouge = calculate_rouge(reference, generated);
                let bleu = calculate_bleu(reference, generated);
                
                rouge_scores.push(rouge);
                bleu_scores.push(bleu);
            }
            pb.inc(1);
        }
    }
    
    pb.finish_with_message("Evaluation completed");
    
    let avg_rouge = RougeScores::average(&rouge_scores);
    let avg_bleu = if !bleu_scores.is_empty() {
        bleu_scores.iter().sum::<f64>() / bleu_scores.len() as f64
    } else {
        0.0
    };
    
    (avg_rouge, avg_bleu)
}

// NEW: Function to check and convert TorchScript models
fn setup_pretrained_weights() -> Result<()> {
    let torchscript_paths = [
        "indobert_clean.pt",
        "indobert_state_dict.pt",
    ];
    
    for path in &torchscript_paths {
        if Path::new(path).exists() && !Path::new("converted.pt").exists() {
            println!(" Found TorchScript model: {}", path);
            println!(" Converting to state_dict format...");
            convert_torchscript_to_state_dict(path, "converted.pt")?;
            break;
        }
    }
    
    Ok(())
}

fn main() -> Result<()> {
    // Set number of threads for parallel processing
    let num_threads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    
    // Set PyTorch threads for CPU computation
    tch::set_num_threads(num_threads as i32);
    
    println!(" IndoBERT Summarizer with Improved TorchScript Support");
    println!("   CPU threads: {}", num_threads);
    println!();

    // Setup pretrained weights first
    setup_pretrained_weights()?;

    let config = TrainingConfig {
        learning_rate: 5e-4,
        batch_size: 8,
        epochs: 10,
        warmup_epochs: 2,
        device: Device::Cpu,
        num_threads,
    };
    
    println!("  Configuration:");
    println!("   Device: {:?}", config.device);
    println!("   Batch size: {}", config.batch_size);
    println!("   Learning rate: {}", config.learning_rate);
    println!("   Epochs: {} (+ {} warmup)", config.epochs, config.warmup_epochs);
    println!("   Parallel threads: {}", config.num_threads);
    println!();
    
    println!(" Loading pre-trained files...");
    let vocab = load_indobert_vocab("indobert_vocab.txt")?;
    let indobert_config = load_indobert_config("indobert_config.json")?;
    println!();
    
    let max_seq_len = 128;
    let tokenizer = IndoBERTTokenizer::new(vocab, max_seq_len, indobert_config.vocab_size);
    
    println!(" Loading training data from CSV...");
    let csv_path = "Benchmark.csv";
    let data = load_data(csv_path)?;
    println!("   Loaded {} documents", data.len());
    
    // Use parallel data preparation
    let train_data = prepare_training_data_parallel(&data, &tokenizer);
    
    let important_count = train_data.iter().filter(|(_, l)| *l == 1).count();
    println!("   Total training samples: {}", train_data.len());
    println!("   Important sentences: {} ({:.2}%)", 
        important_count, 
        100.0 * important_count as f64 / train_data.len() as f64
    );
    
    let vocab_size = tokenizer.vocab_size();
    println!("   Vocabulary size: {}", vocab_size);
    
    let initial_count = train_data.len();
    let train_data: Vec<_> = train_data.into_par_iter()
        .filter(|(tokens, _)| {
            tokens.iter().all(|&token_id| token_id >= 0 && token_id < vocab_size)
        })
        .collect();
    
    println!("   Valid training samples after filtering: {} (removed {})", 
             train_data.len(), initial_count - train_data.len());
    
    let split_idx = (train_data.len() as f64 * 0.8) as usize;
    let (train_set, test_set) = train_data.split_at(split_idx);
    println!("   Train set: {}, Test set: {}", train_set.len(), test_set.len());
    
    println!(" Initializing IndoBERT model...");
    // 1) Bangun model dulu
    // ...

    let st_path = "clean_state_dict.safetensors";
    let pt_path = "clean_state_dict.pt"; // opsional fallback, tapi riskan di env kamu

   // sesudah:
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    // 1) register semua parameter ke VarStore dengan membangun model dulu
    let model = BERTSummarizer::new(&vs.root(), &indobert_config);

    // 2) baru load sekali
    let _ = try_load_ckpt(&mut vs);

    // helper kecil buat info apakah ada yang berubah
    fn param_count(vs: &nn::VarStore) -> i64 {
        // versi A: pakai sum::<i64>()
        vs.variables()
            .values()
            .map(|t| t.numel() as i64)
            .sum::<i64>()
    }
    let before_params = param_count(&vs);

    if Path::new(st_path).exists() {
        // cara 1: paling aman – load di dalam no_grad
        match no_grad(|| vs.read_safetensors(st_path)) {
            Ok(()) => {
                let after_params = param_count(&vs);
                println!(" Weights loaded from {st_path} (params: {after_params}, was {before_params})");
            }
            Err(e) => {
                eprintln!(" read_safetensors (no_grad) gagal: {e}");
                // OPTIONAL: fallback ke .pt, tapi ini sering crash di environment kamu
                if Path::new(pt_path).exists() {
                    match no_grad(|| vs.load_partial(pt_path)) {
                        Ok(missing) => {
                            if !missing.is_empty() {
                                println!(" Missing {} keys (partial):", missing.len());
                                for k in missing.iter().take(20) { println!("   - {k}"); }
                                if missing.len() > 20 { println!("   ... and {} more", missing.len() - 20); }
                            }
                            println!(" Fallback load_partial .pt OK");
                        }
                        Err(e2) => eprintln!(" Fallback load_partial gagal: {e2}"),
                    }
                } else {
                    eprintln!(" Fallback .pt ({pt_path}) tidak ditemukan.");
                }
            }
        }
    } else if Path::new(pt_path).exists() {
        // kalau gak ada safetensors, coba .pt di dalam no_grad
        match no_grad(|| vs.load_partial(pt_path)) {
            Ok(missing) => {
                if !missing.is_empty() {
                    println!(" Missing {} keys (partial):", missing.len());
                    for k in missing.iter().take(20) { println!("   - {k}"); }
                    if missing.len() > 20 { println!("   ... and {} more", missing.len() - 20); }
                }
                println!(" Weights loaded from {pt_path}");
            }
            Err(e) => eprintln!(" load_partial gagal: {e}"),
        }
    } else {
        eprintln!(" Tidak ada file pretrain ditemukan ({st_path} / {pt_path}).");
    }

    let total_params = count_parameters(&vs);
    println!("   Total parameters: {}", total_params);

    let mut optimizer = nn::AdamW::default().build(&vs, config.learning_rate)?;
    
    println!("\n Starting Training...\n");
    println!("{}", "─".repeat(70));
    
    let mut best_accuracy = 0.0;
    let mut best_f1 = 0.0;
    let mut best_rouge_1 = 0.0;
    
    for epoch in 1..=(config.epochs + config.warmup_epochs) {
        let is_warmup = epoch <= config.warmup_epochs;
        let phase = if is_warmup { "Warmup" } else { "Training" };
        
        println!("\n Epoch {}/{} [{}]", epoch, config.epochs + config.warmup_epochs, phase);
        
        let train_loss = train_epoch(&model, &mut optimizer, train_set, &config, vocab_size);
        let (test_acc, test_loss, precision, recall) = evaluate(&model, test_set, &config, vocab_size);
        
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };
        
        println!("    Train Loss: {:.4}", train_loss);
        println!("    Test Loss: {:.4}, Accuracy: {:.4}", test_loss, test_acc);
        println!("    Precision: {:.4}, Recall: {:.4}, F1: {:.4}", precision, recall, f1_score);
        
        // Evaluate on a subset for ROUGE/BLEU (every 2 epochs to save time)
        if epoch % 2 == 0 || epoch == config.epochs + config.warmup_epochs {
            let eval_subset_size = 20.min(data.len());
            let eval_subset = &data[..eval_subset_size];
            let (rouge, bleu) = evaluate_summarization(&model, &tokenizer, eval_subset, max_seq_len, config.device);
            
            println!("\n     ROUGE Scores (on {} samples):", eval_subset_size);
            println!("      ROUGE-1: F1={:.4}, P={:.4}, R={:.4}", rouge.rouge_1_f, rouge.rouge_1_p, rouge.rouge_1_r);
            println!("      ROUGE-2: F1={:.4}, P={:.4}, R={:.4}", rouge.rouge_2_f, rouge.rouge_2_p, rouge.rouge_2_r);
            println!("      ROUGE-L: F1={:.4}, P={:.4}, R={:.4}", rouge.rouge_l_f, rouge.rouge_l_p, rouge.rouge_l_r);
            println!("     BLEU Score: {:.4}", bleu);
            
            if !is_warmup && (test_acc > best_accuracy || f1_score > best_f1 || rouge.rouge_1_f > best_rouge_1) {
                best_accuracy = best_accuracy.max(test_acc);
                best_f1 = best_f1.max(f1_score);
                best_rouge_1 = best_rouge_1.max(rouge.rouge_1_f);
                vs.save("best_indobert_summarizer.pt")?;
                println!("\n     Best model saved!");
            }
        }
    }
    
    println!("\n{}", "─".repeat(70));
    println!("\n Training Completed!");
    println!("    Best Test Accuracy: {:.4}", best_accuracy);
    println!("    Best F1 Score: {:.4}", best_f1);
    println!("    Best ROUGE-1 F1: {:.4}", best_rouge_1);
    
    println!("\n Testing Inference with Detailed Metrics...\n");
    if let Some(test_doc) = data.first() {
        println!(" Original Text (first 200 chars):");
        println!("{}\n", &test_doc.text[..test_doc.text.len().min(200)]);
        
        println!(" Expected Summary:");
        println!("{}\n", test_doc.summary);
        
        let generated = generate_summary_from_model(&model, &tokenizer, &test_doc.text, max_seq_len, config.device);
        
        println!(" Generated Summary:");
        println!("{}\n", generated);
        
        let rouge = calculate_rouge(&test_doc.summary, &generated);
        let bleu = calculate_bleu(&test_doc.summary, &generated);
        
        println!(" Quality Metrics for this example:");
        println!("   ROUGE-1 F1: {:.4}", rouge.rouge_1_f);
        println!("   ROUGE-2 F1: {:.4}", rouge.rouge_2_f);
        println!("   ROUGE-L F1: {:.4}", rouge.rouge_l_f);
        println!("   BLEU Score: {:.4}", bleu);
        
        println!("\n Extracted Important Sentences:");
        let sentences = split_sentences(&test_doc.text);
        
        tch::no_grad(|| {
            for (_idx, sentence) in sentences.iter().enumerate().take(10) {
                let tokens = tokenizer.encode_sentence(sentence);
                
                if tokens.iter().all(|&token_id| token_id >= 0 && token_id < tokenizer.vocab_size()) {
                    let input = Tensor::from_slice(&tokens)
                        .view([1, max_seq_len])
                        .to(config.device);
                    
                    let logits = model.forward(&input, false);
                    let pred = logits.argmax(-1, false);
                    let pred_val = i64::try_from(pred).unwrap();
                    
                    if pred_val == 1 {
                        println!("   ✓ {}", sentence);
                    }
                }
            }
        });
    }
    
    println!("\n{}", "═".repeat(70));
    println!(" Done!");
    
    Ok(())
}