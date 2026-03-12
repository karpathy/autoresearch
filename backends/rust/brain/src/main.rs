use anyhow::Result;
use autoresearch_brain::config::SEQ;
use autoresearch_brain::optim::{Schedule, ScheduleConfig};
use autoresearch_brain::train::{train, TrainConfig};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut load_checkpoint: Option<String> = None;
    let mut diagnostic_steps: Option<usize> = None;
    let mut data_dir: Option<String> = None;
    let mut tokenizer_dir: Option<String> = None;
    let mut stream_input = false;

    let mut max_steps_cli: Option<usize> = None;
    let mut synthetic_cli = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--load-checkpoint" => { i += 1; load_checkpoint = Some(args[i].clone()); }
            "--diagnostic" => { i += 1; diagnostic_steps = Some(args[i].parse().expect("--diagnostic expects a number")); }
            "--data-dir" => { i += 1; data_dir = Some(args[i].clone()); }
            "--tokenizer-path" | "--tokenizer-dir" => { i += 1; tokenizer_dir = Some(args[i].clone()); }
            "--stream-input" => { stream_input = true; }
            "--max-steps" => { i += 1; max_steps_cli = Some(args[i].parse().expect("--max-steps expects a number")); }
            "--synthetic-data" => { i += 1; synthetic_cli = args[i] == "true" || args[i] == "1"; }
            "train" => {} // legacy positional arg
            _ => {}
        }
        i += 1;
    }

    // MAX_STEPS=0 or unset → run indefinitely (until file trigger + cooldown)
    // CLI --max-steps takes precedence over env var MAX_STEPS
    let synthetic_data = synthetic_cli || std::env::var("SYNTHETIC").map(|v| v == "1" || v == "true").unwrap_or(false);
    let max_steps: Option<usize> = max_steps_cli.or_else(|| std::env::var("MAX_STEPS")
        .ok().and_then(|s| s.parse::<usize>().ok()).filter(|&n| n > 0));
    let cooldown_steps: usize = std::env::var("COOLDOWN_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(500);
    let batch_size: usize = std::env::var("BATCH_SIZE")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let num_train_shards: Option<usize> = std::env::var("NUM_TRAIN_SHARDS")
        .ok().and_then(|s| s.parse().ok());
    let total_batch: usize = std::env::var("TOTAL_BATCH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(524288);

    // Schedule hyperparameters (env vars for sweep without recompile)
    let default_sched = ScheduleConfig::default();
    let peak_lr: f64 = std::env::var("PEAK_LR")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(default_sched.peak_lr);
    let warmdown_ratio: f64 = std::env::var("WARMDOWN_RATIO")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(default_sched.warmdown_ratio);
    let weight_decay: f64 = std::env::var("WEIGHT_DECAY")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(default_sched.weight_decay);
    let schedule: Schedule = std::env::var("SCHEDULE")
        .map(|s| Schedule::from_str(&s)).unwrap_or(default_sched.schedule);
    let final_lr_frac: f64 = std::env::var("FINAL_LR_FRAC")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(default_sched.final_lr_frac);

    // Scale other LRs proportionally if PEAK_LR changed
    let lr_scale = peak_lr / default_sched.peak_lr;
    let schedule_cfg = ScheduleConfig {
        peak_lr,
        warmdown_ratio,
        weight_decay,
        schedule,
        final_lr_frac,
        embedding_lr: default_sched.embedding_lr * lr_scale,
        unembedding_lr: default_sched.unembedding_lr * lr_scale,
        scalar_lr: default_sched.scalar_lr * lr_scale,
    };

    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let config = TrainConfig {
        device_batch_size: batch_size,
        total_batch_size: total_batch,
        time_budget_s: 86400.0,
        max_steps,
        cooldown_steps,
        data_dir: data_dir.unwrap_or_else(|| format!("{home}/.cache/autoresearch/shards")),
        tokenizer_dir: tokenizer_dir.unwrap_or_else(|| format!("{home}/.cache/autoresearch/tokenizer")),
        checkpoint_dir: std::env::var("CHECKPOINT_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/autoresearch/checkpoints")),
        eval_interval: std::env::var("EVAL_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(25),
        checkpoint_interval: std::env::var("CHECKPOINT_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(100),
        load_checkpoint,
        diagnostic_steps,
        num_train_shards,
        stream_input,
        synthetic_data,
        schedule_cfg,
    };

    println!("autoresearch-brain");
    if config.synthetic_data { println!("  mode:           SYNTHETIC (MFU baseline — no real data)"); }
    println!("  max_steps:      {}", config.max_steps.map(|n| n.to_string()).unwrap_or("∞".into()));
    println!("  cooldown_steps: {}", config.cooldown_steps);
    println!("  batch:          {}x{} = {} tokens/step", config.device_batch_size,
             config.total_batch_size / (config.device_batch_size * SEQ),
             config.total_batch_size);
    println!("  data:           {}", config.data_dir);
    println!("  peak_lr:        {}", config.schedule_cfg.peak_lr);
    println!("  warmdown_ratio: {} (cooldown shape only)", config.schedule_cfg.warmdown_ratio);
    println!("  weight_decay:   {}", config.schedule_cfg.weight_decay);
    println!("  schedule:       {}", config.schedule_cfg.schedule);
    println!("  final_lr_frac:  {}", config.schedule_cfg.final_lr_frac);
    if lr_scale != 1.0 {
        println!("  (LRs scaled {}x: emb={:.4} unemb={:.6} scalar={:.4})",
                 lr_scale, config.schedule_cfg.embedding_lr,
                 config.schedule_cfg.unembedding_lr, config.schedule_cfg.scalar_lr);
    }
    if let Some(n) = config.num_train_shards {
        println!("  train/val split: {n} train shards, rest = val");
    }

    train(config)
}
