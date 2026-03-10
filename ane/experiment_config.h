// experiment_config.h — Agent edits ONLY this file
// Architecture (changing these resets checkpoint)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define SEQ 512
#define NLAYERS 6
// Optimizer (safe to change between runs)
#define LEARNING_RATE 3e-4f
#define ADAM_BETA1 0.8f
#define ADAM_BETA2 0.95f
#define ADAM_EPS 1e-8f
#define ACCUM_STEPS 1
#define GRAD_CLIP_MAX 1.0f
#define WEIGHT_DECAY 0.2f
#define WARMDOWN_RATIO 0.0f
// Reference: ncdrone/autoresearch-ANE achieved val_loss=5.81 with NL=6 SEQ=512
// Key finding: "more steps > bigger model" — fewer layers = faster steps = more training
// Experiment knobs to explore:
//   LR: try 8e-4, 9e-4, 1e-3, 1.1e-3, 1.2e-3 (best so far: 1e-3 at 12 layers)
//   ACCUM_STEPS: try 4-10 (fewer = more weight updates per budget)
//   NLAYERS: 6 is sweet spot at SEQ=512 (U-shaped curve with depth)
//   SEQ: 512 optimal; 1024 hits SRAM wall on ANE
//   WEIGHT_DECAY: CUDA uses 0.2, start at 0.1 (only on weight matrices, not embed/rmsnorm)
//   WARMDOWN_RATIO: fraction of wall-time for LR decay to 0 (CUDA uses 0.5)
//   ADAM_BETA1/BETA2: CUDA uses (0.8, 0.95) — try these values
