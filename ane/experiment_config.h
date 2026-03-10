// experiment_config.h — Agent edits ONLY this file
// Architecture (changing these resets checkpoint)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define SEQ 256
#define NLAYERS 12
// Optimizer (safe to change between runs)
#define LEARNING_RATE 3e-4f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1e-8f
#define ACCUM_STEPS 4
#define GRAD_CLIP_MAX 1.0f
// Experiment knobs to explore:
//   LR: try 8e-4, 9e-4, 1e-3, 1.1e-3, 1.2e-3 (best so far: 1e-3)
//   ACCUM_STEPS: try 1-2 for more weight updates per wall-time budget
//   HIDDEN: try 3072 (4x DIM) for wider FFN
//   SEQ: try 128 for faster steps (more iterations per budget)
