// experiment_config.h — Agent edits ONLY this file
// Architecture (changing these resets checkpoint)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define SEQ 256
#define NLAYERS 12
// Optimizer (safe to change between runs)
#define LEARNING_RATE 1e-3f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1e-8f
#define ACCUM_STEPS 4
