// train_ane.m — Dynamic weight ANE training
// Compile kernels ONCE at startup, update weights via IOSurface every step.
// No recompilation, no exec() restart.
#include "stories_mil_dynamic.h"
#include "stories_cpu_ops.h"

#define CKPT_PATH_DEFAULT "ane_stories110M_ckpt.bin"
#define MODEL_PATH_DEFAULT "stories110M.bin"
#define DATA_PATH_DEFAULT "tinystories_data00.bin"

#define VAL_WINDOWS 50

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, (vDSP_Length)cols, (vDSP_Length)rows);
}

// ===== Weight loading from llama2.c format =====
static bool load_pretrained(LayerWeights *lw, float *rms_final, float *embed, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Cannot open %s\n", path); return false; }
    Llama2Config cfg;
    fread(&cfg, sizeof(cfg), 1, f);
    printf("  Model config: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
           cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, abs(cfg.vocab_size), cfg.seq_len);
    if (cfg.dim != DIM || cfg.hidden_dim != HIDDEN || cfg.n_layers != NLAYERS) {
        printf("  ERROR: Config mismatch! Expected dim=%d hidden=%d layers=%d\n", DIM, HIDDEN, NLAYERS);
        fclose(f); return false;
    }
    // GQA check: model must match our N_KV_HEADS config
    int model_kv_heads = cfg.n_kv_heads > 0 ? cfg.n_kv_heads : cfg.n_heads;
    if (model_kv_heads != N_KV_HEADS) {
        printf("  ERROR: Model has n_kv_heads=%d, expected N_KV_HEADS=%d\n", model_kv_heads, N_KV_HEADS);
        fclose(f); return false;
    }
    int V = abs(cfg.vocab_size);
    (void)V;

    fread(embed, 4, VOCAB * DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_att, 4, DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wq, 4, WQ_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wk, 4, WK_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wv, 4, WV_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].Wo, 4, WO_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].rms_ffn, 4, DIM, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W1, 4, W1_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W2, 4, W2_SZ, f);
    for (int L = 0; L < NLAYERS; L++) fread(lw[L].W3, 4, W3_SZ, f);
    fread(rms_final, 4, DIM, f);

    fclose(f);
    printf("  Loaded pretrained weights\n");
    return true;
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};

    // SDPA forward: [1, DIM, 1, SDPA_FWD_SP] → [1, 2*DIM+2*KV_DIM, 1, SEQ]
    printf("  Compiling sdpaFwd...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), mask_w,
        DIM*SDPA_FWD_SP*2, (2*DIM+2*KV_DIM)*SEQ*2);
    if (!dk->sdpaFwd) return false;

    // Wo forward: [1, DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling woFwd...\n");
    dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!dk->woFwd) return false;

    // Fused FFN: [1, DIM, 1, FFN_FUSED_SP] → [1, DIM+3*HIDDEN, 1, SEQ]
    printf("  Compiling ffnFused...\n");
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, (DIM+3*HIDDEN)*SEQ*2);
    if (!dk->ffnFused) return false;

    // FFN backward W2^T: [1, DIM, 1, SEQ+HIDDEN] → [1, HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdW2t...\n");
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*FFN_BWD_W2T_SP*2, HIDDEN*SEQ*2);
    if (!dk->ffnBwdW2t) return false;

    // FFN backward W1^T+W3^T: [1, HIDDEN, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*FFN_BWD_W13T_SP*2, DIM*SEQ*2);
    if (!dk->ffnBwdW13t) return false;

    // Wo^T backward: [1, DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling wotBwd...\n");
    dk->wotBwd = compile_kern_mil_w(gen_wot_bwd_dynamic(), @{},
        DIM*WOT_BWD_SP*2, DIM*SEQ*2);
    if (!dk->wotBwd) return false;

    // SDPA bwd1 (weight-free, has mask): [1, 2*DIM+2*KV_DIM, 1, SEQ] → [1, KV_DIM+2*SCORE_CH, 1, SEQ]
    printf("  Compiling sdpaBwd1...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_dynamic(), mask_w,
        (2*DIM+2*KV_DIM)*SEQ*2, (KV_DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    // SDPA bwd2 (weight-free): [1, 2*SCORE_CH+DIM+KV_DIM, 1, SEQ] → [1, DIM+KV_DIM, 1, SEQ]
    printf("  Compiling sdpaBwd2...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2_dynamic(), @{},
        (2*SCORE_CH+DIM+KV_DIM)*SEQ*2, (DIM+KV_DIM)*SEQ*2);
    if (!dk->sdpaBwd2) return false;

    // Q backward: [1, DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling qBwd...\n");
    dk->qBwd = compile_kern_mil_w(gen_q_bwd_dynamic(), @{},
        DIM*Q_BWD_SP*2, DIM*SEQ*2);
    if (!dk->qBwd) return false;

    // KV backward: [1, KV_DIM, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling kvBwd...\n");
    dk->kvBwd = compile_kern_mil_w(gen_kv_bwd_dynamic(), @{},
        KV_DIM*KV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->kvBwd) return false;

    return true;
}

// ===== Checkpoint save/load =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double ct, double cw, int cs, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 2;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_compile = 0; h.cum_train = ct; h.cum_wall = cw;
    h.cum_steps = cs; h.cum_batches = 0; h.adam_t = adam_t;
    h.n_kv_heads = N_KV_HEADS;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WK_SZ,f); fwrite(la[L].Wk.v,4,WK_SZ,f);
        fwrite(la[L].Wv.m,4,WV_SZ,f); fwrite(la[L].Wv.v,4,WV_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *ct, double *cw, int *cs, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 2) { fclose(f); return false; }
    // GQA backward compat: old checkpoints have n_kv_heads=0, treat as MHA
    int ckpt_kv = (h.n_kv_heads > 0) ? h.n_kv_heads : h.n_heads;
    if (ckpt_kv != N_KV_HEADS) {
        printf("Checkpoint n_kv_heads=%d != N_KV_HEADS=%d, cannot load\n", ckpt_kv, N_KV_HEADS);
        fclose(f); return false;
    }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WK_SZ,f); fread(la[L].Wk.v,4,WK_SZ,f);
        fread(la[L].Wv.m,4,WV_SZ,f); fread(la[L].Wv.v,4,WV_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

// ===== Main =====
int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float lr = LEARNING_RATE;
        float adam_b1 = ADAM_BETA1, adam_b2 = ADAM_BETA2, adam_eps = ADAM_EPS;
        int adam_t = 0, start_step = 0;

        const char *ckpt_path = CKPT_PATH_DEFAULT;
        const char *model_path = MODEL_PATH_DEFAULT;
        const char *data_path = DATA_PATH_DEFAULT;
        bool do_resume = false;
        bool fresh = false;
        bool reset_timing = false;
        int wall_time_budget = 300;
        int pos = 0;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--fresh") == 0) fresh = true;
            else if (strcmp(argv[i], "--reset-timing") == 0) reset_timing = true;
            else if (strcmp(argv[i], "--wall-time") == 0 && i+1<argc) wall_time_budget = atoi(argv[++i]);
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--ckpt") == 0 && i+1<argc) ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--model") == 0 && i+1<argc) model_path = argv[++i];
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
            else if (argv[i][0] != '-') {
                if (pos == 0) model_path = argv[i];
                else if (pos == 2) total_steps = atoi(argv[i]);
                else if (pos == 3) lr = atof(argv[i]);
                pos++;
            }
        }

        // Allocate per-layer state
        LayerWeights lw[NLAYERS];
        LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS];
        LayerGrads grads[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc();
            la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc();
            grads[L] = layer_grads_alloc();
        }

        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        double cum_train=0, cum_wall=0;
        int cum_steps=0;

        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(ckpt_path, &start_step, &total_steps, &lr, &resume_loss,
                &cum_train, &cum_wall, &cum_steps, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) {
                if (reset_timing) {
                    cum_wall = 0; cum_train = 0; cum_steps = 0;
                    start_step = 0;
                    printf("[RESUMED weights (adam_t=%d), reset timing for new experiment]\n", adam_t);
                } else {
                    printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
                }
            }
        }
        if (!resuming) {
            printf("=== ANE Dynamic Training ===\n");
            printf("dim=%d hidden=%d heads=%d seq=%d vocab=%d layers=%d\n", DIM, HIDDEN, HEADS, SEQ, VOCAB, NLAYERS);
            printf("wall_time_budget=%ds\n", wall_time_budget);
            bool loaded = false;
            if (!fresh) loaded = load_pretrained(lw, rms_final, embed, model_path);
            if (!loaded) {
                printf("Using random init (scaled Wo/W2)\n");
                srand48(42);
                float scale_d=1.0f/sqrtf(DIM), scale_h=1.0f/sqrtf(HIDDEN);
                float res_scale = 1.0f/sqrtf(2.0f*NLAYERS);
                for (int L=0; L<NLAYERS; L++) {
                    for(size_t i=0;i<WQ_SZ;i++) lw[L].Wq[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WK_SZ;i++) lw[L].Wk[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WV_SZ;i++) lw[L].Wv[i]=scale_d*(2*drand48()-1);
                    for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=scale_d*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                    for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
                    for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                    for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
                }
                for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
                float escale = 0.02f;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
            }
            size_t tp = (size_t)NLAYERS*LAYER_PARAMS + DIM + (size_t)VOCAB*DIM;
            printf("Params: %.2fM\n", tp/1e6);
            printf("Accum %d steps | Adam LR=%.1e b1=%.1f b2=%.3f\n", ACCUM_STEPS, lr, adam_b1, adam_b2);
        }

        // mmap token data
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) {
            printf("Cannot open token data: %s\n", data_path);
            printf("Hint: run `bash ane/download_data.sh` or pass --data /path/to/tinystories_data00.bin\n");
            return 1;
        }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        if (n_tokens <= (size_t)(SEQ + 1)) {
            printf("Token data too short\n"); munmap(token_data, data_len); close(data_fd); return 1;
        }

        // 90/10 train/val split
        size_t val_start = (size_t)(n_tokens * 0.9);
        size_t train_tokens = val_start;
        size_t val_tokens = n_tokens - val_start;
        printf("Token data: %zu tokens (%.1f MB) | train: %zu val: %zu\n",
               n_tokens, data_len/1e6, train_tokens, val_tokens);

        // Precompute transposed weights for forward kernels
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WK_SZ*4);
            Wvt_buf[L]=(float*)malloc(WV_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
            W1t_buf[L]=(float*)malloc(W1_SZ*4); W3t_buf[L]=(float*)malloc(W3_SZ*4);
            transpose_weight(Wqt_buf[L], lw[L].Wq, DIM, DIM);
            transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
            transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        // ===== Compile all kernels ONCE =====
        printf("Compiling 10 dynamic kernels (one-time)...\n");
        uint64_t tc = mach_absolute_time();
        DynLayerKernels dk;
        if (!compile_dynamic_kernels(&dk)) {
            printf("Compilation failed!\n"); return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled 10 kernels in %.0fms (shared across all %d layers)\n", compile_ms, NLAYERS);

        // Allocate per-layer IOSurfaces + requests
        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
            pls[L].woFwd_in      = make_surface(DIM*WO_FWD_SP*2);
            pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
            pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
            pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
            pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
            pls[L].qBwd_in       = make_surface(DIM*Q_BWD_SP*2);
            pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);

            plr[L].sdpaFwd   = make_request(dk.sdpaFwd,   pls[L].sdpaFwd_in);
            plr[L].woFwd     = make_request(dk.woFwd,     pls[L].woFwd_in);
            plr[L].ffnFused  = make_request(dk.ffnFused,  pls[L].ffnFused_in);
            plr[L].ffnBwdW2t = make_request(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in);
            plr[L].ffnBwdW13t= make_request(dk.ffnBwdW13t,pls[L].ffnBwdW13t_in);
            plr[L].wotBwd    = make_request(dk.wotBwd,    pls[L].wotBwd_in);
            plr[L].qBwd      = make_request(dk.qBwd,      pls[L].qBwd_in);
            plr[L].kvBwd     = make_request(dk.kvBwd,     pls[L].kvBwd_in);
        }

        // Stage weights into per-layer surfaces
        for (int L = 0; L < NLAYERS; L++) {
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
            stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
            stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
            stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
            stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
            stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
            stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
            stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
        }
        printf("Per-layer weight staging complete\n\n");

        // Work buffers
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*DIM*4);
        float *dk_buf = (float*)malloc(SEQ*KV_DIM*4);
        float *dv = (float*)malloc(SEQ*KV_DIM*4);
        float *da_buf = (float*)malloc(SEQ*DIM*4);
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*VOCAB*4);
        float *dlogits = (float*)malloc(SEQ*VOCAB*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        float *dsilu = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp2 = (float*)malloc(SEQ*HIDDEN*4);

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_train_ms=0;
        int total_steps_done=0;
        uint64_t t_wall_start = mach_absolute_time();

        srand48(42 + start_step);
        float res_alpha = 1.0f / sqrtf(2.0f * NLAYERS);

        // Zero gradient accumulators
        for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
        memset(grms_final, 0, DIM*4);
        memset(gembed, 0, (size_t)VOCAB*DIM*4);

        for (int step = start_step; step < total_steps; step++) {
            // Check wall-time budget
            double elapsed_s = (tb_ms(mach_absolute_time() - t_wall_start) + cum_wall) / 1000.0;
            if (elapsed_s + 10.0 > wall_time_budget) {
                printf("[wall-time budget nearly exhausted (%.0fs / %ds), stopping training]\n",
                       elapsed_s, wall_time_budget);
                break;
            }

            uint64_t t_step = mach_absolute_time();

            // Sample random position in TRAINING portion only
            size_t max_pos = train_tokens - SEQ - 1;
            size_t rpos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + rpos;
            uint16_t *target_tokens_raw = token_data + rpos + 1;

            // Embedding lookup
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            // ===== FORWARD =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                // CPU RMSNorm1
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);

                // Wait for pending dW cblas
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

                // SDPA forward (ANE): xnorm + Wq^T,Wk^T,Wv^T → attn_out,Q,K,V
                write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
                ane_eval_req(dk.sdpaFwd, plr[L].sdpaFwd);

                // Read SDPA output: [1, 2*DIM+2*KV_DIM, 1, SEQ]
                IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                int off = 0;
                cvt_f16_f32(ac->attn_out, fwd_out + off, DIM*SEQ);    off += DIM*SEQ;
                cvt_f16_f32(ac->Q,        fwd_out + off, DIM*SEQ);    off += DIM*SEQ;
                cvt_f16_f32(ac->K,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
                cvt_f16_f32(ac->V,        fwd_out + off, KV_DIM*SEQ);
                IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);

                // Wo forward (ANE): attn_out @ Wo^T → o_out
                write_wo_fwd_acts(pls[L].woFwd_in, ac->attn_out);
                ane_eval_req(dk.woFwd, plr[L].woFwd);
                io_read_dyn(dk.woFwd->ioOut, ac->o_out, DIM, SEQ);

                // CPU: scaled residual + RMSNorm2
                vDSP_vsma(ac->o_out, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                rmsnorm(ac->x2norm, ac->x2, lw[L].rms_ffn, DIM, SEQ);

                // Fused FFN (ANE): x2norm+x2+W1^T+W3^T+W2 → x_next,h1,h3,silu_out
                write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2);
                ane_eval_req(dk.ffnFused, plr[L].ffnFused);

                // Read fused output: [1, DIM+3*HIDDEN, 1, SEQ]
                IOSurfaceLock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(dk.ffnFused->ioOut);
                off = 0;
                cvt_f16_f32(x_cur,       ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
                cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->silu_out,ffn_out + off, HIDDEN*SEQ);
                IOSurfaceUnlock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
            }

            // Final RMSNorm + classifier + loss
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        VOCAB, SEQ, DIM, 1.0f, embed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
            logit_softcap(logits, VOCAB * SEQ, SOFTCAP);
            float loss = cross_entropy_loss(dlogits, logits, target_tokens_raw, VOCAB, SEQ);
            logit_softcap_bwd(dlogits, logits, VOCAB * SEQ, SOFTCAP);
            float ls = LOSS_SCALE;
            vDSP_vsmul(dlogits, 1, &ls, dlogits, 1, (vDSP_Length)(SEQ*VOCAB));
            last_loss = loss;

            // ===== BACKWARD =====
            // Classifier backward
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        DIM, SEQ, VOCAB, 1.0f, embed, DIM, dlogits, SEQ, 0.0f, dy, SEQ);

            // dEmbed async
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            VOCAB, DIM, SEQ, 1.0f, dlogits, SEQ, x_final, SEQ, 1.0f, gembed, DIM);
            });

            // Final RMSNorm backward
            float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
            rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
            memcpy(dy, dx_rms_final, SEQ*DIM*4);
            free(dx_rms_final);

            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];

                // dffn = alpha * dy
                vDSP_vsmul(dy, 1, &res_alpha, dffn, 1, (vDSP_Length)(SEQ*DIM));

                // FFN backward: dffn @ W2 → dsilu_raw (ANE)
                write_ffn_bwd_w2t_acts(pls[L].ffnBwdW2t_in, dffn);
                ane_eval_req(dk.ffnBwdW2t, plr[L].ffnBwdW2t);
                io_read_dyn(dk.ffnBwdW2t->ioOut, dsilu, HIDDEN, SEQ);

                // SiLU derivative (CPU, vectorized)
                {
                    int n = HIDDEN*SEQ;
                    float minus1 = -1.0f, one = 1.0f;
                    vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
                    vvexpf(silu_tmp, silu_tmp, &n);
                    vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
                    vvrecf(silu_tmp, silu_tmp, &n);  // sig = 1/(1+exp(-h1))
                    // dh3 = dsilu * silu(h1) = dsilu * h1 * sig
                    vDSP_vmul(ac->h1, 1, silu_tmp, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, (vDSP_Length)n);
                    // dh1 = dsilu * h3 * (sig + h1*sig*(1-sig))
                    vDSP_vsadd(silu_tmp, 1, &minus1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vneg(silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);  // 1-sig
                    vDSP_vmul(ac->h1, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);  // h1*(1-sig)
                    vDSP_vsadd(silu_tmp2, 1, &one, silu_tmp2, 1, (vDSP_Length)n);  // 1+h1*(1-sig)
                    vDSP_vmul(silu_tmp, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);  // sig*(1+h1*(1-sig))
                    vDSP_vmul(dsilu, 1, ac->h3, 1, dh1, 1, (vDSP_Length)n);
                    vDSP_vmul(dh1, 1, silu_tmp2, 1, dh1, 1, (vDSP_Length)n);
                }

                // dh1@W1 + dh3@W3 → dx_ffn (ANE)
                write_ffn_bwd_w13t_acts(pls[L].ffnBwdW13t_in, dh1, dh3);
                ane_eval_req(dk.ffnBwdW13t, plr[L].ffnBwdW13t);
                io_read_dyn(dk.ffnBwdW13t->ioOut, dx_ffn, DIM, SEQ);

                // dW FFN async
                float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
                });

                // RMSNorm2 backward
                memset(dx2, 0, SEQ*DIM*4);
                rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];

                // Wo^T backward (ANE): alpha*dx2 @ Wo → da
                float *dx2_scaled = (float*)malloc(SEQ*DIM*4);
                vDSP_vsmul(dx2, 1, &res_alpha, dx2_scaled, 1, (vDSP_Length)(SEQ*DIM));
                write_wot_bwd_acts(pls[L].wotBwd_in, dx2_scaled);
                ane_eval_req(dk.wotBwd, plr[L].wotBwd);
                io_read_dyn(dk.wotBwd->ioOut, da_buf, DIM, SEQ);

                // dWo async
                float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, dx2_scaled, SEQ*DIM*4);
                free(dx2_scaled);
                float *capt_attn = (float*)malloc(SEQ*DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, DIM);
                    free(capt_do); free(capt_attn);
                });

                // SDPA backward part 1: Q,K,V,da → dV,probs,dp
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 0,              ac->Q,   DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, DIM,            ac->K,   KV_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, DIM+KV_DIM,     ac->V,   KV_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, DIM+2*KV_DIM,   da_buf,  DIM, SEQ);
                ane_eval(dk.sdpaBwd1);

                // SDPA backward part 2: probs,dp,Q,K → dQ,dK
                io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut, KV_DIM, 2*SCORE_CH, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH,     ac->Q, DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH+DIM, ac->K, KV_DIM, SEQ);
                ane_eval(dk.sdpaBwd2);

                // Read SDPA backward outputs
                io_read_fp16(dk.sdpaBwd2->ioOut, dq, 0,   DIM, SEQ);
                io_read_fp16(dk.sdpaBwd2->ioOut, dk_buf, DIM, KV_DIM, SEQ);
                io_read_fp16(dk.sdpaBwd1->ioOut, dv, 0,    KV_DIM, SEQ);

                // dWq/dWk/dWv async
                float *capt_dq = (float*)malloc(SEQ*DIM*4); memcpy(capt_dq, dq, SEQ*DIM*4);
                float *capt_dk = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dk, dk_buf, SEQ*KV_DIM*4);
                float *capt_dv = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dv, dv, SEQ*KV_DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                });

                // Q backward (ANE): dq @ Wq → dx_q
                write_q_bwd_acts(pls[L].qBwd_in, dq);
                ane_eval_req(dk.qBwd, plr[L].qBwd);
                io_read_dyn(dk.qBwd->ioOut, dx_attn, DIM, SEQ);

                // KV backward (ANE): dk@Wk + dv@Wv → dx_kv
                float *dx_kv = (float*)malloc(SEQ*DIM*4);
                write_kv_bwd_acts(pls[L].kvBwd_in, dk_buf, dv);
                ane_eval_req(dk.kvBwd, plr[L].kvBwd);
                io_read_dyn(dk.kvBwd->ioOut, dx_kv, DIM, SEQ);

                // dx_attn += dx_kv
                for(int i=0; i<SEQ*DIM; i++) dx_attn[i] += dx_kv[i];
                free(dx_kv);

                // RMSNorm1 backward
                float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
                free(dx_rms1);
            }

            // Embedding backward (full vocab, accumulates into gembed)
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            if (step % 10 == 0 || step == start_step)
                printf("step %-4d loss=%.4f  %.1fms/step\n", step, loss, step_ms);

            fprintf(stderr, "{\"type\":\"step\",\"step\":%d,\"loss\":%.6f}\n", step, loss);

            // Adam update every ACCUM_STEPS
            if ((step+1) % ACCUM_STEPS == 0 || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

                // Gradient averaging (undo loss scaling)
                float gsc = 1.0f / ((float)ACCUM_STEPS * LOSS_SCALE);
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    for(size_t i=0;i<WQ_SZ;i++){g->Wq[i]*=gsc; g->Wo[i]*=gsc;}
                    for(size_t i=0;i<WK_SZ;i++) g->Wk[i]*=gsc;
                    for(size_t i=0;i<WV_SZ;i++) g->Wv[i]*=gsc;
                    for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                    for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                    for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                    for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
                }
                for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;

                // Gradient clipping
                float gnorm = clip_gradients(grads, grms_final, gembed, GRAD_CLIP_MAX);
                if (gnorm > GRAD_CLIP_MAX)
                    printf("  [grad clip: %.2f → %.2f]\n", gnorm, GRAD_CLIP_MAX);

                // Cosine LR schedule with linear warmup
                float scheduled_lr;
                if (adam_t < LR_WARMUP_STEPS) {
                    scheduled_lr = lr * ((float)(adam_t + 1)) / LR_WARMUP_STEPS;
                } else {
                    float decay_ratio = (float)(adam_t - LR_WARMUP_STEPS) / (float)(total_steps - LR_WARMUP_STEPS);
                    if (decay_ratio > 1.0f) decay_ratio = 1.0f;
                    float min_lr = lr * LR_MIN_FRAC;
                    scheduled_lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (lr - min_lr);
                }

                // Adam update with differential LR
                adam_t++;
                float wd = WEIGHT_DECAY;
                float mlr = scheduled_lr * MATRIX_LR_SCALE;
                float elr = scheduled_lr * EMBED_LR_SCALE;
                // Parallel Adam update + transpose + restage across layers
                // (pointer indirection needed for block capture of C arrays)
                LayerWeights *p_lw = lw; LayerAdam *p_la = la;
                LayerGrads *p_gr = grads; PerLayerSurfaces *p_pls = pls;
                float **p_Wqt = Wqt_buf, **p_Wkt = Wkt_buf, **p_Wvt = Wvt_buf, **p_Wot = Wot_buf;
                float **p_W1t = W1t_buf, **p_W3t = W3t_buf;
                dispatch_queue_t par_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
                dispatch_apply(NLAYERS, par_q, ^(size_t Li) {
                    int L = (int)Li;
                    LayerGrads *g = &p_gr[L];
                    adam_update(p_lw[L].Wq, g->Wq, &p_la[L].Wq, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(p_lw[L].Wk, g->Wk, &p_la[L].Wk, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(p_lw[L].Wv, g->Wv, &p_la[L].Wv, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(p_lw[L].Wo, g->Wo, &p_la[L].Wo, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(p_lw[L].W1, g->W1, &p_la[L].W1, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(p_lw[L].W2, g->W2, &p_la[L].W2, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(p_lw[L].W3, g->W3, &p_la[L].W3, adam_t, mlr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(p_lw[L].rms_att, g->rms_att, &p_la[L].rms_att, adam_t, scheduled_lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(p_lw[L].rms_ffn, g->rms_ffn, &p_la[L].rms_ffn, adam_t, scheduled_lr, adam_b1, adam_b2, adam_eps, 0.0f);

                    // Update transposed weight buffers and re-stage
                    transpose_weight(p_Wqt[L], p_lw[L].Wq, DIM, DIM);
                    transpose_weight(p_Wkt[L], p_lw[L].Wk, KV_DIM, DIM);
                    transpose_weight(p_Wvt[L], p_lw[L].Wv, KV_DIM, DIM);
                    transpose_weight(p_Wot[L], p_lw[L].Wo, DIM, DIM);
                    transpose_weight(p_W1t[L], p_lw[L].W1, HIDDEN, DIM);
                    transpose_weight(p_W3t[L], p_lw[L].W3, HIDDEN, DIM);

                    stage_sdpa_fwd_weights(p_pls[L].sdpaFwd_in, p_Wqt[L], p_Wkt[L], p_Wvt[L]);
                    stage_wo_fwd_weights(p_pls[L].woFwd_in, p_Wot[L]);
                    stage_ffn_fused_weights(p_pls[L].ffnFused_in, p_W1t[L], p_W3t[L], p_lw[L].W2);
                    stage_ffn_bwd_w2t_weights(p_pls[L].ffnBwdW2t_in, p_lw[L].W2);
                    stage_ffn_bwd_w13t_weights(p_pls[L].ffnBwdW13t_in, p_lw[L].W1, p_lw[L].W3);
                    stage_wot_bwd_weights(p_pls[L].wotBwd_in, p_lw[L].Wo);
                    stage_q_bwd_weights(p_pls[L].qBwd_in, p_lw[L].Wq);
                    stage_kv_bwd_weights(p_pls[L].kvBwd_in, p_lw[L].Wk, p_lw[L].Wv);

                    layer_grads_zero(&p_gr[L]);
                });
                adam_update(rms_final, grms_final, &arms_final, adam_t, scheduled_lr, adam_b1, adam_b2, adam_eps, 0.0f);
                adam_update(embed, gembed, &aembed, adam_t, elr, adam_b1, adam_b2, adam_eps, 0.0f);

                // Zero grads (layers done inside dispatch_apply above)
                memset(grms_final, 0, DIM*4);
                memset(gembed, 0, (size_t)VOCAB*DIM*4);
            }
        }

        // ===== VALIDATION (forward-only, uses staged dynamic kernels — no recompile needed!) =====
        printf("\n=== Validation ===\n");
        float val_loss = -1.0f;
        if (val_tokens > (size_t)(SEQ + 1)) {
            size_t val_stride = SEQ + 1;
            int n_windows = 0;
            float total_val_loss = 0;
            size_t vpos = val_start;

            for (int w = 0; w < VAL_WINDOWS && vpos + val_stride <= n_tokens; w++, vpos += val_stride) {
                uint16_t *vinput = token_data + vpos;
                uint16_t *vtarget_raw = token_data + vpos + 1;
                embed_lookup(x_cur, embed, vinput, DIM, SEQ);
                for (int L=0; L<NLAYERS; L++) {
                    rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                    write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
                    ane_eval_req(dk.sdpaFwd, plr[L].sdpaFwd);
                    float *attn_tmp = (float*)malloc(SEQ*DIM*4);
                    io_read_fp16(dk.sdpaFwd->ioOut, attn_tmp, 0, DIM, SEQ);
                    write_wo_fwd_acts(pls[L].woFwd_in, attn_tmp);
                    ane_eval_req(dk.woFwd, plr[L].woFwd);
                    float *o_tmp = (float*)malloc(SEQ*DIM*4);
                    io_read_dyn(dk.woFwd->ioOut, o_tmp, DIM, SEQ);
                    float *x2_tmp = (float*)malloc(SEQ*DIM*4);
                    vDSP_vsma(o_tmp, 1, &res_alpha, x_cur, 1, x2_tmp, 1, (vDSP_Length)(SEQ*DIM));
                    float *x2norm_tmp = (float*)malloc(SEQ*DIM*4);
                    rmsnorm(x2norm_tmp, x2_tmp, lw[L].rms_ffn, DIM, SEQ);
                    write_ffn_fused_acts(pls[L].ffnFused_in, x2norm_tmp, x2_tmp);
                    ane_eval_req(dk.ffnFused, plr[L].ffnFused);
                    io_read_fp16(dk.ffnFused->ioOut, x_cur, 0, DIM, SEQ);
                    free(attn_tmp); free(o_tmp); free(x2_tmp); free(x2norm_tmp);
                }
                rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            VOCAB, SEQ, DIM, 1.0f, embed, DIM, x_final, SEQ, 0.0f, logits, SEQ);
                float wloss = cross_entropy_loss(dlogits, logits, vtarget_raw, VOCAB, SEQ);
                total_val_loss += wloss;
                n_windows++;
            }
            if (n_windows > 0) {
                val_loss = total_val_loss / n_windows;
                printf("val_loss: %.6f (%d windows)\n", val_loss, n_windows);
            }
        }

        // Efficiency report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        total_train_ms += cum_train;
        wall += cum_wall; total_steps_done += cum_steps;
        double fwd_flops = NLAYERS * (4.0*DIM*(DIM+KV_DIM)*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ);
        double sdpa_flops = NLAYERS * 2.0*HEADS*5*SEQ*SEQ*HD;
        double ane_flops = (fwd_flops*2 + sdpa_flops) * total_steps_done;
        double ane_tflops = (total_train_ms > 0) ? ane_flops / (total_train_ms * 1e9) : 0;
        double ane_util = (total_train_ms > 0) ? 100.0*ane_tflops/15.8 : 0;
        double ms_per_step = (total_steps_done > 0) ? total_train_ms/total_steps_done : 0;

        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Wall time:       %.0f ms (%.1f s)\n", wall, wall/1000);
        printf("Compile time:    %.0f ms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/wall);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100*total_train_ms/wall);
        printf("Avg train:       %.1f ms/step\n", ms_per_step);
        printf("ANE TFLOPS:      %.2f sustained\n", ane_tflops);
        printf("ANE utilization: %.1f%% of 15.8 TFLOPS\n", ane_util);

        // ===== JSON output =====
        printf("\n###JSON###\n");
        printf("{\"status\":\"ok\",\"val_loss\":%.6f,\"train_loss\":%.6f,"
               "\"steps\":%d,\"ms_per_step\":%.2f,\"wall_time_s\":%.1f,"
               "\"compile_time_s\":%.1f,\"ane_util_pct\":%.2f}\n",
               val_loss, last_loss, total_steps_done, ms_per_step, wall/1000,
               compile_ms/1000, ane_util);

        // Save final checkpoint
        save_checkpoint(ckpt_path, total_steps_done + start_step, total_steps, lr, last_loss,
            total_train_ms, wall, total_steps_done, adam_t,
            lw, la, rms_final, &arms_final, embed, &aembed);

        // Cleanup
        free_per_layer(pls, plr);
        free_kern(dk.sdpaFwd); free_kern(dk.woFwd); free_kern(dk.ffnFused);
        free_kern(dk.ffnBwdW2t); free_kern(dk.ffnBwdW13t); free_kern(dk.wotBwd);
        free_kern(dk.sdpaBwd1); free_kern(dk.sdpaBwd2);
        free_kern(dk.qBwd); free_kern(dk.kvBwd);
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W3t_buf[L]);
        }
        munmap(token_data, data_len); close(data_fd);
        free(rms_final); free(embed); free(grms_final); free(gembed);
        adam_free(&arms_final); adam_free(&aembed);
        free(dy); free(dffn); free(dx_ffn); free(dx2); free(dx_attn);
        free(dq); free(dk_buf); free(dv); free(da_buf);
        free(x_cur); free(x_final); free(xnorm_buf);
        free(logits); free(dlogits);
        free(dh1); free(dh3); free(dsilu); free(silu_tmp); free(silu_tmp2);
    }
    return 0;
}
