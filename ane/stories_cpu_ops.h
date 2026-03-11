// stories_cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, softmax
#pragma once
#include "stories_config.h"


static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    float *rms_tmp = (float*)malloc(S * sizeof(float));
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
    free(ss); free(rms_tmp);
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    float *rms_tmp = (float*)malloc(S * sizeof(float));
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = (float*)malloc(S*4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(rms_tmp, 1, dy+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(rms_tmp, 1, rrms, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(rms_tmp, 1, rrms, 1, rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
    free(ss); free(rrms); free(dot); free(rms_tmp);
}

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps, float wd) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    float inv_bc1 = 1.0f / bc1, inv_bc2 = 1.0f / bc2;
    float one_minus_b1 = 1.0f - b1, one_minus_b2 = 1.0f - b2;
    vDSP_Length n = (vDSP_Length)s->n;

    // Decoupled weight decay (AdamW): w -= wd * lr * w
    if (wd > 0) {
        float neg_wd_lr = -wd * lr;
        vDSP_vsma(w, 1, &neg_wd_lr, w, 1, w, 1, n);
    }

    // m = b1*m + (1-b1)*g
    vDSP_vsmul(s->m, 1, &b1, s->m, 1, n);
    vDSP_vsma(g, 1, &one_minus_b1, s->m, 1, s->m, 1, n);

    // v = b2*v + (1-b2)*g*g
    vDSP_vsmul(s->v, 1, &b2, s->v, 1, n);
    float *g_sq = (float*)malloc(s->n * 4);
    vDSP_vmul(g, 1, g, 1, g_sq, 1, n);
    vDSP_vsma(g_sq, 1, &one_minus_b2, s->v, 1, s->v, 1, n);

    // mhat = m / bc1, vhat = v / bc2
    float *mhat = (float*)malloc(s->n * 4);
    float *vhat = (float*)malloc(s->n * 4);
    vDSP_vsmul(s->m, 1, &inv_bc1, mhat, 1, n);
    vDSP_vsmul(s->v, 1, &inv_bc2, vhat, 1, n);

    // w -= lr * mhat / (sqrt(vhat) + eps)
    int ni = (int)s->n;
    vvsqrtf(vhat, vhat, &ni);
    vDSP_vsadd(vhat, 1, &eps, vhat, 1, n);
    vDSP_vdiv(vhat, 1, mhat, 1, mhat, 1, n);  // mhat = mhat / (sqrt(vhat) + eps)
    float neg_lr = -lr;
    vDSP_vsma(mhat, 1, &neg_lr, w, 1, w, 1, n);

    free(g_sq); free(mhat); free(vhat);
}

// Logit softcapping: cap * tanh(logits / cap), clamps logits to [-cap, cap]
static void logit_softcap(float *logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    vDSP_vsmul(logits, 1, &inv_cap, logits, 1, (vDSP_Length)n);
    int ni = n;
    vvtanhf(logits, logits, &ni);
    vDSP_vsmul(logits, 1, &cap, logits, 1, (vDSP_Length)n);
}

// Logit softcap backward: dlogits *= 1 - tanh²(x/cap) = 1 - (capped/cap)²
static void logit_softcap_bwd(float *dlogits, const float *capped_logits, int n, float cap) {
    float inv_cap = 1.0f / cap;
    float *tmp = (float*)malloc(n * sizeof(float));
    vDSP_vsmul(capped_logits, 1, &inv_cap, tmp, 1, (vDSP_Length)n);  // tanh values
    vDSP_vmul(tmp, 1, tmp, 1, tmp, 1, (vDSP_Length)n);  // tanh²
    float neg_one = -1.0f, one = 1.0f;
    vDSP_vsmsa(tmp, 1, &neg_one, &one, tmp, 1, (vDSP_Length)n);  // 1 - tanh²
    vDSP_vmul(dlogits, 1, tmp, 1, dlogits, 1, (vDSP_Length)n);
    free(tmp);
}

// Cross-entropy loss + gradient for logits (column-major: [VOCAB, SEQ])
// logits[v*SEQ+t] = logit for vocab v, position t
// targets[t] = target token id for position t
// Returns mean CE loss, writes dlogits = softmax(logits) - one_hot(targets)
// Data is column-major [V, S], but we process per-column (stride=1 within col is v*S+t, stride between v's is S)
// For vDSP: transpose to row-major scratch [S, V] to vectorize softmax per position
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    // Work in transposed layout [S, V] where each row is one position's logits (contiguous)
    float *buf = (float*)malloc(S * V * 4);
    // Transpose [V,S] → [S,V]: buf[t*V+v] = logits[v*S+t]
    vDSP_mtrans(logits, 1, buf, 1, (vDSP_Length)S, (vDSP_Length)V);

    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        float *row = buf + t * V;
        // max
        float maxv;
        vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        // row -= maxv
        float neg_max = -maxv;
        vDSP_vsadd(row, 1, &neg_max, row, 1, (vDSP_Length)V);
        // exp in-place
        int n = V;
        vvexpf(row, row, &n);
        // sum
        float sum;
        vDSP_sve(row, 1, &sum, (vDSP_Length)V);
        // normalize
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(row, 1, &inv_sum, row, 1, (vDSP_Length)V);
        // loss
        int tgt = targets[t];
        if (tgt < 0 || tgt >= V) { fprintf(stderr, "WARN: target token %d out of vocab range [0,%d), skipping\n", tgt, V); continue; }
        total_loss -= logf(row[tgt] + 1e-10f);
        // gradient: softmax - one_hot, then /S
        row[tgt] -= 1.0f;
        vDSP_vsmul(row, 1, &invS, row, 1, (vDSP_Length)V);
    }
    // Transpose back [S,V] → [V,S]
    vDSP_mtrans(buf, 1, dlogits, 1, (vDSP_Length)V, (vDSP_Length)S);
    free(buf);
    return total_loss / S;
}

// Vocab compaction: map full 32K vocab to ~1900 active tokens
typedef struct {
    int compact_vocab;          // number of active tokens
    int *full_to_compact;       // [VOCAB] → compact id (-1 if unused)
    int *compact_to_full;       // [compact_vocab] → full vocab id
} VocabMap;

static VocabMap vocab_map_build(const uint16_t *data, size_t n_tokens, int full_vocab) {
    VocabMap vm;
    vm.full_to_compact = (int*)malloc(full_vocab * sizeof(int));
    memset(vm.full_to_compact, -1, full_vocab * sizeof(int));
    for (size_t i = 0; i < n_tokens; i++)
        vm.full_to_compact[data[i]] = 0;
    int cid = 0;
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] == 0)
            vm.full_to_compact[v] = cid++;
        else
            vm.full_to_compact[v] = -1;
    }
    vm.compact_vocab = cid;
    vm.compact_to_full = (int*)malloc(cid * sizeof(int));
    for (int v = 0; v < full_vocab; v++)
        if (vm.full_to_compact[v] >= 0)
            vm.compact_to_full[vm.full_to_compact[v]] = v;
    return vm;
}

static float *vocab_compact_embed(const float *full_embed, const VocabMap *vm, int dim) {
    float *ce = (float*)malloc((size_t)vm->compact_vocab * dim * 4);
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(ce + c*dim, full_embed + vm->compact_to_full[c]*dim, dim*4);
    return ce;
}

static void vocab_scatter_grads(float *full_gembed, const float *compact_gembed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++) {
        int fv = vm->compact_to_full[c];
        for (int d = 0; d < dim; d++)
            full_gembed[fv*dim + d] += compact_gembed[c*dim + d];
    }
}

static void vocab_update_full(float *full_embed, const float *compact_embed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(full_embed + vm->compact_to_full[c]*dim, compact_embed + c*dim, dim*4);
}

// Embedding lookup: token_ids → x [DIM, SEQ] (channel-first)
// embed is [VOCAB, DIM] row-major (vocab_size rows, dim cols)
static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= VOCAB) { fprintf(stderr, "WARN: token %d out of range [0,%d)\n", tok, VOCAB); continue; }
        for (int d = 0; d < dim; d++) {
            x[d*seq + t] = embed[tok*dim + d];
        }
    }
}

// Gradient clipping: global L2 norm across all gradient buffers
// Returns the pre-clip norm for logging
static float clip_gradients(LayerGrads *grads, float *grms_final, float *gembed, float max_norm) {
    double norm_sq = 0;
    for (int L = 0; L < NLAYERS; L++) {
        float dot;
        vDSP_dotpr(grads[L].Wq, 1, grads[L].Wq, 1, &dot, (vDSP_Length)WQ_SZ); norm_sq += dot;
        vDSP_dotpr(grads[L].Wk, 1, grads[L].Wk, 1, &dot, (vDSP_Length)WK_SZ); norm_sq += dot;
        vDSP_dotpr(grads[L].Wv, 1, grads[L].Wv, 1, &dot, (vDSP_Length)WV_SZ); norm_sq += dot;
        vDSP_dotpr(grads[L].Wo, 1, grads[L].Wo, 1, &dot, (vDSP_Length)WO_SZ); norm_sq += dot;
        vDSP_dotpr(grads[L].W1, 1, grads[L].W1, 1, &dot, (vDSP_Length)W1_SZ); norm_sq += dot;
        vDSP_dotpr(grads[L].W2, 1, grads[L].W2, 1, &dot, (vDSP_Length)W2_SZ); norm_sq += dot;
        vDSP_dotpr(grads[L].W3, 1, grads[L].W3, 1, &dot, (vDSP_Length)W3_SZ); norm_sq += dot;
        vDSP_dotpr(grads[L].rms_att, 1, grads[L].rms_att, 1, &dot, (vDSP_Length)DIM); norm_sq += dot;
        vDSP_dotpr(grads[L].rms_ffn, 1, grads[L].rms_ffn, 1, &dot, (vDSP_Length)DIM); norm_sq += dot;
    }
    float dot;
    vDSP_dotpr(grms_final, 1, grms_final, 1, &dot, (vDSP_Length)DIM); norm_sq += dot;
    vDSP_dotpr(gembed, 1, gembed, 1, &dot, (vDSP_Length)((size_t)VOCAB*DIM)); norm_sq += dot;

    float total_norm = sqrtf((float)norm_sq);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (int L = 0; L < NLAYERS; L++) {
            vDSP_vsmul(grads[L].Wq, 1, &scale, grads[L].Wq, 1, (vDSP_Length)WQ_SZ);
            vDSP_vsmul(grads[L].Wk, 1, &scale, grads[L].Wk, 1, (vDSP_Length)WK_SZ);
            vDSP_vsmul(grads[L].Wv, 1, &scale, grads[L].Wv, 1, (vDSP_Length)WV_SZ);
            vDSP_vsmul(grads[L].Wo, 1, &scale, grads[L].Wo, 1, (vDSP_Length)WO_SZ);
            vDSP_vsmul(grads[L].W1, 1, &scale, grads[L].W1, 1, (vDSP_Length)W1_SZ);
            vDSP_vsmul(grads[L].W2, 1, &scale, grads[L].W2, 1, (vDSP_Length)W2_SZ);
            vDSP_vsmul(grads[L].W3, 1, &scale, grads[L].W3, 1, (vDSP_Length)W3_SZ);
            vDSP_vsmul(grads[L].rms_att, 1, &scale, grads[L].rms_att, 1, (vDSP_Length)DIM);
            vDSP_vsmul(grads[L].rms_ffn, 1, &scale, grads[L].rms_ffn, 1, (vDSP_Length)DIM);
        }
        vDSP_vsmul(grms_final, 1, &scale, grms_final, 1, (vDSP_Length)DIM);
        vDSP_vsmul(gembed, 1, &scale, gembed, 1, (vDSP_Length)((size_t)VOCAB*DIM));
    }
    return total_norm;
}

// Embedding backward: accumulate dE[tok] += dx[:,t] for each position
static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= VOCAB) { continue; }
        for (int d = 0; d < dim; d++) {
            d_embed[tok*dim + d] += dx[d*seq + t];
        }
    }
}
