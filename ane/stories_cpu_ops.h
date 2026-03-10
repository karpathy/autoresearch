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

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
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
