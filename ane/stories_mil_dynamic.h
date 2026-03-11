// stories_mil_dynamic.h — Dynamic weight MIL generators for ANE kernels
// Weights passed via IOSurface spatial dimension, not baked as BLOBFILEs.
// Kernels compiled ONCE at startup, weights updated via IOSurface writes.
// Adapted from fiale-plus/ANE training_dynamic for MHA (no GQA, no RoPE).
#pragma once
#include "stories_io.h"

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// SP macros defined in stories_io.h (included above)

// ===== Helper: simple dynamic matmul kernel =====
// y = x @ W where input is [1, IC, 1, SEQ+OC] with act at sp=0, weight at sp=SEQ
// Output: [1, OC, 1, SEQ]
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq) {
    int sp = seq + oc;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
    // Slice activation [1,IC,1,SEQ]
    [m appendString:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];
    // Slice weight [1,IC,1,OC]
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];
    // Reshape + transpose for matmul
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];
    // matmul: [1,1,SEQ,IC] @ [1,1,IC,OC] → [1,1,SEQ,OC]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== SDPA forward (dynamic, no RoPE, GQA-capable) =====
// Input: [1, DIM, 1, SEQ+DIM+2*KV_DIM] — xnorm + Wq^T + Wk^T + Wv^T
// Output: [1, 2*DIM+2*KV_DIM, 1, SEQ] — concat(attn_out, Q, K, V)
// Baked weight: causal mask only
// GQA: when N_KV_HEADS < HEADS, K/V are tiled to match Q head count
static NSString *gen_sdpa_fwd_dynamic(void) {
    float sc = 1.0f/sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SDPA_FWD_SP];

    // Slice xnorm [1,DIM,1,SEQ]
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = slice_by_size(x=x,begin=bx,size=sx)[name=string(\"xn\")];\n", DIM, SEQ];

    // Slice Wq^T [1,DIM,1,DIM]
    [m appendFormat:@"        tensor<int32, [4]> bq = const()[name=string(\"bq\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<int32, [4]> swd = const()[name=string(\"swd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wq = slice_by_size(x=x,begin=bq,size=swd)[name=string(\"Wq\")];\n", DIM, DIM];
    // Slice Wk^T [1,DIM,1,KV_DIM]
    [m appendFormat:@"        tensor<int32, [4]> bk = const()[name=string(\"bk\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+DIM];
    [m appendFormat:@"        tensor<int32, [4]> swk = const()[name=string(\"swk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wk = slice_by_size(x=x,begin=bk,size=swk)[name=string(\"Wk\")];\n", DIM, KV_DIM];
    // Slice Wv^T [1,DIM,1,KV_DIM]
    [m appendFormat:@"        tensor<int32, [4]> bv = const()[name=string(\"bv\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+DIM+KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wv = slice_by_size(x=x,begin=bv,size=swk)[name=string(\"Wv\")];\n", DIM, KV_DIM];

    // Reshape xnorm for matmul: [1,DIM,1,SEQ] → [1,1,DIM,SEQ] → [1,1,SEQ,DIM]
    [m appendFormat:@"        tensor<int32, [4]> r2 = const()[name=string(\"r2\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xn2 = reshape(shape=r2,x=xn)[name=string(\"xn2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n", SEQ, DIM];
    // Reshape weights
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wq2 = reshape(shape=rw,x=Wq)[name=string(\"Wq2\")];\n", DIM, DIM];
    [m appendFormat:@"        tensor<int32, [4]> rwk = const()[name=string(\"rwk\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wk2 = reshape(shape=rwk,x=Wk)[name=string(\"Wk2\")];\n", DIM, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wv2 = reshape(shape=rwk,x=Wv)[name=string(\"Wv2\")];\n", DIM, KV_DIM];

    // Q matmul: [1,1,SEQ,DIM] @ [1,1,DIM,DIM] → [1,1,SEQ,DIM]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];\n", SEQ, DIM];
    // KV matmul: [1,1,SEQ,DIM] @ [1,1,DIM,KV_DIM] → [1,1,SEQ,KV_DIM]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];\n", SEQ, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];\n", SEQ, KV_DIM];

    // Transpose Q back: [1,1,SEQ,DIM] → [1,1,DIM,SEQ] → reshape [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qt = transpose(perm=pm,x=qm)[name=string(\"qt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dsh = const()[name=string(\"dsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = reshape(shape=dsh,x=qt)[name=string(\"qf\")];\n", DIM, SEQ];
    // Transpose KV back: [1,1,SEQ,KV_DIM] → [1,1,KV_DIM,SEQ] → reshape [1,KV_DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> kt = transpose(perm=pm,x=km)[name=string(\"kt\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vt = transpose(perm=pm,x=vm)[name=string(\"vt\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = reshape(shape=kvsh,x=kt)[name=string(\"kf\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = reshape(shape=kvsh,x=vt)[name=string(\"vf\")];\n", KV_DIM, SEQ];

    // Reshape Q to heads: [1,DIM,1,SEQ] → [1,HEADS,HD,SEQ] → transpose → [1,HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> hsh = const()[name=string(\"hsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=hsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    // Reshape KV to heads: [1,KV_DIM,1,SEQ] → [1,N_KV_HEADS,HD,SEQ] → transpose → [1,N_KV_HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> kvhsh = const()[name=string(\"kvhsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=kvhsh,x=kf)[name=string(\"rk\")];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k0 = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", N_KV_HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=kvhsh,x=vf)[name=string(\"rv\")];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v0 = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", N_KV_HEADS, SEQ, HD];

    // GQA expansion: tile KV heads to match Q head count
    if (HEADS_PER_KV > 1) {
        // [1,N_KV_HEADS,SEQ,HD] → [1,N_KV_HEADS,1,SEQ*HD] → tile → [1,N_KV_HEADS,HEADS_PER_KV,SEQ*HD] → [1,HEADS,SEQ,HD]
        [m appendFormat:@"        tensor<int32, [4]> gqa_fl = const()[name=string(\"gqafl\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> k_fl = reshape(shape=gqa_fl,x=k0)[name=string(\"kfl\")];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> v_fl = reshape(shape=gqa_fl,x=v0)[name=string(\"vfl\")];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<int32, [4]> gqa_rp = const()[name=string(\"gqarp\"), val=tensor<int32, [4]>([1,1,%d,1])];\n", HEADS_PER_KV];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_tl = tile(reps=gqa_rp,x=k_fl)[name=string(\"ktl\")];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v_tl = tile(reps=gqa_rp,x=v_fl)[name=string(\"vtl\")];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = reshape(shape=hsh,x=k_tl)[name=string(\"kex\")];\n", HEADS, SEQ, HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = reshape(shape=hsh,x=v_tl)[name=string(\"vex\")];\n", HEADS, SEQ, HD];
    } else {
        // MHA: k0/v0 already at HEADS
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = identity(x=k0)[name=string(\"kex\")];\n", HEADS, SEQ, HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = identity(x=v0)[name=string(\"vex\")];\n", HEADS, SEQ, HD];
    }

    // Scaled dot-product attention
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS, SEQ, HD];

    // Reshape attn_out back to flat [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=dsh,x=at)[name=string(\"ra\")];\n", DIM, SEQ];

    // Output: concat(attn_out, Q, K, V) — K/V at KV_DIM for GQA
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(af,qf,kf,vf))[name=string(\"cat\")];\n", 2*DIM+2*KV_DIM, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// woFwd: attn_out @ Wo^T → o_out (IC=DIM, OC=DIM)
static NSString *gen_wo_fwd_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, DIM, SEQ);
}

// ===== Fused FFN forward: W1,W3 + SiLU + W2 + residual =====
// Input: [1, DIM, 1, 2*SEQ+3*HIDDEN] — x2norm + x2 + W1^T + W3^T + W2
// Output: [1, DIM+3*HIDDEN, 1, SEQ] — concat(x_next, h1, h3, silu_out)
static NSString *gen_ffn_fused_dynamic(void) {
    int sp_in = FFN_FUSED_SP;
    int out_ch = DIM + 3*HIDDEN;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, sp_in];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];

    // Slice x2norm, x2, W1^T, W3^T, W2
    [m appendString:@"        tensor<int32, [4]> b_xn = const()[name=string(\"b_xn\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> s_ds = const()[name=string(\"s_ds\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2norm = slice_by_size(x=x,begin=b_xn,size=s_ds)[name=string(\"x2norm\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b_x2 = const()[name=string(\"b_x2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x2 = slice_by_size(x=x,begin=b_x2,size=s_ds)[name=string(\"x2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b_w1 = const()[name=string(\"b_w1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<int32, [4]> s_wh = const()[name=string(\"s_wh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W1 = slice_by_size(x=x,begin=b_w1,size=s_wh)[name=string(\"W1\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<int32, [4]> b_w3 = const()[name=string(\"b_w3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W3 = slice_by_size(x=x,begin=b_w3,size=s_wh)[name=string(\"W3\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<int32, [4]> b_w2 = const()[name=string(\"b_w2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+2*HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W2r = slice_by_size(x=x,begin=b_w2,size=s_wh)[name=string(\"W2r\")];\n", DIM, HIDDEN];

    // xnorm matmul: [SEQ,DIM] @ [DIM,HIDDEN] → [SEQ,HIDDEN] for h1,h3
    [m appendFormat:@"        tensor<int32, [4]> rd = const()[name=string(\"rd\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xn2 = reshape(shape=rd,x=x2norm)[name=string(\"xn2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> xnt = transpose(perm=pm,x=xn2)[name=string(\"xnt\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W12 = reshape(shape=rw,x=W1)[name=string(\"W12\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W32 = reshape(shape=rw,x=W3)[name=string(\"W32\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)[name=string(\"h1m\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)[name=string(\"h3m\")];\n", SEQ, HIDDEN];

    // Reshape back to [1,HIDDEN,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h1t = transpose(perm=pm,x=h1m)[name=string(\"h1t\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> h3t = transpose(perm=pm,x=h3m)[name=string(\"h3t\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rh = const()[name=string(\"rh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = reshape(shape=rh,x=h1t)[name=string(\"h1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = reshape(shape=rh,x=h3t)[name=string(\"h3\")];\n", HIDDEN, SEQ];

    // SiLU + gate
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN, SEQ];

    // gate @ W2: W2 is [DIM, HIDDEN] stored as-is, transpose inside kernel
    [m appendFormat:@"        tensor<int32, [4]> rg = const()[name=string(\"rg\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> g2 = reshape(shape=rg,x=gate)[name=string(\"g2\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> gt = transpose(perm=pm,x=g2)[name=string(\"gtt\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W22 = reshape(shape=rw,x=W2r)[name=string(\"W22\")];\n", DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W2t = transpose(perm=pm,x=W22)[name=string(\"W2t\")];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> fm = matmul(transpose_x=bF,transpose_y=bF,x=gt,y=W2t)[name=string(\"fm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> ft = transpose(perm=pm,x=fm)[name=string(\"ft\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rd2 = const()[name=string(\"rd2\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ffn_out = reshape(shape=rd2,x=ft)[name=string(\"ffn_out\")];\n", DIM, SEQ];

    // Residual: x_next = x2 + alpha * ffn_out
    float alpha = 1.0f / sqrtf(2.0f * NLAYERS);
    [m appendFormat:@"        fp16 res_alpha = const()[name=string(\"res_alpha\"), val=fp16(%g)];\n", alpha];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ffn_scaled = mul(x=ffn_out,y=res_alpha)[name=string(\"ffn_sc\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> x_next = add(x=x2,y=ffn_scaled)[name=string(\"x_next\")];\n", DIM, SEQ];

    // Output: concat(x_next, h1, h3, silu_out=gate)
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(x_next,h1,h3,gate))[name=string(\"cat\")];\n", out_ch, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ffnBwdW2t: dffn @ W2 → dsilu_raw (IC=DIM, OC=HIDDEN)
static NSString *gen_ffn_bwd_w2t_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, HIDDEN, SEQ);
}

// ffnBwdW13t: dh1 @ W1 + dh3 @ W3 → dx_ffn (IC=HIDDEN)
// Input: [1, HIDDEN, 1, 2*SEQ+2*DIM] — dh1 + dh3 + W1 + W3
// Output: [1, DIM, 1, SEQ]
static NSString *gen_ffn_bwd_w13t_dynamic(void) {
    int sp_in = FFN_BWD_W13T_SP;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", HIDDEN, sp_in];

    [m appendFormat:@"        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = slice_by_size(x=x,begin=b0,size=sh)[name=string(\"dh1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = slice_by_size(x=x,begin=b1,size=sh)[name=string(\"dh3\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W1t = slice_by_size(x=x,begin=b2,size=sw)[name=string(\"W1t\")];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> W3t = slice_by_size(x=x,begin=b3,size=sw)[name=string(\"W3t\")];\n", HIDDEN, DIM];

    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh12 = reshape(shape=ra,x=dh1)[name=string(\"dh12\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh1t = transpose(perm=pm,x=dh12)[name=string(\"dh1t\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh32 = reshape(shape=ra,x=dh3)[name=string(\"dh32\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dh3t = transpose(perm=pm,x=dh32)[name=string(\"dh3t\")];\n", SEQ, HIDDEN];
    [m appendFormat:@"        tensor<int32, [4]> rww = const()[name=string(\"rww\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W1t2 = reshape(shape=rww,x=W1t)[name=string(\"W1t2\")];\n", HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W3t2 = reshape(shape=rww,x=W3t)[name=string(\"W3t2\")];\n", HIDDEN, DIM];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dx1m = matmul(transpose_x=bF,transpose_y=bF,x=dh1t,y=W1t2)[name=string(\"dx1m\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dx3m = matmul(transpose_x=bF,transpose_y=bF,x=dh3t,y=W3t2)[name=string(\"dx3m\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxm = add(x=dx1m,y=dx3m)[name=string(\"dxm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxt = transpose(perm=pm,x=dxm)[name=string(\"dxt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (dx);\n}\n"];
    return m;
}

// wotBwd: dy @ Wo → da (IC=DIM, OC=DIM) — computes Wo^T @ dy
static NSString *gen_wot_bwd_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, DIM, SEQ);
}

// qBwd: dq @ Wq → dx_q (IC=DIM, OC=DIM) — computes Wq^T @ dq
static NSString *gen_q_bwd_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, DIM, SEQ);
}

// kvBwd: dk @ Wk + dv @ Wv → dx_kv (IC=KV_DIM for GQA)
// Input: [1, KV_DIM, 1, 2*SEQ+2*DIM] — dk + dv + Wk + Wv
// Output: [1, DIM, 1, SEQ]
static NSString *gen_kv_bwd_dynamic(void) {
    int sp_in = KV_BWD_SP;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", KV_DIM, sp_in];

    [m appendFormat:@"        tensor<int32, [4]> sh = const()[name=string(\"sh\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=x,begin=b0,size=sh)[name=string(\"dk\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=x,begin=b1,size=sh)[name=string(\"dv\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wkt = slice_by_size(x=x,begin=b2,size=sw)[name=string(\"Wkt\")];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", 2*SEQ+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> Wvt = slice_by_size(x=x,begin=b3,size=sw)[name=string(\"Wvt\")];\n", KV_DIM, DIM];

    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dk2 = reshape(shape=ra,x=dk)[name=string(\"dk2\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dkt = transpose(perm=pm,x=dk2)[name=string(\"dkt\")];\n", SEQ, KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dv2 = reshape(shape=ra,x=dv)[name=string(\"dv2\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dvt = transpose(perm=pm,x=dv2)[name=string(\"dvt\")];\n", SEQ, KV_DIM];
    [m appendFormat:@"        tensor<int32, [4]> rww = const()[name=string(\"rww\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wkt2 = reshape(shape=rww,x=Wkt)[name=string(\"Wkt2\")];\n", KV_DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> Wvt2 = reshape(shape=rww,x=Wvt)[name=string(\"Wvt2\")];\n", KV_DIM, DIM];

    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    // [SEQ,KV_DIM] @ [KV_DIM,DIM] → [SEQ,DIM]
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxk = matmul(transpose_x=bF,transpose_y=bF,x=dkt,y=Wkt2)[name=string(\"dxk\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxv = matmul(transpose_x=bF,transpose_y=bF,x=dvt,y=Wvt2)[name=string(\"dxv\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxm = add(x=dxk,y=dxv)[name=string(\"dxm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> dxt = transpose(perm=pm,x=dxm)[name=string(\"dxt\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = reshape(shape=ro,x=dxt)[name=string(\"dx\")];\n", DIM, SEQ];
    [m appendString:@"    } -> (dx);\n}\n"];
    return m;
}

// ===== SDPA backward part 1 (weight-free, has mask, GQA-capable) =====
// Input: [1, 2*DIM+2*KV_DIM, 1, SEQ] = concat(Q, K, V, da)
// Output: [1, KV_DIM+2*SCORE_CH, 1, SEQ] = concat(dV, probs, dp)
// GQA: K/V expanded before attention, dV reduced back to N_KV_HEADS
static NSString *gen_sdpa_bwd1_dynamic(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd1_in = 2*DIM + 2*KV_DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd1_in, SEQ];

    // Slice Q [DIM], K [KV_DIM], V [KV_DIM], da [DIM]
    [m appendFormat:@"        tensor<int32, [4]> szq = const()[name=string(\"szq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=szq)[name=string(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> szk = const()[name=string(\"szk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=szk)[name=string(\"s1\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=szk)[name=string(\"s2\")];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+2*KV_DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> da = slice_by_size(x=x,begin=b3,size=szq)[name=string(\"s3\")];\n", DIM, SEQ];

    // Reshape Q and da to heads: [1,DIM,1,SEQ] → [1,HEADS,HD,SEQ] → [1,HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=da)[name=string(\"rd\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> d = transpose(perm=pm,x=dr)[name=string(\"td\")];\n", HEADS, SEQ, HD];
    // Reshape KV to N_KV_HEADS: [1,KV_DIM,1,SEQ] → [1,N_KV_HEADS,HD,SEQ] → [1,N_KV_HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> kvrsh = const()[name=string(\"kvrsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=kvrsh,x=kf)[name=string(\"rk\")];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k0 = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", N_KV_HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=kvrsh,x=vf)[name=string(\"rv\")];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v0 = transpose(perm=pm,x=vr)[name=string(\"tv\")];\n", N_KV_HEADS, SEQ, HD];

    // GQA expansion
    if (HEADS_PER_KV > 1) {
        [m appendFormat:@"        tensor<int32, [4]> gqa_fl = const()[name=string(\"gqafl\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> k_fl = reshape(shape=gqa_fl,x=k0)[name=string(\"kfl\")];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> v_fl = reshape(shape=gqa_fl,x=v0)[name=string(\"vfl\")];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<int32, [4]> gqa_rp = const()[name=string(\"gqarp\"), val=tensor<int32, [4]>([1,1,%d,1])];\n", HEADS_PER_KV];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_tl = tile(reps=gqa_rp,x=k_fl)[name=string(\"ktl\")];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v_tl = tile(reps=gqa_rp,x=v_fl)[name=string(\"vtl\")];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = reshape(shape=rsh,x=k_tl)[name=string(\"kex\")];\n", HEADS, SEQ, HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = reshape(shape=rsh,x=v_tl)[name=string(\"vex\")];\n", HEADS, SEQ, HD];
    } else {
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = identity(x=k0)[name=string(\"kex\")];\n", HEADS, SEQ, HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = identity(x=v0)[name=string(\"vex\")];\n", HEADS, SEQ, HD];
    }

    // Recompute attention
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ, SEQ, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS, SEQ, SEQ];

    // dV = probs^T @ da: [HEADS,SEQ,HD], dp = da @ V_exp^T: [HEADS,SEQ,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=d)[name=string(\"dv\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=d,y=v)[name=string(\"dp\")];\n", HEADS, SEQ, SEQ];

    // GQA reduction: dV [HEADS,SEQ,HD] → [N_KV_HEADS,SEQ,HD] by summing groups
    if (HEADS_PER_KV > 1) {
        // [1,HEADS,SEQ,HD] → [1,N_KV_HEADS,HEADS_PER_KV,SEQ*HD] → reduce_sum → [1,N_KV_HEADS,1,SEQ*HD]
        [m appendFormat:@"        tensor<int32, [4]> grp_sh = const()[name=string(\"grps\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv_grp = reshape(shape=grp_sh,x=dv4)[name=string(\"dvg\")];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendString:@"        tensor<int32, [1]> grp_ax = const()[name=string(\"gax\"), val=tensor<int32, [1]>([2])];\n"];
        [m appendString:@"        bool grp_kd = const()[name=string(\"gkd\"), val=bool(true)];\n"];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv_red = reduce_sum(x=dv_grp,axes=grp_ax,keep_dims=grp_kd)[name=string(\"dvr\")];\n", N_KV_HEADS, SEQ*HD];
        // [1,N_KV_HEADS,1,SEQ*HD] → reshape [1,N_KV_HEADS,SEQ,HD] → transpose → [1,N_KV_HEADS,HD,SEQ] → reshape [1,KV_DIM,1,SEQ]
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv_h = reshape(shape=kvrsh,x=dv_red)[name=string(\"dvh\")];\n", N_KV_HEADS, SEQ, HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv_h)[name=string(\"dvt\")];\n", N_KV_HEADS, HD, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n", KV_DIM, SEQ];
    } else {
        // MHA: dV already at HEADS = N_KV_HEADS
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n", HEADS, HD, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n", KV_DIM, SEQ];
    }

    // Flatten probs and dp for output
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];\n", SCORE_CH, SEQ];

    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];\n", KV_DIM+2*SCORE_CH, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ===== SDPA backward part 2 (weight-free, GQA-capable) =====
// Input: [1, 2*SCORE_CH+DIM+KV_DIM, 1, SEQ] = concat(probs, dp, Q, K)
// Output: [1, DIM+KV_DIM, 1, SEQ] = concat(dQ, dK)
// GQA: K expanded before computation, dK reduced back to N_KV_HEADS
static NSString *gen_sdpa_bwd2_dynamic(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + DIM + KV_DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];

    // Slice probs [SCORE_CH], dp [SCORE_CH], Q [DIM], K [KV_DIM]
    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];\n", SCORE_CH, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_q = const()[name=string(\"szq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_q)[name=string(\"s2\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_k = const()[name=string(\"szk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_k)[name=string(\"s3\")];\n", KV_DIM, SEQ];

    // Reshape probs/dp to heads
    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];\n", HEADS, SEQ, SEQ];
    // Reshape Q to heads: [1,DIM,1,SEQ] → [1,HEADS,HD,SEQ] → [1,HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS, HD, SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS, SEQ, HD];
    // Reshape K to KV heads: [1,KV_DIM,1,SEQ] → [1,N_KV_HEADS,HD,SEQ] → [1,N_KV_HEADS,SEQ,HD]
    [m appendFormat:@"        tensor<int32, [4]> kvrsh = const()[name=string(\"kvrsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=kvrsh,x=kf)[name=string(\"rk\")];\n", N_KV_HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k0 = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", N_KV_HEADS, SEQ, HD];

    // GQA expansion for K
    if (HEADS_PER_KV > 1) {
        [m appendFormat:@"        tensor<int32, [4]> gqa_fl = const()[name=string(\"gqafl\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> k_fl = reshape(shape=gqa_fl,x=k0)[name=string(\"kfl\")];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<int32, [4]> gqa_rp = const()[name=string(\"gqarp\"), val=tensor<int32, [4]>([1,1,%d,1])];\n", HEADS_PER_KV];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_tl = tile(reps=gqa_rp,x=k_fl)[name=string(\"ktl\")];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = reshape(shape=rsh,x=k_tl)[name=string(\"kex\")];\n", HEADS, SEQ, HD];
    } else {
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = identity(x=k0)[name=string(\"kex\")];\n", HEADS, SEQ, HD];
    }

    // Softmax backward: ds = probs * (dp - sum(probs*dp, axis=-1))
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];\n", HEADS, SEQ, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];\n", HEADS, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];\n", HEADS, SEQ, SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];\n", HEADS, SEQ, SEQ];

    // dQ = ds @ K_exp: [HEADS,SEQ,HD], dK = ds^T @ Q: [HEADS,SEQ,HD]
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];\n", HEADS, SEQ, HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];\n", HEADS, SEQ, HD];

    // Reshape dQ to flat [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];\n", HEADS, HD, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fsq = const()[name=string(\"fsq\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fsq,x=dqt)[name=string(\"dqf\")];\n", DIM, SEQ];

    // GQA reduction: dK [HEADS,SEQ,HD] → [N_KV_HEADS,SEQ,HD]
    if (HEADS_PER_KV > 1) {
        [m appendFormat:@"        tensor<int32, [4]> grp_sh = const()[name=string(\"grps\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk_grp = reshape(shape=grp_sh,x=dk4)[name=string(\"dkg\")];\n", N_KV_HEADS, HEADS_PER_KV, SEQ*HD];
        [m appendString:@"        tensor<int32, [1]> grp_ax = const()[name=string(\"gax\"), val=tensor<int32, [1]>([2])];\n"];
        [m appendString:@"        bool grp_kd = const()[name=string(\"gkd\"), val=bool(true)];\n"];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk_red = reduce_sum(x=dk_grp,axes=grp_ax,keep_dims=grp_kd)[name=string(\"dkr\")];\n", N_KV_HEADS, SEQ*HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk_h = reshape(shape=kvrsh,x=dk_red)[name=string(\"dkh\")];\n", N_KV_HEADS, SEQ, HD];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk_h)[name=string(\"dkt\")];\n", N_KV_HEADS, HD, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> fsk = const()[name=string(\"fsk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fsk,x=dkt)[name=string(\"dkf\")];\n", KV_DIM, SEQ];
    } else {
        // MHA: dK already at correct size
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];\n", HEADS, HD, SEQ];
        [m appendFormat:@"        tensor<int32, [4]> fsk = const()[name=string(\"fsk\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", KV_DIM, SEQ];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fsk,x=dkt)[name=string(\"dkf\")];\n", KV_DIM, SEQ];
    }

    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];\n", DIM+KV_DIM, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Causal mask blob
static NSData *g_mask_blob = nil;
static NSData *get_mask_blob(void) {
    if (!g_mask_blob) {
        _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
        for(int t=0;t<SEQ;t++) for(int t2=0;t2<SEQ;t2++)
            mask[t*SEQ+t2] = (t2<=t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        g_mask_blob = build_blob_fp16(mask, SEQ*SEQ);
        free(mask);
    }
    return g_mask_blob;
}
