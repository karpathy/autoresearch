// ane_mil_gen.h — Generate MIL text for conv-based linear ops + weight blobs
#pragma once
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Build an FP16 weight blob with the required header structure.
// weights_f32: source weights in row-major [out_ch, in_ch]
// Returns NSData with header + FP16 weights
static NSData *mil_build_weight_blob(const float *weights_f32, int out_ch, int in_ch) {
    NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2; // FP16
    NSUInteger total = 64 + 64 + wsize; // global header + chunk header + data
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8) = (uint32_t)wsize;   // data_size
    *(uint32_t*)(chunk + 16) = 128;               // data_offset (from file start)
    // Convert f32 → fp16 (simple truncation via _Float16)
    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)out_ch * in_ch; i++)
        fp16[i] = (_Float16)weights_f32[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Generate MIL for a single matmul: y = W @ x (using matmul op, weights as input)
// Input x: [1, in_ch, spatial] fp32
// Input W: [1, out_ch, in_ch] fp32
// Output:  [1, out_ch, spatial] fp32
static NSString *mil_gen_matmul(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, %d]> x, tensor<fp32, [1, %d, %d]> W) {\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n"
        "        tensor<fp16, [1, %d, %d]> W16 = cast(dtype = to_fp16, x = W)[name = string(\"cast_W\")];\n"
        "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
        "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
        "        tensor<fp16, [1, %d, %d]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)[name = string(\"mm\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial, out_ch, in_ch,
        in_ch, spatial, out_ch, in_ch,
        out_ch, spatial, out_ch, spatial];
}

// Keep the baked-weight version for reference (used in inference-only scenarios)
static NSString *mil_gen_conv(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial, in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, spatial, out_ch, spatial];
}

// Generate MIL for fused QKV: 3 parallel convs from same input
// Input:  [1, dim, 1, S]
// Outputs: Q[1, dim, 1, S], K[1, dim, 1, S], V[1, dim, 1, S]
// Weight blob layout: Wq[dim,dim] @ offset 64, Wk @ offset 64+cs, Wv @ offset 64+2*cs
// where cs = 64 + dim*dim*2
static NSString *mil_gen_qkv(int dim, int spatial) {
    NSUInteger cs = 64 + (NSUInteger)dim * dim * 2;
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wq = const()[name = string(\"Wq\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wk = const()[name = string(\"Wk\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> Wv = const()[name = string(\"Wv\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> q16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x16)[name = string(\"conv_q\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> k16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x16)[name = string(\"conv_k\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> v16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x16)[name = string(\"conv_v\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> q = cast(dtype = to_fp32, x = q16)[name = string(\"cast_q\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> k = cast(dtype = to_fp32, x = k16)[name = string(\"cast_k\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> v = cast(dtype = to_fp32, x = v16)[name = string(\"cast_v\")];\n"
        "    } -> (q, k, v);\n"
        "}\n",
        dim, spatial, dim, spatial,
        dim, dim, dim, dim,
        dim, dim, dim, dim, (unsigned long)(64 + cs),
        dim, dim, dim, dim, (unsigned long)(64 + 2*cs),
        dim, spatial, dim, spatial, dim, spatial,
        dim, spatial, dim, spatial, dim, spatial];
}

// Build weight blob for fused QKV (3 weight matrices concatenated)
static NSData *mil_build_qkv_weight_blob(const float *wq, const float *wk, const float *wv, int dim) {
    NSUInteger wsize = (NSUInteger)dim * dim * 2;
    NSUInteger cs = 64 + wsize;
    NSUInteger total = 64 + 3 * cs;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    const float *ws[3] = {wq, wk, wv};
    for (int w = 0; w < 3; w++) {
        uint8_t *chunk = buf + 64 + w * cs;
        chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE;
        chunk[4]=0x01;
        *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
        *(uint32_t*)(chunk + 16) = (uint32_t)(64 + w * cs + 64); // absolute data offset
        _Float16 *fp16 = (_Float16*)(chunk + 64);
        for (NSUInteger i = 0; i < (NSUInteger)dim * dim; i++)
            fp16[i] = (_Float16)ws[w][i];
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Build weight blob for fused FFN up (w1 + w3, both [hidden_dim, dim])
static NSData *mil_build_ffn_up_weight_blob(const float *w1, const float *w3, int hidden_dim, int dim) {
    NSUInteger wsize = (NSUInteger)hidden_dim * dim * 2;
    NSUInteger cs = 64 + wsize;
    NSUInteger total = 64 + 2 * cs;
    uint8_t *buf = (uint8_t*)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    const float *ws[2] = {w1, w3};
    for (int w = 0; w < 2; w++) {
        uint8_t *chunk = buf + 64 + w * cs;
        chunk[0]=0xEF; chunk[1]=0xBE; chunk[2]=0xAD; chunk[3]=0xDE;
        chunk[4]=0x01;
        *(uint32_t*)(chunk + 8) = (uint32_t)wsize;
        *(uint32_t*)(chunk + 16) = (uint32_t)(64 + w * cs + 64); // absolute data offset
        _Float16 *fp16 = (_Float16*)(chunk + 64);
        for (NSUInteger i = 0; i < (NSUInteger)hidden_dim * dim; i++)
            fp16[i] = (_Float16)ws[w][i];
    }
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Generate MIL for fused FFN up: w1 + w3 parallel convs
static NSString *mil_gen_ffn_up(int dim, int hidden_dim, int spatial) {
    NSUInteger cs = 64 + (NSUInteger)hidden_dim * dim * 2;
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W1 = const()[name = string(\"W1\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W3 = const()[name = string(\"W3\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(%lu)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> h1 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x16)[name = string(\"conv_w1\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> h3 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x16)[name = string(\"conv_w3\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> out1 = cast(dtype = to_fp32, x = h1)[name = string(\"cast_h1\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> out3 = cast(dtype = to_fp32, x = h3)[name = string(\"cast_h3\")];\n"
        "    } -> (out1, out3);\n"
        "}\n",
        dim, spatial, dim, spatial,
        hidden_dim, dim, hidden_dim, dim,
        hidden_dim, dim, hidden_dim, dim, (unsigned long)(64 + cs),
        hidden_dim, spatial, hidden_dim, spatial,
        hidden_dim, spatial, hidden_dim, spatial];
}
