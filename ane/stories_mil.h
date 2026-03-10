// stories_mil.h — MIL program generators for ANE kernels
// Same architecture as single-layer train_large.m but parameterized
#pragma once
#include "stories_io.h"

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// SDPA forward + taps: x_in → rmsnorm → QKV+SDPA+Wo → concat(o_out, Q, K, V, attn_out, xnorm)
static NSString *gen_sdpa_fwd_taps(void) {
    float sc = 1.0f/sqrtf((float)HD);
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", HEADS,SEQ,HD];
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ,SEQ,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn))[name=string(\"cat\")];\n", 6*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN forward + taps: x2 → rmsnorm → FFN → concat(ffn_out, h1, h3, silu_out, x2norm)
static NSString *gen_ffn_fwd_taps(void) {
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", DIM,HIDDEN,DIM,HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn))[name=string(\"cat\")];\n", 2*DIM+3*HIDDEN,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN backward: concat(dffn,h1,h3) → concat(dx,dh1,dh3)
static NSString *gen_ffn_bwd(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@CONV_CONST];
    [m appendString:@"        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=string(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=string(\"s1x\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=string(\"s3x\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];\n", HIDDEN, DIM, HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)[name=string(\"cw2\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendString:@"        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig)[name=string(\"oms\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs)[name=string(\"brk\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)[name=string(\"cw1\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)[name=string(\"cw3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3)[name=string(\"adx\")];\n", DIM, SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=string(\"cat\")];\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// QKV backward: concat(dq,dk,dv) → dx
static NSString *gen_qkvb(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 3*DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dq = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wqt = const()[name=string(\"Wqt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wqt.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wkt = const()[name=string(\"Wkt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wkt.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wvt = const()[name=string(\"Wvt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wvt.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxq = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wqt,x=dq)[name=string(\"cq\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxk = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wkt,x=dk)[name=string(\"ck\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxv = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wvt,x=dv)[name=string(\"cv\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxqk = add(x=dxq,y=dxk)[name=string(\"aqk\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=dxqk,y=dxv)[name=string(\"out\")];\n", DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 1 + Wo^T
static NSString *gen_sdpa_bwd1(void) {
    float sc = 1.0f/sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 4*DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx2f = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wot = const()[name=string(\"Wot\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wot.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> df = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wot,x=dx2f)[name=string(\"cwo\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf)[name=string(\"rv\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr)[name=string(\"tv\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=df)[name=string(\"rd\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> da = transpose(perm=pm,x=dr)[name=string(\"td\")];\n", HEADS,SEQ,HD];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ,SEQ,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=da)[name=string(\"dv\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];\n", SCORE_CH,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];\n", DIM+2*SCORE_CH,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 2: concat(probs,dp,Q,K) → concat(dQ,dK)
static NSString *gen_sdpa_bwd2(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + 2*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];\n", HEADS,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];\n", 2*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Mask blob (causal mask [SEQ,SEQ])
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
