// ane_runtime.h — Reusable ANE in-memory compile/load/eval wrapper
// Uses _ANEInMemoryModel via private AppleNeuralEngine.framework
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>

typedef struct {
    id model;               // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;             // _ANERequest
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
} ANEKernel;

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;
static bool g_ane_loaded = false;

static void ane_init(void) {
    if (g_ane_loaded) return;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ane_loaded = true;
}

static IOSurfaceRef ane_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

// Compile a MIL graph with weight blob into an ANE kernel.
// milText: NSData of MIL text
// weightData: NSData of raw weight blob (can be nil)
// inputSizes/outputSizes: arrays of byte sizes for each I/O tensor
static ANEKernel *ane_compile(NSData *milText, NSData *weightData,
                               int nInputs, size_t *inputSizes,
                               int nOutputs, size_t *outputSizes) {
    ane_init();
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightData}};
    }
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milText, wdict, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    // Pre-populate temp dir with MIL + weights
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    if (weightData)
        [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE compile failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE load failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    ANEKernel *k = calloc(1, sizeof(ANEKernel));
    k->model = mdl;
    k->tmpDir = td;
    k->nInputs = nInputs;
    k->nOutputs = nOutputs;
    k->inputBytes = malloc(nInputs * sizeof(size_t));
    k->outputBytes = malloc(nOutputs * sizeof(size_t));
    memcpy(k->inputBytes, inputSizes, nInputs * sizeof(size_t));
    memcpy(k->outputBytes, outputSizes, nOutputs * sizeof(size_t));

    // Create IOSurfaces
    k->ioInputs = malloc(nInputs * sizeof(IOSurfaceRef));
    k->ioOutputs = malloc(nOutputs * sizeof(IOSurfaceRef));
    for (int i = 0; i < nInputs; i++)
        k->ioInputs[i] = ane_create_surface(inputSizes[i]);
    for (int i = 0; i < nOutputs; i++)
        k->ioOutputs[i] = ane_create_surface(outputSizes[i]);

    // Build request
    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:nInputs];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:nInputs];
    for (int i = 0; i < nInputs; i++) {
        [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:nOutputs];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:nOutputs];
    for (int i = 0; i < nOutputs; i++) {
        [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
        [oIdx addObject:@(i)];
    }
    k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);

    return k;
}

static void ane_write_input(ANEKernel *k, int idx, const void *data, size_t bytes) {
    IOSurfaceLock(k->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(k->ioInputs[idx], 0, NULL);
}

static void ane_read_output(ANEKernel *k, int idx, void *data, size_t bytes) {
    IOSurfaceLock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(k->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

static bool ane_eval(ANEKernel *k) {
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, k->request, &e);
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n",
                e ? [[e description] UTF8String] : "unknown error");
    }
    return ok;
}

static void ane_free(ANEKernel *k) {
    if (!k) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    for (int i = 0; i < k->nInputs; i++) CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++) CFRelease(k->ioOutputs[i]);
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    free(k->ioInputs); free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    free(k);
}
