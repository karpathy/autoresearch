# Codebase Concerns

**Analysis Date:** 2026-03-25

## Tech Debt

**Duplicate `PadToSquare` class:**
- Issue: `PadToSquare` is defined twice in `finetune_trendyol_arcface3.py` -- once at line 57 using `tf.pad` (undefined import, likely a bug) and again at line 226 using `TF.pad` (correct import from `torchvision.transforms.functional`). The first definition shadows the second at module scope, but the second re-shadows the first. Whichever is used depends on class resolution order.
- Files: `finetune_trendyol_arcface3.py:57-76`, `finetune_trendyol_arcface3.py:226-244`
- Impact: The first definition at line 57 references `tf.pad` which is never imported -- this will raise `NameError` if instantiated before the second definition overrides it. In practice the second definition (line 226) wins because Python uses the last class definition at module scope.
- Fix approach: Delete the first `PadToSquare` definition (lines 57-76). It is dead code with a bug.

**Duplicate collate functions:**
- Issue: `collate_distill` and `collate_arcface` at lines 669-690 in `finetune_trendyol_arcface3.py` are identical implementations. This is unnecessary duplication.
- Files: `finetune_trendyol_arcface3.py:669-690`
- Impact: Maintenance burden -- any fix to one must be applied to the other.
- Fix approach: Unify into a single `collate_with_paths` function.

**Hardcoded absolute paths throughout finetuning script:**
- Issue: Multiple absolute paths to `/data/mnt/mnt_ml_shared/...` are hardcoded as default argument values and inline constants. This makes the script non-portable across machines.
- Files: `finetune_trendyol_arcface3.py:86`, `finetune_trendyol_arcface3.py:1185`, `finetune_trendyol_arcface3.py:1191`, `finetune_trendyol_arcface3.py:1197`, `finetune_trendyol_arcface3.py:1276-1279`
- Impact: Script fails on any machine without these exact mount paths. The ONNX teacher path at line 86 is not even configurable via CLI -- it is embedded in the `TrendyolEmbedder.__init__` default.
- Fix approach: Move all default paths to a configuration file or environment variables. Make the ONNX path a required CLI argument or use a discoverable default.

**Stale mutable state on function object:**
- Issue: `main._best_combined` is stored as an attribute on the `main` function object (line 1409, 1480-1481). This is an unusual pattern that persists state across calls if `main()` were ever called twice.
- Files: `finetune_trendyol_arcface3.py:1409`, `finetune_trendyol_arcface3.py:1480-1481`
- Impact: Fragile -- breaks if the function is called more than once in a process. Confusing to readers.
- Fix approach: Use a local variable instead. The `_best_combined` value is only used within the same `main()` invocation.

**Global module-level execution in `train.py`:**
- Issue: `train.py` executes all setup and training at module scope (not inside `if __name__ == "__main__"`). Lines 457-631 are all top-level statements. This means importing `train.py` for any reason (testing, tooling) triggers full model initialization and training.
- Files: `train.py:457-631`
- Impact: Cannot import individual components (GPT, MuonAdamW, etc.) without triggering CUDA initialization and training. Makes unit testing impossible.
- Fix approach: Wrap lines 457-631 in `if __name__ == "__main__":` block.

**`requirements.txt` is disconnected from `pyproject.toml`:**
- Issue: `requirements.txt` lists a completely different set of dependencies (loguru, onnxruntime-gpu, Pillow, timm, torchvision, transformers, matplotlib) than `pyproject.toml` (kernels, rustbpe, tiktoken, pyarrow, etc.). These serve different scripts but live in the same directory with no documentation of which is for what.
- Files: `requirements.txt`, `pyproject.toml`
- Impact: Confusion about which dependency file to use. `pyproject.toml` is for `train.py`/`prepare.py` (LLM pretraining). `requirements.txt` is for `finetune_trendyol_arcface3.py` (vision finetuning). Neither references the other.
- Fix approach: Document which dependency file serves which script. Consider separate subdirectories or a unified dependency management approach.

## Known Bugs

**First `PadToSquare` uses undefined `tf` variable:**
- Symptoms: `NameError: name 'tf' is not defined` if the first `PadToSquare` class (line 57) is instantiated
- Files: `finetune_trendyol_arcface3.py:69-74`
- Trigger: The class is defined but the second definition at line 226 shadows it. The bug is latent -- it only manifests if someone references the first class explicitly.
- Workaround: The second definition at line 226 is the one actually used by all code paths.

**Silent data corruption on image load failure:**
- Symptoms: When an image file cannot be opened, both `CombinedDistillDataset.__getitem__` (line 522-524) and `CombinedArcFaceDataset.__getitem__` (line 652-656) silently replace the failed image with a random sample from the primary dataset. No logging occurs.
- Files: `finetune_trendyol_arcface3.py:520-524`, `finetune_trendyol_arcface3.py:652-656`
- Trigger: Corrupted image file on disk, race condition with file deletion, or permission error.
- Workaround: None. The bare `except Exception` swallows all errors including unexpected ones (e.g., `KeyboardInterrupt` in Python 3 is not caught, but `MemoryError` is).

**`_TEACHER_MEM_CACHE` unbounded memory growth:**
- Symptoms: Memory usage grows without bound during training as teacher embeddings are cached in `_TEACHER_MEM_CACHE` (line 757). With large datasets, this can exhaust system RAM.
- Files: `finetune_trendyol_arcface3.py:757`
- Trigger: Training on a dataset with many unique image paths.
- Workaround: The disk cache at `cache_dir` means embeddings are computed only once per image, but the in-memory dict still grows.

## Security Considerations

**Pickle deserialization of tokenizer:**
- Risk: `prepare.py` loads a tokenizer via `pickle.load()` (line 219). Pickle deserialization can execute arbitrary code. If the cached tokenizer file at `~/.cache/autoresearch/tokenizer/tokenizer.pkl` is tampered with, arbitrary code runs when `train.py` imports the tokenizer.
- Files: `prepare.py:219`
- Current mitigation: The pickle file is generated locally by `prepare.py` (line 178-179), not downloaded from the internet.
- Recommendations: Consider using a safer serialization format, or at minimum validate the pickle source. The file path is predictable (`~/.cache/autoresearch/tokenizer/tokenizer.pkl`).

**`trust_remote_code=True` in teacher model loading:**
- Risk: `DINOv2Teacher` loads a HuggingFace model with `trust_remote_code=True` (line 195), which executes arbitrary Python code from the model repository.
- Files: `finetune_trendyol_arcface3.py:195`
- Current mitigation: The model is loaded from a specific known repository (`Trendyol/trendyol-dino-v2-ecommerce-256d`).
- Recommendations: Pin the model revision to a specific commit hash to prevent supply-chain attacks if the repo is compromised.

**MD5 used for cache keys:**
- Risk: Teacher embedding cache uses MD5 hashes of file paths as cache filenames (line 776). MD5 is cryptographically broken. Two different paths could theoretically produce the same hash, causing one image's embedding to be used for another.
- Files: `finetune_trendyol_arcface3.py:776`
- Current mitigation: Collision probability is astronomically low for the number of files in a typical dataset. The risk is theoretical.
- Recommendations: Use SHA-256 for correctness. The performance difference is negligible.

**LD_LIBRARY_PATH manipulation at import time:**
- Risk: `finetune_trendyol_arcface3.py` modifies `LD_LIBRARY_PATH` at module import time (lines 14-24) based on site-packages contents. This affects all subsequent library loading in the process.
- Files: `finetune_trendyol_arcface3.py:14-24`
- Current mitigation: Only adds paths from within the Python site-packages directory.
- Recommendations: Use a proper CUDA environment setup instead of runtime path manipulation.

## Performance Bottlenecks

**Best-fit document packing with linear scan:**
- Problem: The dataloader in `prepare.py` uses a linear scan (O(n)) over a buffer of 1000 documents to find the best-fit document for each slot (lines 316-320). This runs for every position in every row of every batch.
- Files: `prepare.py:315-320`
- Cause: Linear search through `doc_buffer` list for every position fill.
- Improvement path: Use a sorted data structure (e.g., sorted list with bisect) for O(log n) lookups. However, with buffer_size=1000 this may not be a real bottleneck compared to GPU training time.

**Teacher embedding inference not batched optimally:**
- Problem: `load_teacher_embeddings` in `finetune_trendyol_arcface3.py` opens PIL images one at a time in a list comprehension (line 786), then passes them all to `encode_batch`. The image loading is sequential.
- Files: `finetune_trendyol_arcface3.py:786`
- Cause: Sequential PIL image opening for uncached images.
- Improvement path: Use a DataLoader with multiple workers for parallel image loading, or pre-cache all embeddings in a separate pass.

**GC disabled in training loop:**
- Problem: `train.py` disables Python garbage collection after step 0 (lines 593-596) to avoid ~500ms stalls. GC only runs every 5000 steps (line 597-598). This trades latency for potential memory accumulation.
- Files: `train.py:593-598`
- Cause: Python's cyclic GC causes unpredictable pauses during GPU training.
- Improvement path: This is an intentional optimization. The periodic GC at 5000-step intervals is a reasonable compromise. No change needed unless memory issues arise.

**Retrieval eval computes full NxN similarity matrix:**
- Problem: `run_retrieval_eval` computes `emb @ emb.T` (line 1122) which is O(n^2) in both compute and memory. With `max_samples=5000` this is a 5000x5000 matrix, which is fine, but would not scale.
- Files: `finetune_trendyol_arcface3.py:1122`
- Cause: Brute-force similarity computation.
- Improvement path: Use FAISS or chunked similarity computation for larger eval sets. Current default of 5000 samples is manageable.

## Fragile Areas

**GPU-specific kernel selection in `train.py`:**
- Files: `train.py:21-24`
- Why fragile: Flash Attention 3 kernel is selected based on `torch.cuda.get_device_capability()`. The code assumes exactly two options: Hopper (9,0) uses one repo, everything else uses another. New GPU architectures (e.g., Blackwell) may need different handling. The `kernels` package and its repos (`varunneal/flash-attention-3`, `kernels-community/flash-attn3`) are external dependencies that could break.
- Safe modification: Test on the target GPU before deploying. Add fallback to PyTorch's native SDPA if FA3 is unavailable.
- Test coverage: None.

**`prepare.py` is marked as read-only but has no enforcement:**
- Files: `program.md:29`, `prepare.py`
- Why fragile: The README/program.md states `prepare.py` must not be modified, but there is no file permission, CI check, or linting rule to enforce this. The autonomous experimentation loop in `program.md` could accidentally modify it.
- Safe modification: Do not modify. The evaluation metric (`evaluate_bpb`) lives here.
- Test coverage: None.

**Monkey-patching transformers library:**
- Files: `finetune_trendyol_arcface3.py:173-185`
- Why fragile: `_patch_transformers_compat()` monkey-patches `PreTrainedModel.mark_tied_weights_as_initialized` to work around a transformers 5.x compatibility issue. This will break if the transformers library changes the method signature or if the underlying issue is fixed.
- Safe modification: Check if the patch is still needed when upgrading transformers. Remove when no longer necessary.
- Test coverage: None.

**Window size computation assumes specific pattern chars:**
- Files: `train.py:196-206`
- Why fragile: `_compute_window_sizes` only accepts "S" and "L" characters in the window pattern. Any other character causes an assertion failure. The assertion at line 197 provides no helpful error message.
- Safe modification: Add descriptive error messages. Document valid pattern characters.
- Test coverage: None.

## Scaling Limits

**Single-GPU only:**
- Current capacity: Training runs on a single GPU only. The optimizer (`MuonAdamW`) and training loop have no distributed training support.
- Limit: Model size and training speed are bounded by single-GPU VRAM and compute.
- Scaling path: The code comments note this is "Single-GPU, single-file" by design. Multi-GPU would require significant refactoring of the optimizer (gradient all-reduce for Muon's orthogonalization step) and dataloader.

**In-memory teacher embedding cache:**
- Current capacity: All teacher embeddings for accessed images are held in `_TEACHER_MEM_CACHE` (a module-level dict).
- Limit: System RAM. With 256-dim float32 embeddings (~1KB each), 1M images would use ~1GB. Larger datasets could exhaust memory.
- Scaling path: Add an LRU cache with a configurable maximum size, or rely entirely on the disk cache.

## Dependencies at Risk

**`kernels` package (external kernel registry):**
- Risk: `train.py` depends on the `kernels` package (line 20) to fetch Flash Attention 3 implementations from GitHub repos (`varunneal/flash-attention-3`, `kernels-community/flash-attn3`). These are third-party repos that could be deleted, renamed, or push breaking changes.
- Impact: Training script cannot start if the kernel cannot be fetched.
- Migration plan: Pin specific kernel versions. Consider vendoring the FA3 implementation or falling back to PyTorch native attention.

**`rustbpe` tokenizer:**
- Risk: `rustbpe` (version >= 0.1.0) is a Rust-based BPE tokenizer with a Python binding. It is a relatively niche package. If it becomes unmaintained, building from source on new platforms could fail.
- Impact: Cannot train new tokenizers without it. Existing trained tokenizers (pickled tiktoken objects) can still be loaded.
- Migration plan: The trained tokenizer is stored as a tiktoken `Encoding` object, so `rustbpe` is only needed for tokenizer training, not inference.

**`onnxruntime-gpu` version pinning:**
- Risk: `requirements.txt` pins `onnxruntime-gpu>=1.17,<1.20`. This version range may not support newer CUDA versions or GPU architectures.
- Impact: ONNX teacher model inference may fail on newer GPU setups.
- Migration plan: Test with newer onnxruntime versions as they are released. The LD_LIBRARY_PATH hack in `finetune_trendyol_arcface3.py:14-24` suggests this has already been a pain point.

## Missing Critical Features

**No test suite:**
- Problem: There are zero test files in the repository. No unit tests, integration tests, or smoke tests for any component.
- Blocks: Cannot verify correctness of model, optimizer, dataloader, or evaluation code without running full training. Refactoring is risky.

**No configuration management:**
- Problem: `train.py` uses module-level constants (lines 432-451) with a comment "edit these directly, no CLI flags needed." `finetune_trendyol_arcface3.py` uses argparse with hardcoded defaults. There is no unified configuration system.
- Blocks: Cannot run experiments with different configurations without editing source code (for `train.py`) or passing many CLI flags (for finetuning).

**No checkpoint/resume for pretraining:**
- Problem: `train.py` has no checkpoint saving or resumption capability. If training is interrupted (OOM, hardware failure, timeout), all progress is lost.
- Blocks: Cannot recover from interruptions during the 5-minute training window. This is partially by design (the time budget is short), but becomes a concern for longer runs or larger models.

## Test Coverage Gaps

**All code is untested:**
- What's not tested: Every component -- GPT model, MuonAdamW optimizer, dataloader packing, BPB evaluation, tokenizer training, finetuning pipeline, ArcFace loss, teacher embedding caching, ONNX export.
- Files: `train.py`, `prepare.py`, `finetune_trendyol_arcface3.py`
- Risk: Bugs in loss computation, gradient flow, data loading, or evaluation silently produce incorrect results. The only validation is whether `val_bpb` improves, which does not catch subtle correctness issues.
- Priority: High for core components (optimizer correctness, evaluation metric, data loading). Medium for finetuning pipeline.

---

*Concerns audit: 2026-03-25*
