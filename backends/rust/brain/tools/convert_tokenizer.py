"""
Convert tokenizer to .tiktoken + config files for Rust fallback loader.

Reads from tokenizer.json (preferred) or tokenizer.pkl (pickle).
Outputs:
  - tokenizer.tiktoken: "base64(token_bytes) rank" per line
  - tokenizer_config.txt: pat_str, special tokens, bos_id

The primary Rust loader reads tokenizer.json directly, so this script is
only needed if you want the .tiktoken fallback format.

Usage:
  python3 brain/convert_tokenizer.py [--tokenizer-dir ~/.cache/autoresearch/tokenizer]
"""

import os
import json
import base64
import argparse


def convert(tokenizer_dir):
    json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    tiktoken_path = os.path.join(tokenizer_dir, "tokenizer.tiktoken")
    config_path = os.path.join(tokenizer_dir, "tokenizer_config.txt")

    if os.path.exists(json_path):
        # Load from JSON (the format Rust also reads)
        with open(json_path) as f:
            d = json.load(f)
        pat_str = d["pattern"]
        mergeable_ranks = {base64.b64decode(b64): rank for b64, rank in d["mergeable_ranks"]}
        special_tokens = {name: rank for name, rank in d["special_tokens"]}
    elif os.path.exists(pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            enc = pickle.load(f)
        pat_str = enc._pat_str
        mergeable_ranks = {bytes(k): v for k, v in enc._mergeable_ranks.items()}
        special_tokens = dict(enc._special_tokens)
    else:
        print(f"ERROR: no tokenizer.json or tokenizer.pkl in {tokenizer_dir}")
        return

    # Find BOS
    bos_id = special_tokens.get("<|reserved_0|>")
    if bos_id is None:
        print("ERROR: <|reserved_0|> not in special_tokens")
        return

    # Write .tiktoken
    with open(tiktoken_path, "w") as f:
        for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda x: x[1]):
            b64 = base64.b64encode(token_bytes).decode("ascii")
            f.write(f"{b64} {rank}\n")

    # Write config
    with open(config_path, "w") as f:
        f.write(f"pat_str:{pat_str}\n")
        for name, rank in sorted(special_tokens.items(), key=lambda x: x[1]):
            f.write(f"special:{name} {rank}\n")
        f.write(f"bos_id:{bos_id}\n")

    print(f"Wrote {tiktoken_path} ({len(mergeable_ranks)} ranks)")
    print(f"Wrote {config_path} (pat_str + {len(special_tokens)} special tokens, bos_id={bos_id})")

    # Verify roundtrip
    try:
        import tiktoken
        ranks_check = {}
        with open(tiktoken_path) as f:
            for line in f:
                if not line.strip():
                    continue
                b64, rank = line.strip().split()
                ranks_check[base64.b64decode(b64)] = int(rank)
        assert ranks_check == mergeable_ranks

        enc2 = tiktoken.Encoding(
            name="check", pat_str=pat_str,
            mergeable_ranks=ranks_check, special_tokens=special_tokens,
        )
        test = "Hello world! Numbers: 123."
        assert enc2.encode_ordinary(test) == list(
            tiktoken.Encoding(
                name="orig", pat_str=pat_str,
                mergeable_ranks=mergeable_ranks, special_tokens=special_tokens,
            ).encode_ordinary(test)
        )
        print("Roundtrip verified OK")
    except ImportError:
        print("tiktoken not installed, skipping verification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", default=None)
    args = parser.parse_args()
    if args.tokenizer_dir is None:
        args.tokenizer_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tokenizer")
    convert(args.tokenizer_dir)
