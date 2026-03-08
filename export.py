"""Export trained model to Ollama.
Usage: python export.py experiments/1709856000/ --name autoresearch-latest
"""
import argparse, os, subprocess

def export(model_dir, name="autoresearch-latest"):
    abs_path = os.path.abspath(model_dir)
    assert os.path.exists(f"{abs_path}/model.safetensors"), f"No model in {model_dir}"
    modelfile_path = f"{abs_path}/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(f'FROM {abs_path}\nPARAMETER temperature 0.8\n')
    print(f"Creating Ollama model '{name}'...")
    r = subprocess.run(["ollama", "create", name, "-f", modelfile_path], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"Done! Run: ollama run {name}")
    else:
        print(f"Failed: {r.stderr}\nManual: ollama create {name} -f {modelfile_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--name", default="autoresearch-latest")
    export(**vars(p.parse_args()))
