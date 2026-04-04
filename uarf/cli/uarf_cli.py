#!/usr/bin/env python3
"""
UARF Command Line Interface
Ein einfacher Befehl für alle Plattformen.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Universal AutoResearch Framework (UARF) - ML Training für alle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  uarf auto-setup                    Automatische Hardware-Erkennung und Setup
  uarf run                           Startet Training mit Standard-Konfiguration
  uarf run --model Qwen/Qwen2.5-0.5B --time 600
  uarf suggest                       Zeigt empfohlene Modelle für deine Hardware
  uarf export --format gguf          Exportiert trainiertes Modell
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Verfügbare Befehle')
    
    # auto-setup Befehl
    setup_parser = subparsers.add_parser('auto-setup', help='Automatische Hardware-Erkennung')
    setup_parser.add_argument('--print-config', action='store_true', 
                              help='Konfiguration als JSON ausgeben')
    
    # run Befehl
    run_parser = subparsers.add_parser('run', help='Training starten')
    run_parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B',
                           help='Hugging Face Model ID')
    run_parser.add_argument('--dataset', type=str, default='karpathy/tinyshakespeare',
                           help='Dataset Name')
    run_parser.add_argument('--time', type=int, default=300,
                           help='Zeitbudget in Sekunden (default: 300)')
    run_parser.add_argument('--batch-size', type=int, default=None,
                           help='Batch Size (auto wenn nicht angegeben)')
    run_parser.add_argument('--max-seq-len', type=int, default=None,
                           help='Maximale Sequenzlänge')
    run_parser.add_argument('--lr', type=float, default=2e-4,
                           help='Learning Rate')
    run_parser.add_argument('--output-dir', type=str, default='./outputs',
                           help='Output Verzeichnis')
    run_parser.add_argument('--config', type=str, default=None,
                           help='Pfad zur Config JSON Datei')
    run_parser.add_argument('--device', type=str, default='auto',
                           choices=['auto', 'cuda', 'cpu', 'mps'],
                           help='Device Auswahl')
    run_parser.add_argument('--precision', type=str, default='auto',
                           choices=['auto', 'fp32', 'fp16', 'bf16', 'int8'],
                           help='Präzision')
    
    # suggest Befehl
    suggest_parser = subparsers.add_parser('suggest', help='Modellempfehlungen anzeigen')
    suggest_parser.add_argument('--task', type=str, default='text-generation',
                               help='Aufgabentyp')
    suggest_parser.add_argument('--limit', type=int, default=5,
                               help='Maximale Anzahl Empfehlungen')
    
    # export Befehl
    export_parser = subparsers.add_parser('export', help='Modell exportieren')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                              help='Pfad zum Checkpoint')
    export_parser.add_argument('--format', type=str, required=True,
                              choices=['gguf', 'onnx', 'tflite'],
                              help='Export-Format')
    export_parser.add_argument('--output', type=str, default=None,
                              help='Ausgabedatei')
    export_parser.add_argument('--quantization', type=str, default='Q4_K_M',
                              help='Quantisierung für GGUF')
    
    # detect Befehl
    detect_parser = subparsers.add_parser('detect', help='Hardware erkennen')
    detect_parser.add_argument('--json', action='store_true',
                              help='Als JSON ausgeben')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # Commands ausführen
    if args.command == 'auto-setup':
        from uarf import HardwareDetector
        detector = HardwareDetector()
        detector.print_summary()
        
        if args.print_config:
            import json
            config = detector.get_optimal_config()
            print(json.dumps(config, indent=2))
    
    elif args.command == 'run':
        # Import nur wenn benötigt
        from uarf import HardwareDetector, UARFConfig, UniversalTrainer
        
        # Hardware erkennen
        print("🔍 Hardware wird erkannt...")
        detector = HardwareDetector()
        detector.print_summary()
        
        # Config erstellen
        if args.config:
            config = UARFConfig.from_json(args.config)
        else:
            config = UARFConfig(
                model_id=args.model,
                dataset_name=args.dataset,
                time_budget_seconds=args.time,
                device=args.device,
                precision=args.precision,
                output_dir=args.output_dir,
                learning_rate=args.lr
            )
            
            # Batch Size und andere Parameter automatisch setzen
            hardware_config = detector.get_optimal_config()
            if args.batch_size:
                config.batch_size = args.batch_size
            else:
                config.batch_size = hardware_config.get('batch_size', 32)
            
            if args.max_seq_len:
                config.max_seq_len = args.max_seq_len
            else:
                config.max_seq_len = hardware_config.get('max_seq_len', 1024)
            
            config.update_from_hardware(hardware_config)
        
        # Config anzeigen
        config.print_summary()
        
        # Training starten
        trainer = UniversalTrainer(config)
        trainer.train()
    
    elif args.command == 'suggest':
        from uarf import HardwareDetector, ModelSelector
        
        detector = HardwareDetector()
        selector = ModelSelector(detector.specs)
        selector.print_suggestions(args.task)
    
    elif args.command == 'export':
        print("📤 Export-Funktionalität wird entwickelt...")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Format: {args.format}")
        print(f"Quantisierung: {args.quantization}")
        # TODO: Export-Logik implementieren
    
    elif args.command == 'detect':
        from uarf import HardwareDetector
        
        detector = HardwareDetector()
        
        if args.json:
            import json
            specs_dict = {
                'platform': detector.specs.platform,
                'cpu_count': detector.specs.cpu_count,
                'ram_total_gb': detector.specs.ram_total,
                'ram_available_gb': detector.specs.ram_available,
                'gpu_available': detector.specs.gpu_available,
                'gpu_name': detector.specs.gpu_name,
                'gpu_vram_gb': detector.specs.gpu_vram,
                'is_mobile': detector.specs.is_mobile,
                'is_colab': detector.specs.is_colab,
                'is_cluster': detector.specs.is_cluster
            }
            print(json.dumps(specs_dict, indent=2))
        else:
            detector.print_summary()


if __name__ == '__main__':
    main()
