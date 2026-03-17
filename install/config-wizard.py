#!/usr/bin/env python3
"""
Interactive Configuration Wizard for AutoResearch
Guides users through system setup with templates and defaults
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
from dataclasses import dataclass


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class ConfigChoice:
    """Represents a user choice option"""
    label: str
    value: Any
    description: str = ""


class ConfigWizard:
    """Interactive configuration wizard"""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.config_dir = repo_root / 'config'
        self.config = {}
        self.current_stage = 0

    def print_header(self, title: str):
        """Print formatted header"""
        width = 60
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{title.center(width)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*width}{Colors.ENDC}\n")

    def print_choice(self, index: int, option: ConfigChoice):
        """Print a choice option"""
        marker = "●" if index == 0 else "○"
        print(f"{Colors.BOLD}{marker}{Colors.ENDC} {option.label}")
        if option.description:
            print(f"  {Colors.YELLOW}{option.description}{Colors.ENDC}")

    def ask_choice(self, question: str, options: List[ConfigChoice],
                   default: int = 0) -> ConfigChoice:
        """Ask user to select from multiple choices"""
        print(f"{Colors.BOLD}{question}{Colors.ENDC}")
        for i, option in enumerate(options):
            self.print_choice(i, option)

        while True:
            response = input(f"\n{Colors.GREEN}Select [{default + 1}-{len(options)}] (default {default + 1}): {Colors.ENDC}").strip()
            if not response:
                return options[default]
            try:
                choice_index = int(response) - 1
                if 0 <= choice_index < len(options):
                    return options[choice_index]
                print(f"{Colors.RED}Invalid choice. Try again.{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Please enter a number.{Colors.ENDC}")

    def ask_text(self, question: str, default: str = "") -> str:
        """Ask user for text input"""
        prompt = f"{Colors.GREEN}{question}{Colors.ENDC}"
        if default:
            prompt += f" [{Colors.YELLOW}{default}{Colors.ENDC}]"
        prompt += ": "

        response = input(prompt).strip()
        return response if response else default

    def ask_number(self, question: str, default: int = 1, min_val: int = 1) -> int:
        """Ask user for numeric input"""
        while True:
            response = self.ask_text(question, str(default))
            try:
                value = int(response)
                if value >= min_val:
                    return value
                print(f"{Colors.RED}Must be >= {min_val}{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.RED}Please enter a valid number{Colors.ENDC}")

    def stage_system_profile(self) -> Dict:
        """Stage 1: System Profile Configuration"""
        self.print_header("System Configuration")

        institution = self.ask_text("Research institution/company name", "Independent Researcher")
        username = self.ask_text("Your name", "Researcher")
        email = self.ask_text("Email (for research attribution)", "")

        print(f"\n{Colors.BOLD}GPU Configuration:{Colors.ENDC}")
        gpu_choice = self.ask_choice(
            "How should agents use GPU?",
            [
                ConfigChoice("Use all available GPUs", "all",
                            "Distribute agents across all detected GPUs"),
                ConfigChoice("Use specific GPU(s)", "specific",
                            "Manually select which GPU(s) to use"),
                ConfigChoice("CPU only (testing)", "cpu",
                            "Run on CPU (slower, for testing)"),
            ]
        )

        gpu_config = {"mode": gpu_choice.value}
        if gpu_choice.value == "specific":
            gpu_ids = self.ask_text("GPU indices (e.g. 0,1,2)", "0")
            gpu_config["indices"] = [int(x.strip()) for x in gpu_ids.split(',')]

        cache_location = Path.home() / '.cache' / 'autoresearch'
        custom_cache = self.ask_text("Cache directory", str(cache_location))

        return {
            'institution': institution,
            'username': username,
            'email': email,
            'gpu': gpu_config,
            'cache_dir': custom_cache,
        }

    def stage_api_services(self) -> Dict:
        """Stage 2: API Service Configuration"""
        self.print_header("API Service Configuration")

        print(f"{Colors.BOLD}Which AI services will you use?{Colors.ENDC}")
        print("(You can use multiple services)\n")

        services = {}

        # Local services
        use_local = input(f"{Colors.YELLOW}Use local models (Ollama/vLLM)? (y/n, default y): {Colors.ENDC}").lower()
        if use_local != 'n':
            services['local_ollama'] = {
                'enabled': True,
                'base_url': 'http://localhost:11434/v1',
                'model': 'mistral',
                'api_key': 'none',
            }

        # OpenAI
        use_openai = input(f"{Colors.YELLOW}Use OpenAI API? (y/n, default y): {Colors.ENDC}").lower()
        if use_openai != 'n':
            api_key = self.ask_text("OpenAI API key (or press Enter to set later)")
            services['openai'] = {
                'enabled': bool(api_key),
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-4-turbo',
                'api_key': api_key if api_key else 'your key here',
            }

        # Anthropic
        use_anthropic = input(f"{Colors.YELLOW}Use Anthropic Claude API? (y/n): {Colors.ENDC}").lower()
        if use_anthropic == 'y':
            api_key = self.ask_text("Anthropic API key (or press Enter to set later)")
            services['anthropic'] = {
                'enabled': bool(api_key),
                'base_url': 'https://api.anthropic.com',
                'model': 'claude-opus-4.6',
                'api_key': api_key if api_key else 'your key here',
            }

        # Chinese services
        use_chinese = input(f"{Colors.YELLOW}Use Chinese AI services (Deepseek, Qwen)? (y/n): {Colors.ENDC}").lower()
        if use_chinese == 'y':
            services['deepseek'] = {
                'enabled': False,
                'base_url': 'https://api.deepseek.com/v1',
                'model': 'deepseek-v3',
                'api_key': 'your key here',
                'description': 'Uncomment and add key to enable',
            }
            services['qwen'] = {
                'enabled': False,
                'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'model': 'qwen-max',
                'api_key': 'your key here',
                'description': 'Uncomment and add key to enable',
            }

        return services

    def stage_research_template(self) -> Dict:
        """Stage 3: Research Template Selection"""
        self.print_header("Research Template Selection")

        template_choice = self.ask_choice(
            "What will you research?",
            [
                ConfigChoice("Technical Wiki", "technical_wiki",
                            "Continuously grow ML/AI knowledge base"),
                ConfigChoice("Explanatory Wiki", "explanatory_wiki",
                            "Create detailed concept explanations"),
                ConfigChoice("Game Dev Assets", "game_dev_assets",
                            "Generate game development resources"),
                ConfigChoice("Custom Research", "custom",
                            "Define your own research objectives"),
            ]
        )

        template_config = {'template': template_choice.value}

        if template_choice.value == "custom":
            research_focus = self.ask_text("Describe your research focus")
            template_config['custom_focus'] = research_focus

        return template_config

    def stage_deployment_target(self) -> Dict:
        """Stage 4: Deployment Target Configuration"""
        self.print_header("Deployment Target Configuration")

        targets = {}

        use_github = input(f"{Colors.YELLOW}Deploy to GitHub wiki? (y/n, default y): {Colors.ENDC}").lower()
        if use_github != 'n':
            repo_url = self.ask_text("GitHub repo URL (e.g. https://github.com/user/wiki)")
            targets['github_wiki'] = {
                'enabled': bool(repo_url),
                'repo_url': repo_url,
                'branch': 'research-findings',
                'auto_sync': True,
            }

        use_db = input(f"{Colors.YELLOW}Deploy to database (PostgreSQL/MongoDB)? (y/n): {Colors.ENDC}").lower()
        if use_db == 'y':
            db_choice = self.ask_choice(
                "Which database?",
                [
                    ConfigChoice("PostgreSQL", "postgresql", "Structured data"),
                    ConfigChoice("MongoDB", "mongodb", "Flexible documents"),
                ]
            )

            if db_choice.value == "postgresql":
                targets['postgresql'] = {
                    'enabled': True,
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'autoresearch',
                    'username': self.ask_text("Database username", "autoresearch"),
                    'password': self.ask_text("Database password"),
                }

        use_s3 = input(f"{Colors.YELLOW}Use S3-compatible storage for archives? (y/n): {Colors.ENDC}").lower()
        if use_s3 == 'y':
            targets['s3'] = {
                'enabled': True,
                'endpoint': self.ask_text("S3 endpoint URL"),
                'bucket': self.ask_text("Bucket name", "autoresearch-results"),
                'access_key': self.ask_text("Access key"),
                'secret_key': self.ask_text("Secret key"),
            }

        return targets

    def stage_multi_agent_config(self) -> Dict:
        """Stage 5: Multi-Agent Configuration"""
        self.print_header("Multi-Agent Configuration")

        agent_count_choice = self.ask_choice(
            "How many agents for research swarm?",
            [
                ConfigChoice("1 agent (solo mode)", 1, "Single agent, simple setup"),
                ConfigChoice("2 agents", 2, "Basic multi-agent"),
                ConfigChoice("4 agents (recommended)", 4, "Full research swarm"),
                ConfigChoice("Custom number", 0, "Specify your own"),
            ]
        )

        if agent_count_choice.value == 0:
            num_agents = self.ask_number("Number of agents", 4, min_val=1)
        else:
            num_agents = agent_count_choice.value

        return {
            'num_agents': num_agents,
            'collaboration_protocol': 'round-robin-debate',
            'specialization': True,
        }

    def stage_retention_policy(self) -> Dict:
        """Stage 6: Data Retention Configuration"""
        self.print_header("Data Retention Configuration")

        print(f"{Colors.BOLD}How long to keep research data?{Colors.ENDC}\n")

        hot_days = self.ask_number("Hot storage (immediate access) - days", 1)
        warm_days = self.ask_number("Warm storage (compressed cloud) - days", 30)
        cold_days = self.ask_number("Cold storage (archival) - days", 180)

        return {
            'hot_storage_days': hot_days,
            'warm_storage_days': warm_days,
            'cold_storage_days': cold_days,
            'compress_after_days': warm_days // 2,
            'summarize_before_archive': True,
        }

    def generate_yaml_configs(self, config: Dict) -> Dict:
        """Generate YAML configuration files from user choices"""
        configs = {}

        # System config
        configs['system.yaml'] = {
            'system': config.get('system_profile', {}),
            'services': config.get('api_services', {}),
            'research': config.get('research_template', {}),
            'deployment': config.get('deployment_targets', {}),
            'agents': {
                'num_agents': config.get('multi_agent_config', {}).get('num_agents', 1),
                'collaboration_protocol': 'round-robin-debate',
            },
            'retention': config.get('retention_policy', {}),
        }

        return configs

    def save_configs(self, configs: Dict) -> bool:
        """Save configuration files to disk"""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)

            # Save main system config
            system_config_path = self.config_dir / 'system.yaml'
            with open(system_config_path, 'w') as f:
                yaml.dump(configs['system.yaml'], f, default_flow_style=False, sort_keys=False)

            print(f"{Colors.GREEN}✓ Saved: {system_config_path}{Colors.ENDC}")

            # Save service templates
            services_dir = self.config_dir / 'services'
            services_dir.mkdir(exist_ok=True)

            # OpenAI template
            openai_template = self.config_dir / 'services' / 'openai.yaml'
            openai_template.write_text("""# OpenAI Services Configuration
# Uncomment the services you want to use

openai:
  enabled: true
  provider: openai
  base_url: "https://api.openai.com/v1"
  model: "gpt-4-turbo"
  api_key: "your key here"
  max_tokens: 8000
  rate_limit_tokens_per_minute: 10000
  temperature: 0.7
  description: "Primary reasoning model for research direction"

openai_gpt35:
  enabled: false
  # base_url: "https://api.openai.com/v1"
  # model: "gpt-3.5-turbo"
  # api_key: "your key here"
  # description: "Fallback for cost optimization"
""")
            print(f"{Colors.GREEN}✓ Saved: {openai_template}{Colors.ENDC}")

            # Anthropic template
            anthropic_template = self.config_dir / 'services' / 'anthropic.yaml'
            anthropic_template.write_text("""# Anthropic Claude Services
# Uncomment to enable

# anthropic_opus:
#   enabled: true
#   provider: anthropic
#   base_url: "https://api.anthropic.com"
#   model: "claude-opus-4.6"
#   api_key: "your key here"
#   max_tokens: 8000
#   description: "Advanced reasoning for complex research questions"

# anthropic_sonnet:
#   enabled: false
#   provider: anthropic
#   base_url: "https://api.anthropic.com"
#   model: "claude-sonnet-4"
#   api_key: "your key here"
#   description: "Fast synthesis and summary generation"
""")
            print(f"{Colors.GREEN}✓ Saved: {anthropic_template}{Colors.ENDC}")

            # Chinese APIs template
            chinese_template = self.config_dir / 'services' / 'chinese-apis.yaml'
            chinese_template.write_text("""# Chinese AI Services (US-accessible APIs)
# Uncomment and add keys to enable

# deepseek:
#   enabled: false
#   provider: deepseek
#   base_url: "https://api.deepseek.com/v1"
#   model: "deepseek-v3"
#   api_key: "your key here"

# qwen:
#   enabled: false
#   provider: aliyun
#   base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   model: "qwen-max"
#   api_key: "your key here"

# baichuan:
#   enabled: false
#   provider: baichuan
#   base_url: "https://api.baichuan-ai.com/v1"
#   model: "Baichuan3-Turbo"
#   api_key: "your key here"
""")
            print(f"{Colors.GREEN}✓ Saved: {chinese_template}{Colors.ENDC}")

            # Local services template
            local_template = self.config_dir / 'services' / 'local-services.yaml'
            local_template.write_text("""# Local LLM Services (no API keys needed)

ollama:
  enabled: false  # Uncomment and ensure ollama is running
  provider: ollama
  base_url: "http://localhost:11434/v1"
  model: "mistral"
  rate_limit_tokens_per_minute: "unlimited"
  description: "Local open-source model. Setup: ollama pull mistral"

vllm:
  enabled: false  # Uncomment and ensure vLLM is running
  provider: vllm
  base_url: "http://localhost:8000/v1"
  model: "mistral-7b"
  gpu_device: 0
  description: "Faster local inference. Setup: vllm serve mistral-7b"
""")
            print(f"{Colors.GREEN}✓ Saved: {local_template}{Colors.ENDC}")

            return True
        except Exception as e:
            print(f"{Colors.RED}✗ Error saving configs: {e}{Colors.ENDC}")
            return False

    def run(self) -> Dict:
        """Run complete configuration wizard"""
        self.print_header("AutoResearch Configuration Wizard")
        print(f"{Colors.BOLD}This wizard will guide you through system setup.{Colors.ENDC}")
        print(f"You can edit configuration files later in: {Colors.CYAN}{self.config_dir}{Colors.ENDC}\n")

        input(f"{Colors.YELLOW}Press Enter to begin...{Colors.ENDC}")

        # Run stages
        self.config['system_profile'] = self.stage_system_profile()
        self.config['api_services'] = self.stage_api_services()
        self.config['research_template'] = self.stage_research_template()
        self.config['deployment_targets'] = self.stage_deployment_target()
        self.config['multi_agent_config'] = self.stage_multi_agent_config()
        self.config['retention_policy'] = self.stage_retention_policy()

        # Generate and save configs
        self.print_header("Saving Configuration")
        yaml_configs = self.generate_yaml_configs(self.config)
        self.save_configs(yaml_configs)

        # Summary
        self.print_header("Configuration Complete!")
        print(f"{Colors.GREEN}✓ Configuration saved successfully!{Colors.ENDC}\n")
        print(f"{Colors.BOLD}Next steps:{Colors.ENDC}")
        print(f"1. Edit API keys in: {Colors.CYAN}config/services/{Colors.ENDC}")
        print(f"2. Review main config: {Colors.CYAN}config/system.yaml{Colors.ENDC}")
        print(f"3. Start research: {Colors.CYAN}ar start --agents {self.config['multi_agent_config']['num_agents']}{Colors.ENDC}")
        print(f"4. Monitor progress: {Colors.CYAN}ar status{Colors.ENDC}\n")

        return self.config


def main():
    """Entry point for config wizard"""
    repo_root = Path(__file__).parent.parent
    wizard = ConfigWizard(repo_root)
    wizard.run()


if __name__ == '__main__':
    main()
