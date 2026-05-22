import yaml
from string import Template
import os

class PromptManager:
    """
    Manages structured prompts to ensure reproducibility.
    Loads templates from prompts.yaml and injects context dynamically.
    """
    def __init__(self, prompt_file='configs/prompts.yaml'):
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found at {prompt_file}")
            
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.templates = yaml.safe_load(f)

    def get_prompt(self, template_type, **kwargs):
        """
        Injects dynamic data (query, paths, etc.) into the template.
        Uses string.Template for secure variable replacement.
        """
        # Select category: e.g., 'generation_templates'
        category = 'generation_templates' if 'evidence' in kwargs or 'paths' in kwargs else 'system_prompts'
        
        # Get specific template string
        template_str = self.templates.get(category, {}).get(template_type)
        
        if not template_str:
            raise ValueError(f"Template '{template_type}' not found in {category}")

        # Perform injection
        return Template(template_str).safe_substitute(kwargs)

# manager = PromptManager()
# formatted_prompt = manager.get_prompt('evidence_aware_prompt', query="What is...", paths="[A->B->C]")
# print(formatted_prompt)
