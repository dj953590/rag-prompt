import sys

sys.path.insert(0, "../../")

from promptwizard.glue.promptopt.instantiate import GluePromptOpt
import os

from dotenv import load_dotenv

load_dotenv(override=True)

if not os.path.exists("data"):
    os.mkdir("data")

path_to_config = "configs"
promptopt_config_path = os.path.join(path_to_config, "promptopt_config.yaml")
setup_config_path = os.path.join(path_to_config, "setup_config.yaml")


gp = GluePromptOpt(promptopt_config_path,
                   setup_config_path,
                   dataset_jsonl=None,
                   data_processor=None)

best_prompt, expert_profile = gp.get_best_prompt(use_examples=False, run_without_train_examples=True,
                                                 generate_synthetic_examples=False)
print(f"Best prompt: {best_prompt} \nExpert profile: {expert_profile}")
