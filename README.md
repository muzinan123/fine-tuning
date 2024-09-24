# Fine-Tuning

This repository contains code and resources for fine-tuning various language models, including ChatGLM3, LLaMA 2, and others. It provides tools and scripts for pre-training, fine-tuning, and deploying these models.

## Features

- Support for multiple language models (ChatGLM3, LLaMA 2)
- Pre-training scripts
- Fine-tuning scripts
- DeepSpeed integration for efficient training
- Web demo for testing fine-tuned models

## Directory Structure
![12](https://github.com/user-attachments/assets/55f0d424-e960-4eb7-ba5a-2baa7ce4a553)

- `chatglm3/`: ChatGLM3 specific code and resources
- `data/`: Training and evaluation data
- `deepspeed/`: DeepSpeed configuration and scripts
- `llama2/`: LLaMA 2 specific code and resources
- `pretraining/`: Pre-training scripts and configurations
- `web_demo/`: Web interface for demonstrating fine-tuned models

## Requirements

See `requirements.txt` for a list of Python dependencies.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/muzinan123/fine-tuning.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Follow the instructions in each model-specific directory for pre-training, fine-tuning, and deployment.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [ChatGLM3](https://github.com/THUDM/ChatGLM3)
- [LLaMA 2](https://github.com/facebookresearch/llama)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

## Disclaimer

This project is for research purposes only. Please ensure you comply with the licenses of the original models and datasets used in this project.
