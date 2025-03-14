import subprocess
import sys
from typing import Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

try:
    from llama_cpp import Llama
except ImportError:
    user_input = input("The 'llama-cpp-python' library is required. Install it now? [y/N]: ")
    if user_input.lower() == "y":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
            from llama_cpp import Llama
        except subprocess.CalledProcessError:
            print("Failed to install 'llama-cpp-python'. Please install it manually using 'pip install llama-cpp-python'.")
            sys.exit(1)
    else:
        print("The required 'ollama' library is not installed.")
        sys.exit(1)


class LlamaCppEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "nomic-embed-text"
        self.config.embedding_dims = self.config.embedding_dims or 512

        # self.client = Client(host=self.config.ollama_base_url)
        self.client = Llama(model_path=self.config.model, embedding=True, n_ctx=self.config.embedding_dims)

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Ollama.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        response = self.client.create_embedding(input=text, model=self.config.model)

        return response["data"][0]["embedding"]
