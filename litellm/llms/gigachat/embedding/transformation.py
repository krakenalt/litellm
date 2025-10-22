from typing import List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import AllMessageValues, BaseLLMException
from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

from ..common_utils import GigaChatError


class GigaChatEmbeddingConfig(BaseEmbeddingConfig):
    """
    Transformations for gigachat /embeddings endpoint
    """
