from typing import List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import AllMessageValues, BaseLLMException
from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse
from litellm.utils import ModelResponse

from ..common_utils import GigaChatError


class GigaChatEmbeddingConfig(BaseEmbeddingConfig):
    """
    Transformations for gigachat /embeddings endpoint
    """

    def __init__(self):
        super().__init__()

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "gigachat"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllEmbeddingInputValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate and prepare environment for GigaChat embedding API calls
        """
        import litellm
        if api_key is None:
            api_key = litellm.get_secret_str("GIGACHAT_API_KEY")

        if api_key is None:
            raise ValueError("GIGACHAT_API_KEY not found in environment variables")

        # Set default API base if not provided
        if api_base is None:
            api_base = "https://gigachat.devices.sberbank.ru/api/v1/embeddings"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def transform_request(
        self,
        model: str,
        input: List[AllEmbeddingInputValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style embedding request to GigaChat format
        """
        # Build request body for GigaChat embeddings API
        request_body = {
            "model": model,
            "input": input,
        }

        # Add optional parameters
        for param, value in optional_params.items():
            if param == "encoding_format":
                request_body["encoding_format"] = value
            # Add other parameters as needed based on GigaChat API

        return request_body

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        input: List[AllEmbeddingInputValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
    ) -> EmbeddingResponse:
        """
        Transform GigaChat embedding response to OpenAI format
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse GigaChat embedding response as JSON: {raw_response.text}, Error: {str(e)}")

        # Transform GigaChat response to OpenAI format
        if "data" in response_json:
            # Already in OpenAI-like format
            data = response_json["data"]
        else:
            # Transform from GigaChat format to OpenAI format
            data = [{
                "embedding": response_json.get("embedding", []),
                "index": 0,
            }]

        # Update model response
        model_response.data = data

        if "usage" in response_json:
            model_response.usage = litellm.utils.Usage(
                prompt_tokens=response_json["usage"].get("prompt_tokens", 0),
                completion_tokens=response_json["usage"].get("completion_tokens", 0),
                total_tokens=response_json["usage"].get("total_tokens", 0),
            )

        model_response.model = model

        return model_response

def get_supported_openai_params(self, model: str) -> list:
    """
    Get supported OpenAI parameters for GigaChat embeddings
    """
    return [
        "encoding_format",
        # Add more parameters if GigaChat API supports them
    ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to GigaChat format
        """
        for param, value in non_default_params.items():
            if param == "encoding_format":
                optional_params["encoding_format"] = value

        return optional_params
