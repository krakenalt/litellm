from typing import Any, List, Optional, Union

import httpx

import litellm
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

        # If no API key, try to get one via OAuth
        if api_key is None:
            api_key = self._get_oauth_token()

        if api_key is None:
            raise ValueError("GIGACHAT_API_KEY not found and OAuth credentials not provided")

        # Set default API base if not provided
        if api_base is None:
            api_base = litellm.get_secret_str("GIGACHAT_API_BASE") or "https://gigachat.devices.sberbank.ru/api/v1/embeddings"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def _get_oauth_token(self) -> Optional[str]:
        """
        Get OAuth token using credentials+scope or username+password
        """
        import litellm

        auth_url = litellm.get_secret_str("GIGACHAT_AUTH_URL") or "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        try:
            import httpx

            # Method 1: Using credentials and scope
            credentials = litellm.get_secret_str("GIGACHAT_CREDENTIALS") or litellm.get_secret_str("GIGACHAT_API_KEY")
            scope = litellm.get_secret_str("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS"

            if credentials:
                data = {
                    "scope": scope
                }
                # Try credentials first (client_id:client_secret format)
                if ":" in credentials:
                    client_id, client_secret = credentials.split(":", 1)
                    data.update({
                        "client_id": client_id,
                        "client_secret": client_secret
                    })
                else:
                    # Single credential (API key)
                    data.update({
                        "client_id": credentials,
                        "client_secret": ""  # Empty for API key auth
                    })

            else:
                # Method 2: Using username and password
                username = litellm.get_secret_str("GIGACHAT_USERNAME")
                password = litellm.get_secret_str("GIGACHAT_PASSWORD")

                if not username or not password:
                    return None

                data = {
                    "scope": scope,
                    "username": username,
                    "password": password
                }

            response = httpx.post(auth_url, data=data, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            return token_data.get("access_token")

        except Exception as e:
            print(f"Failed to get OAuth token: {e}")
            return None

    def transform_embedding_request(
        self,
        model: str,
        input: List[AllEmbeddingInputValues],
        optional_params: dict,
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

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..common_utils import GigaChatError
        return GigaChatError(status_code=status_code, message=error_message, headers=headers)
