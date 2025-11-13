import re
import time
import uuid
from typing import Any, List, Optional, Union
from urllib.parse import urljoin

import httpx
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues, AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse


class GigaChatEmbeddingConfig(BaseEmbeddingConfig):
    """
    Transformations for gigachat /embeddings endpoint
    """

    def __init__(self):
        super().__init__()
        self._token_cache: dict[str, Any] = {"token": None, "expires_at": 0}

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "gigachat"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        match = re.search(r'/v(\d+)/', api_base)
        if not match:
            api_base = urljoin(api_base, "v1/embeddings")
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
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
            raise ValueError(
                "GIGACHAT_API_KEY not found and OAuth credentials not provided"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "GigaChat-python-lib",
            **headers,
        }

        return headers

    @staticmethod
    def _check_timestamp_unit(timestamp):
        """In login+password auth expires_at is in seconds, while in scope+cred in milliseconds"""
        if len(str(timestamp)) == 10:
            return "seconds"
        else:
            return "milliseconds"

    def _is_token_expired(self) -> bool:
        """Check if cached OAuth token is expired or missing."""
        now = (
            time.time()
            if self._check_timestamp_unit(self._token_cache["expires_at"]) == "seconds"
            else time.time() * 1000
        )
        return not self._token_cache["token"] or now >= self._token_cache["expires_at"]

    def _get_oauth_token(self) -> Optional[str]:
        """
        Get or refresh OAuth token using either:
        - username/password (returns tok + exp)
        - client credentials (returns access_token + expires_at)
        """

        # Reuse cached token if valid
        if not self._is_token_expired():
            return self._token_cache["token"]

        import httpx
        import litellm

        auth_url = (
            litellm.get_secret_str("GIGACHAT_AUTH_URL")
            or "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        )

        username = litellm.get_secret_str("GIGACHAT_USERNAME")
        password = litellm.get_secret_str("GIGACHAT_PASSWORD")
        scope = litellm.get_secret_str("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS"
        credentials = litellm.get_secret_str("GIGACHAT_CREDENTIALS")

        if username and password:
            # Username/password OAuth flow
            response = httpx.post(
                auth_url,
                auth=(username, password),
                timeout=30,
                verify=False,
            )
            response.raise_for_status()
            data = response.json()
            token = data.get("tok")
            expires_at = float(data.get("exp", 0))

        else:
            # Client credentials flow
            if not credentials:
                raise ValueError("Missing GIGACHAT_CREDENTIALS or username/password")
            headers = {
                "User-Agent": "GigaChat-python-lib",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {credentials}",
            }
            data = {"scope": scope}
            response = httpx.post(
                auth_url, headers=headers, data=data, timeout=30, verify=False
            )
            response.raise_for_status()
            data = response.json()
            token = data.get("access_token")
            expires_at = float(data.get("expires_at", 0))

        if not token:
            raise ValueError("OAuth did not return a token")

        # Cache the token
        self._token_cache["token"] = token
        self._token_cache["expires_at"] = expires_at or (time.time() + 1800)

        return token

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style embedding request to GigaChat format
        """
        if isinstance(input, list) and (
            isinstance(input[0], list) or isinstance(input[0], int)
        ):
            raise ValueError("Input must be a list of strings")
        request_body = {
            "model": model,
            "input": input,
        }

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
            print(response_json)
        except Exception as e:
            raise ValueError(
                f"Failed to parse GigaChat embedding response as JSON: {raw_response.text}, Error: {str(e)}"
            )
        return EmbeddingResponse(**response_json)

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GigaChat embeddings
        """
        return ["encoding_format"]

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
        return {}

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..common_utils import GigaChatError

        return GigaChatError(
            status_code=status_code, message=error_message, headers=headers
        )
