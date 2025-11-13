import base64
import hashlib
import json
import re
import time
import traceback
from typing import Optional, Tuple, Union, Type, List, Any, TYPE_CHECKING, Dict
import httpx
from pydantic import BaseModel

from litellm._uuid import uuid
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client, HTTPHandler, \
    _get_httpx_client
from litellm.llms.gigachat.common_utils import GigaChatError
import litellm
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders, ModelResponseStream
from litellm.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.llms.openai import ChatCompletionToolParam

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

class GigaChatCustomStreamWrapper(CustomStreamWrapper):

    def handle_gigachat_chunk(self, chunk: Any):
        """
        Process a raw SSE chunk. Supports cases where a single chunk contains multiple `data:` lines.
        Returns one event per call and buffers the rest in an internal queue.
        """
        # If there are already parsed events in the queue - return the next one
        if isinstance(chunk, str):
            for raw_line in chunk.splitlines():
                if not raw_line:
                    continue
                name, _, line_str = raw_line.partition(": ")
                if name != "data":
                    continue
                if line_str == "[DONE]":
                    return {}
                else:
                    return json.loads(line_str)
        return None

    def chunk_creator(self, chunk: Any):
        try:
            data_json = self.handle_gigachat_chunk(chunk)
            # If the chunk is not a GigaChat data line, skip
            if data_json is None:
                return None

            # Handle stream terminator
            if data_json == {}:
                # Mark finish and let parent finalize the last chunk
                self.received_finish_reason = "stop"
                return self.finish_reason_handler()

            # Normal data event
            model_response = ModelResponseStream(**data_json)
            # Extract text content from delta
            delta_content = ""
            try:
                delta_content = (
                    model_response.choices[0].delta.get("content", "") or ""
                )
            except Exception:
                delta_content = ""

            response_obj: Dict[str, Any] = {
                "text": delta_content,
                "is_finished": False,
                "finish_reason": "",
            }

            completion_obj: Dict[str, Any] = {"content": delta_content}

            return self.return_processed_chunk_logic(
                completion_obj=completion_obj,
                model_response=model_response,  # type: ignore
                response_obj=response_obj,
            )

        except StopIteration:
            raise StopIteration
        except Exception as e:
            traceback.format_exc()
            setattr(e, "message", str(e))
            raise exception_type(
                model=self.model,
                custom_llm_provider=self.custom_llm_provider,
                original_exception=e,
            )
class GigaChatConfig(BaseConfig):
    """
    Configuration for GigaChat API integration.

    Supports both direct API key authentication and OAuth flow.

    Environment variables:
    - GIGACHAT_API_KEY: Direct API key (fallback for OAuth)
    - GIGACHAT_CREDENTIALS: Client credentials in format "client_id:client_secret" or single API key
    - GIGACHAT_SCOPE: OAuth scope (default: GIGACHAT_API_PERS)
    - GIGACHAT_USERNAME: Username for OAuth (alternative to credentials)
    - GIGACHAT_PASSWORD: Password for OAuth (alternative to credentials)
    - GIGACHAT_AUTH_URL: OAuth endpoint URL (default: https://ngw.devices.sberbank.ru:9443/api/v2/oauth)
    - GIGACHAT_API_BASE: API base URL (default: https://gigachat.devices.sberbank.ru/api/v1/chat/completions)

    Usage:
        # Method 1: With credentials and scope
        export GIGACHAT_CREDENTIALS="client_id:client_secret"
        export GIGACHAT_SCOPE="GIGACHAT_API_PERS"

        # Method 2: With username and password
        export GIGACHAT_USERNAME="your_username"
        export GIGACHAT_PASSWORD="your_password"
        export GIGACHAT_SCOPE="GIGACHAT_API_PERS"

        # Method 3: Direct API key (legacy)
        export GIGACHAT_API_KEY="your_api_key"

        # Custom URLs
        export GIGACHAT_AUTH_URL="https://your-custom-auth.com/oauth"
        export GIGACHAT_API_BASE="https://your-custom-api.com/v1"
    """

    def __init__(self):
        super().__init__()
        self._token_cache: dict[str, Any] = {"token": None, "expires_at": 0}

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "gigachat"

    def get_supported_openai_params(self, model: str) -> list[str]:

        params = [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "top_p",
            "temperature",
            "frequency_penalty",
            "tools",
            "tool_choice",
            "functions",
            "response_format",
        ]
        if "reason" in model.lower():
            params.append("reasoning_effort")
        return params

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
        Validate and prepare environment for GigaChat API calls
        """
        if api_key is None:
            api_key = litellm.get_secret_str("GIGACHAT_API_KEY")

        # If no API key, try to get one via OAuth
        if api_key is None:
            api_key = self._get_oauth_token()

        if api_key is None:
            raise ValueError("GIGACHAT_API_KEY not found and OAuth credentials not provided")

        if api_base is None:
            api_base = litellm.get_secret_str("GIGACHAT_API_BASE") or "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "GigaChat-python-lib",
            **headers,
        }

        return headers

    def _is_token_expired(self) -> bool:
        """Check if cached OAuth token is expired or missing."""
        now = time.time()
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

        auth_url = litellm.get_secret_str("GIGACHAT_AUTH_URL") or \
                   "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        try:
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
                response = httpx.post(auth_url, headers=headers, data=data, timeout=30, verify=False)
                response.raise_for_status()
                data = response.json()
                token = data.get("access_token")
                expires_at = float(data.get("expires_at", 0))

            if not token:
                raise ValueError("OAuth did not return a token")

            # Cache the token
            self._token_cache["token"] = token
            self._token_cache["expires_at"] = expires_at or (time.time() + 3600)

            return token

        except Exception as e:
            print(f"[GigaChat] OAuth token fetch failed: {e}")
            return None

    def _transform_messages(self, messages: List[Dict], headers: dict) -> List[Dict]:
        """Transforms messages to GigaChat format."""
        transformed_messages = []
        attachment_count = 0

        for i, message in enumerate(messages):
            # Remove unused fields
            message.pop("name", None)

            # Normalize roles
            if message["role"] == "developer":
                message["role"] = "system"
            elif message["role"] == "system" and i > 0:
                message["role"] = "user"
            elif message["role"] == "tool":
                message["role"] = "function"
                try:
                    json.loads(message.get("content", ""))
                except json.JSONDecodeError:
                    message["content"] = json.dumps(
                        message.get("content", ""), ensure_ascii=False
                    )

            # Handle content
            if message.get("content") is None:
                message["content"] = ""

            # Handle tool_calls
            if "tool_calls" in message and message["tool_calls"]:
                message["function_call"] = message["tool_calls"][0]["function"]
                try:
                    message["function_call"]["arguments"] = json.loads(
                        message["function_call"]["arguments"]
                    )
                except json.JSONDecodeError as e:
                    pass
            if isinstance(message["content"], list):
                texts, attachments = self._process_content_parts(message["content"], headers)
                message["content"] = "\n".join(texts)
                message["attachments"] = attachments
                attachment_count += len(attachments)

            transformed_messages.append(message)
        if attachment_count > 10:
            self._limit_attachments(transformed_messages, max_total_attachments=10)
        return transformed_messages

    def _limit_attachments(self, messages: List[Dict], max_total_attachments: int = 10) -> None:
        """
        Limits the total number of attachments across all messages to max_total_attachments.
        Trims extra attachments while iterating messages in order of appearance.
        """
        remaining = max_total_attachments
        for msg in messages:
            attachments = msg.get("attachments")
            if not isinstance(attachments, list) or len(attachments) == 0:
                continue
            if remaining <= 0:
                msg["attachments"] = []
                continue
            if len(attachments) > remaining:
                msg["attachments"] = attachments[:remaining]
                remaining = 0
            else:
                remaining -= len(attachments)

    def upload_image(self, image_url: str, headers: dict) -> Optional[str]:
        """Uploads an image for use with GigaChat and returns a file_id (if implemented)."""
        base64_matches = re.search(r"data:(.+);(.+),(.+)", image_url)
        try:
            if not base64_matches:
                response = httpx.get(image_url, timeout=30)
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    return None
            else:
                content_type, type_, image_str = base64_matches.groups()
                if type_ != "base64":
                    return None
                # Validate base64 — do not upload, only check correctness
                _ = base64.b64decode(image_str)
            # Stub: return None if upload integration is not configured
            return None
        except:
            pass

    def _process_content_parts(
            self, content_parts: List[Dict], headers: dict
    ) -> Tuple[List[str], List[str]]:
        """Processes content parts (text and images)."""
        texts = []
        attachments = []

        for content_part in content_parts:
            if content_part.get("type") == "text":
                texts.append(content_part.get("text", ""))
            elif (
                    content_part.get("type") == "image_url"
                    and content_part.get("image_url")
            ):
                file_id = self.upload_image(
                    content_part["image_url"]["url"],
                    headers
                )
                if file_id:
                    attachments.append(file_id)

        # Limit number of images
        if len(attachments) > 2:
            attachments = attachments[:2]

        return texts, attachments

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style request to GigaChat format.
        """
        # Transform messages to GigaChat format
        gigachat_messages = self._transform_messages(messages, headers)
        # Build request body
        request_body = {
            "model": model,
            "messages": gigachat_messages,
        }

        for param, value in optional_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                request_body["max_tokens"] = value
            elif param == "temperature":
                request_body["temperature"] = value
            elif param == "top_p":
                request_body["top_p"] = value
            elif param == "stream":
                request_body["stream"] = value
            elif param == "tools" or param == "functions":
                gigachat_tools = self._construct_gigachat_tool(tools=optional_params["tools"])
                request_body["functions"] = gigachat_tools
            elif param == "response_format":
                if value.get("json_schema") and value["json_schema"].get("schema"):
                    request_body["response_format"] = {"type": "json_schema",
                                                       **value["json_schema"]}
        return request_body

    def _process_function_call(self, message: dict):
        arguments = json.dumps(
            message["function_call"]["arguments"],
            ensure_ascii=False,
        )
        function_call = {
            "name": message["function_call"]["name"],
            "arguments": arguments,
        }
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4()}",
                "type": "function",
                "function": function_call,
            }
        ]
        if message.get("finish_reason") == "function_call":
            message["finish_reason"] = "tool_calls"

    def transform_response(
            self,
            model: str,
            raw_response: httpx.Response,
            model_response: "ModelResponse",
            logging_obj: LiteLLMLoggingObj,
            request_data: dict,
            messages: List[AllMessageValues],
            optional_params: dict,
            litellm_params: dict,
            encoding: Any,
            api_key: Optional[str] = None,
            json_mode: Optional[bool] = None,
    ) -> "ModelResponse":
        """
        Transform GigaChat response to OpenAI format
        """
        try:
            response_json = raw_response.json()
            message = response_json["choices"][0]["message"]
            if "function_call" in message:
                message["function_call"]["arguments"] = json.dumps(message["function_call"]["arguments"])
                self._process_function_call(message)
        except Exception as e:
            raise ValueError(f"Failed to parse GigaChat response as JSON: {e}")

        return ModelResponse(**response_json)

    def _construct_gigachat_tool(self, tools: Optional[list] = None):
        if tools is None:
            tools = []
        gigachat_tools = []
        for tool in tools:
            gigachat_tool = self._translate_openai_tool_to_gigachat(tool)
            gigachat_tools.append(gigachat_tool)

        return gigachat_tools

    def _translate_openai_tool_to_gigachat(self, openai_tool: dict):
        """GigaChat tool looks like this:
        {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                }"""

        # OpenAI tools look like this
        """
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
        """
        gigachat_tool = openai_tool.copy()
        if "function" in gigachat_tool:
            gigachat_tool = gigachat_tool["function"]
        gigachat_tool.pop("type", None)

        return gigachat_tool

    def get_json_schema_from_pydantic_object(
        self, response_format: Optional[Union[Type[BaseModel], dict]]
    ) -> Optional[dict]:

        json_schema_openai = type_to_response_format_param(response_format=response_format)
        return {"type": "json_schema", **json_schema_openai["json_schema"]}

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "frequency_penalty":
                optional_params["repetition_penalty"] = value
            if (
                param == "response_format"
                and isinstance(value, dict)
                and value.get("type") == "json_schema"
            ):
                optional_params["response_format"] = value
            if param == "tools" or param == "functions":
                optional_params["functions"] = value

        non_default_params.pop("tools", None)
        non_default_params.pop("functions", None)
        return optional_params

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return True

    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[AsyncHTTPHandler] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> "CustomStreamWrapper":
        if client is None or isinstance(client, HTTPHandler):
            client = get_async_httpx_client(llm_provider=LlmProviders.GIGACHAT, params={"ssl_verify": False})

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
                stream=True,
                logging_obj=logging_obj,

            )
        except httpx.HTTPStatusError as e:
            raise GigaChatError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise GigaChatError(status_code=response.status_code, message=response.text)

        completion_stream = response.aiter_text()
        streaming_response = GigaChatCustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )
        return streaming_response

    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> "CustomStreamWrapper":
        if client is None or isinstance(client, HTTPHandler):
            client = _get_httpx_client(params={"ssl_verify": False})

        try:
            response = client.post(
                api_base,
                headers=headers,
                data=json.dumps(data),
                stream=True,
                logging_obj=logging_obj,

            )
        except httpx.HTTPStatusError as e:
            raise GigaChatError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise GigaChatError(status_code=response.status_code, message=response.text)

        completion_stream = response.iter_text()
        streaming_response = GigaChatCustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider=custom_llm_provider,
            logging_obj=logging_obj,
        )

        return streaming_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..common_utils import GigaChatError
        return GigaChatError(status_code=status_code, message=error_message, headers=headers)
