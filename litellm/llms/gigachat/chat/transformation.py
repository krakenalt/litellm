from typing import Optional, Tuple, Union, Type, List, Any

import httpx
from pydantic import BaseModel

from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import ModelResponse


class GigaChatConfig(BaseConfig):

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
        if "reason" in model:
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

        if api_key is None:
            raise ValueError("GIGACHAT_API_KEY not found in environment variables")

        if api_base is None:
            api_base = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style request to GigaChat format
        """
        # Transform messages to GigaChat format
        gigachat_messages = []
        for message in messages:
            if message.get("role") == "system":
                # GigaChat might handle system messages differently
                gigachat_messages.append({
                    "role": "user",  # or "system" depending on API
                    "content": message.get("content", "")
                })
            else:
                gigachat_messages.append({
                    "role": message.get("role", "user"),
                    "content": message.get("content", "")
                })

        # Build request body
        request_body = {
            "model": model,
            "messages": gigachat_messages,
        }

        # Add optional parameters
        for param, value in optional_params.items():
            if param == "max_tokens":
                request_body["max_tokens"] = value
            elif param == "temperature":
                request_body["temperature"] = value
            elif param == "top_p":
                request_body["top_p"] = value
            elif param == "stream":
                request_body["stream"] = value
            elif param == "functions":
                request_body["functions"] = value
            # Add other parameters as needed based on GigaChat API

        return request_body

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj,
        api_key: Optional[str],
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform GigaChat response to OpenAI format
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse GigaChat response as JSON: {e}")

        # Transform GigaChat response to OpenAI format
        if "choices" in response_json:
            # Already in OpenAI-like format
            choices = response_json["choices"]
        else:
            # Transform from GigaChat format to OpenAI format
            choices = [{
                "message": {
                    "role": "assistant",
                    "content": response_json.get("content", ""),
                },
                "finish_reason": response_json.get("finish_reason", "stop"),
                "index": 0,
            }]

        # Update model response
        model_response.choices = [litellm.utils.convert_to_model_response_object(choice) for choice in choices]

        if "usage" in response_json:
            model_response.usage = litellm.utils.Usage(
                prompt_tokens=response_json["usage"].get("prompt_tokens", 0),
                completion_tokens=response_json["usage"].get("completion_tokens", 0),
                total_tokens=response_json["usage"].get("total_tokens", 0),
            )

        model_response.model = model
        model_response.created = response_json.get("created", 0)

        return model_response

    def _construct_gigachat_tool(self, tools: Optional[list] = None):
        if tools is None:
            tools = []
        gigachat_tools = []
        for tool in tools:
            gigachat_tool = self._translate_openai_tool_to_gigachat(tool)
            gigachat_tools.append(gigachat_tool)

        return gigachat_tools

    def _translate_openai_tool_to_gigachat(self, openai_tool: dict):
        """Gigachat tool looks like this:
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
        gigachat_tool = {**openai_tool["function"]}

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
                if value.get("json_schema") and value["json_schema"].get("schema"):
                    optional_params["format"] = value["json_schema"]["schema"]
            if param == "tools" or param == "functions":
                gigachat_tools = self._construct_gigachat_tool(tools=optional_params["tools"])
                optional_params["functions"] = gigachat_tools

        non_default_params.pop("tools", None)
        return optional_params
