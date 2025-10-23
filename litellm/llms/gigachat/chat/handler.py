"""
GigaChat Completion - uses HTTP requests to make calls to GigaChat API

Request/Response transformation is handled in `transformation.py`
"""

import copy
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    Usage,
)

from ...base import BaseLLM
from ..common_utils import GigaChatError
from .transformation import GigaChatConfig


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
    json_mode: bool,
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_aclient

    try:
        response = await client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise GigaChatError(
            status_code=e.response.status_code,
            message=await e.response.aread(),
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise GigaChatError(status_code=500, message=str(e))

    # For now, return the response directly - streaming implementation can be added later
    # if needed based on GigaChat API capabilities
    completion_stream = response

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response=completion_stream,
        additional_args={"complete_input_dict": data},
    )

    return completion_stream, response.headers


def make_sync_call(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
    timeout: Optional[Union[float, httpx.Timeout]],
    json_mode: bool,
) -> Tuple[Any, httpx.Headers]:
    if client is None:
        client = litellm.module_level_client

    try:
        response = client.post(
            api_base, headers=headers, data=data, stream=True, timeout=timeout
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        raise GigaChatError(
            status_code=e.response.status_code,
            message=e.response.read(),
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise GigaChatError(status_code=500, message=str(e))

    if response.status_code != 200:
        response_headers = getattr(response, "headers", None)
        raise GigaChatError(
            status_code=response.status_code,
            message=response.read(),
            headers=response_headers,
        )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="response received",
        additional_args={"complete_input_dict": data},
    )

    return response, response.headers


class GigaChatChatCompletion(BaseLLM):
    def __init__(self) -> None:
        super().__init__()

    async def acompletion_stream_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        client: Optional[AsyncHTTPHandler],
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data: dict,
        json_mode: bool,
        optional_params=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
    ):
        # GigaChat streaming implementation - placeholder for now
        # This would need to be implemented based on GigaChat's streaming API
        pass

    async def acompletion_function(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        encoding,
        api_key,
        logging_obj,
        stream,
        _is_function_call,
        data: dict,
        optional_params: dict,
        json_mode: bool,
        litellm_params: dict,
        provider_config: BaseConfig,
        logger_fn=None,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Union[ModelResponse, "CustomStreamWrapper"]:
        async_handler = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.CUSTOM
        )

        try:
            response = await async_handler.post(
                api_base, headers=headers, json=data, timeout=timeout
            )
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=messages,
                api_key=api_key,
                original_response=str(e),
                additional_args={"complete_input_dict": data},
            )
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            if error_response and hasattr(error_response, "text"):
                error_text = getattr(error_response, "text", error_text)
            raise GigaChatError(
                message=error_text,
                status_code=status_code,
                headers=error_headers,
            )

        return provider_config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_llm_provider: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        acompletion=None,
        logger_fn=None,
        headers={},
        client=None,
    ):
        from litellm.utils import ProviderConfigManager

        optional_params = copy.deepcopy(optional_params)
        stream = optional_params.pop("stream", None)
        json_mode: bool = optional_params.pop("json_mode", False)
        _is_function_call = False
        messages = copy.deepcopy(messages)

        # Validate environment and get headers
        headers = GigaChatConfig().validate_environment(
            api_key=api_key,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        config = ProviderConfigManager.get_provider_chat_config(
            model=model,
            provider=litellm.LlmProviders(custom_llm_provider),
        )
        if config is None:
            raise ValueError(
                f"Provider config not found for model: {model} and provider: {custom_llm_provider}"
            )

        data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key=api_key,
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )
        print_verbose(f"_is_function_call: {_is_function_call}")

        if acompletion is True:
            if stream is True:
                print_verbose("makes async gigachat streaming POST request")
                data["stream"] = stream
                return self.acompletion_stream_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    json_mode=json_mode,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    timeout=timeout,
                    client=(
                        client
                        if client is not None and isinstance(client, AsyncHTTPHandler)
                        else None
                    ),
                )
            else:
                return self.acompletion_function(
                    model=model,
                    messages=messages,
                    data=data,
                    api_base=api_base,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    api_key=api_key,
                    provider_config=config,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    _is_function_call=_is_function_call,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    headers=headers,
                    client=client,
                    json_mode=json_mode,
                    timeout=timeout,
                )
        else:
            ## COMPLETION CALL
            if stream is True:
                data["stream"] = stream
                completion_stream, headers = make_sync_call(
                    client=client,
                    api_base=api_base,
                    headers=headers,
                    data=json.dumps(data),
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                    timeout=timeout,
                    json_mode=json_mode,
                )
                # For now, return a basic streaming wrapper
                # This would need proper streaming implementation based on GigaChat API
                from litellm.utils import CustomStreamWrapper
                return CustomStreamWrapper(
                    completion_stream=completion_stream,
                    model=model,
                    custom_llm_provider="gigachat",
                    logging_obj=logging_obj,
                )

            else:
                if client is None or not isinstance(client, HTTPHandler):
                    client = _get_httpx_client(
                        params={"timeout": timeout}
                    )
                else:
                    client = client

                try:
                    response = client.post(
                        api_base,
                        headers=headers,
                        data=json.dumps(data),
                        timeout=timeout,
                    )
                except Exception as e:
                    status_code = getattr(e, "status_code", 500)
                    error_headers = getattr(e, "headers", None)
                    error_text = getattr(e, "text", str(e))
                    error_response = getattr(e, "response", None)
                    if error_headers is None and error_response:
                        error_headers = getattr(error_response, "headers", None)
                    if error_response and hasattr(error_response, "text"):
                        error_text = getattr(error_response, "text", error_text)
                    raise GigaChatError(
                        message=error_text,
                        status_code=status_code,
                        headers=error_headers,
                    )

        return config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            json_mode=json_mode,
        )

    def embedding(self):
        # logic for parsing in - calling - parsing out model embedding calls
        pass
