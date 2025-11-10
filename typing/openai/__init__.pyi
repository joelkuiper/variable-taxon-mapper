from .lib._old_api import *
import httpx as _httpx
from . import types as types
from ._base_client import DEFAULT_MAX_RETRIES as DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT as DEFAULT_TIMEOUT, DefaultAioHttpClient as DefaultAioHttpClient, DefaultAsyncHttpxClient as DefaultAsyncHttpxClient, DefaultHttpxClient as DefaultHttpxClient
from ._client import AsyncClient as AsyncClient, AsyncOpenAI as AsyncOpenAI, AsyncStream as AsyncStream, Client as Client, OpenAI as OpenAI, RequestOptions as RequestOptions, Stream as Stream, Timeout as Timeout, Transport as Transport
from ._constants import DEFAULT_CONNECTION_LIMITS as DEFAULT_CONNECTION_LIMITS
from ._exceptions import APIConnectionError as APIConnectionError, APIError as APIError, APIResponseValidationError as APIResponseValidationError, APIStatusError as APIStatusError, APITimeoutError as APITimeoutError, AuthenticationError as AuthenticationError, BadRequestError as BadRequestError, ConflictError as ConflictError, ContentFilterFinishReasonError as ContentFilterFinishReasonError, InternalServerError as InternalServerError, InvalidWebhookSignatureError as InvalidWebhookSignatureError, LengthFinishReasonError as LengthFinishReasonError, NotFoundError as NotFoundError, OpenAIError as OpenAIError, PermissionDeniedError as PermissionDeniedError, RateLimitError as RateLimitError, UnprocessableEntityError as UnprocessableEntityError
from ._models import BaseModel as BaseModel
from ._types import NOT_GIVEN as NOT_GIVEN, NoneType as NoneType, NotGiven as NotGiven, Omit as Omit, ProxiesTypes as ProxiesTypes, not_given as not_given, omit as omit
from ._utils import file_from_path as file_from_path
from ._version import __title__ as __title__, __version__ as __version__
from .lib.azure import AzureOpenAI as AzureOpenAI
from typing_extensions import override

__all__ = ['types', '__version__', '__title__', 'NoneType', 'Transport', 'ProxiesTypes', 'NotGiven', 'NOT_GIVEN', 'not_given', 'Omit', 'omit', 'OpenAIError', 'APIError', 'APIStatusError', 'APITimeoutError', 'APIConnectionError', 'APIResponseValidationError', 'BadRequestError', 'AuthenticationError', 'PermissionDeniedError', 'NotFoundError', 'ConflictError', 'UnprocessableEntityError', 'RateLimitError', 'InternalServerError', 'LengthFinishReasonError', 'ContentFilterFinishReasonError', 'InvalidWebhookSignatureError', 'Timeout', 'RequestOptions', 'Client', 'AsyncClient', 'Stream', 'AsyncStream', 'OpenAI', 'AsyncOpenAI', 'file_from_path', 'BaseModel', 'DEFAULT_TIMEOUT', 'DEFAULT_MAX_RETRIES', 'DEFAULT_CONNECTION_LIMITS', 'DefaultHttpxClient', 'DefaultAsyncHttpxClient', 'DefaultAioHttpClient']

class _ModuleClient(OpenAI):
    @property
    @override
    def api_key(self) -> str | None: ...
    @api_key.setter
    def api_key(self, value: str | None) -> None: ...
    @property
    @override
    def organization(self) -> str | None: ...
    @organization.setter
    def organization(self, value: str | None) -> None: ...
    @property
    @override
    def project(self) -> str | None: ...
    @project.setter
    def project(self, value: str | None) -> None: ...
    @property
    @override
    def webhook_secret(self) -> str | None: ...
    @webhook_secret.setter
    def webhook_secret(self, value: str | None) -> None: ...
    @property
    @override
    def base_url(self) -> _httpx.URL: ...
    @base_url.setter
    def base_url(self, url: _httpx.URL | str) -> None: ...
    @property
    @override
    def timeout(self) -> float | Timeout | None: ...
    @timeout.setter
    def timeout(self, value: float | Timeout | None) -> None: ...
    @property
    @override
    def max_retries(self) -> int: ...
    @max_retries.setter
    def max_retries(self, value: int) -> None: ...

class _AzureModuleClient(_ModuleClient, AzureOpenAI): ...

class _AmbiguousModuleClientUsageError(OpenAIError):
    def __init__(self) -> None: ...
