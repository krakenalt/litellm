from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    request: str
    """Запрос пользователя"""
    params: Dict[str, Any]


class FunctionParametersProperty(BaseModel):
    """Функция, которая может быть вызвана моделью"""

    type_: str = Field(default="object", alias="type")
    """Тип аргумента функции"""
    description: str = ""
    """Описание аргумента"""
    items: Optional[Dict[str, Any]] = None
    """Возможные значения аргумента"""
    enum: Optional[List[str]] = None
    """Возможные значения enum"""
    properties: Optional[Dict[Any, "FunctionParametersProperty"]] = None


class FunctionParameters(BaseModel):
    """Функция, которая может быть вызвана моделью"""

    type_: str = Field(default="object", alias="type")
    """Тип параметров функции"""
    properties: Optional[Dict[Any, FunctionParametersProperty]] = None
    """Описание функции"""
    required: Optional[List[str]] = None
    """Список обязательных параметров"""


class Function(BaseModel):
    """Функция, которая может быть вызвана моделью"""

    name: str
    """Название функции"""
    description: Optional[str] = None
    """Описание функции"""
    parameters: Optional[FunctionParameters] = None
    """Список параметров функции"""
    few_shot_examples: Optional[List[FewShotExample]] = None
    return_parameters: Optional[Dict[Any, Any]] = None
    """Список возвращаемых параметров функции"""


class FunctionCall(BaseModel):
    """Вызов функции"""

    name: str
    """Название функции"""
    arguments: Optional[Dict[Any, Any]] = None
    """Описание функции"""
