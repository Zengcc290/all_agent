
from abc import ABC, abstractmethod

import importlib
import json
import re

class BaseTool(ABC):

    def __init__(
        self, 
        name:str,
        description:str,
        func :callable,
        args_schema:dict[str,any],  #参数的规则约束

    ):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func

    @abstractmethod
    def run(self,query: str) -> str:
        pass

    @abstractmethod
    def parse_args(self,query: str) -> dict[str,any]:
        pass

    def get_tool_args_schema(self) -> dict[str,any]:
        return self.args_schema


    def extract_json_from_markdown(text: str):
        # 匹配 ```json ... ``` 代码块
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        return None

    def _run(self,query: str) -> str:

class register_tool:
    def __init__(self):
        self.tool_names = set()
        self.tools = dict[str,any]


    def register_tool(self,tool:BaseTool):
        if tool.name in self.tool_names:
            print(f"tool name {tool.name} already registered, overwrite tool {tool.name}")
        self.tool_names.add(tool.name) 
        self.tools[tool.name] = tool


