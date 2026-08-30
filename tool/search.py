
from base import BaseTool

class SearchTool(BaseTool):

    name = "search"
    description = "useful when you want to search the internet"
    args_schema = ""
    func = run()

    def __init__(self, name: str, description: str, func: callable, args_schema: dict[str, any]):
        super().__init__(name, description, func, args_schema)


    def parse_args(self,query: str) -> dict[str,any]:



    def run(self,query: str) -> str:

        