from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None ,max_retries: int = 10):

        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL")
        self.max_retries = max_retries


        errors : list = []
        if self.api_key is None:
            errors.append("LLM_API_KEY is not set in the environment variables.")
        if self.base_url is None:
            errors.append("LLM_BASE_URL is not set in the environment variables.")
        if self.model is None:
            errors.append("LLM_MODEL is not set in the environment variables.")

        if errors:
            error_message = "\n".join(errors)
            raise ValueError(f"Configuration Error: {error_message}")
        
        try :
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,max_retries=self.max_retries)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.client = None
            raise ValueError("Failed to initialize OpenAI client. Please check your API key and base URL.")

    def get_query(self) -> str :
        query = input("请输入你的问题：")
        return query
    def think(self, messages: list[dict[str, str]], temperature: float = 0.7, timeout: int = 60,stream_response_bool: bool = True) -> str:

        try:

            print(f"-----正在向模型{self.model}发送请求-----")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("-----正在思考-----")
            if stream_response_bool:
                return "".join(self.stream_response(response))
            else :
                all_text = self.all_response(response)
                return all_text
        
        except Exception as e:
            print(f"Error during LLM think operation: {e}")
            raise ValueError("Failed to get response from LLM. Please check your request and try again.")


    def stream_response(self, response):
        try :
            for chunk in response:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    yield content
        except Exception as e:
            print(f"Error during streaming response: {e}")
            raise ValueError("Failed to stream response from LLM. Please check your request and try again.")

    def all_response(self,response):

        all_text = ""
        try :
            for chunk in response:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content
                if content:
                    all_text += content
            return all_text
        except Exception as e:
            print(f"Error during collecting all response: {e}")
            raise ValueError("Failed to collect all response from LLM. Please check your request and try again.")


