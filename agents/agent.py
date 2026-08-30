from abc import ABC, abstractmethod
from agents.llm import LLM
from openai import OpenAI

MAX_RETRYS = 10

class agent(ABC):
    def __init__(self, name):

        self.llm = LLM()
        self.tools :dict[dict[str,any]] = []
        self.providers :dict[dict[str,any]] = []
        self.global_model = "None"

        self.name = name
        self.role : list[str] = ["user", "assistant", "system", "tool"]
        self.prompt : dict[str, str] = {}
        self.history : list[str] = []
        self.max_retries = MAX_RETRYS

    def clear_history(self):
        self.history = []

    @abstractmethod
    def run(self,query: str) -> str:
        pass

#   设置

    def set_system_prompt(self, prompt: str):
        self.prompt["system"] = prompt

    def set_user_prompt(self, prompt: str):
        self.prompt["user"] = prompt

    def set_assistant_prompt(self, prompt: str):
        self.prompt["assistant"] = prompt

    def set_tool_prompt(self, prompt: str):
        self.prompt["tool"] = prompt

    def set_global_model(self, model: str):
        self.global_model = model


#   常规函数

    def detect_models(self,base_url:str,api_key:str) -> list[str] :
        detect_client = OpenAI(api_key=api_key, base_url=base_url,max_retries=self.max_retries)
        return detect_client.models.list()


#   增
    #   provider
    def add_provider(self,provider_name:str,api_key:str,base_url:str,default_model:str = None) :

            if provider_name in self.providers:
                raise ValueError(f"Provider '{provider_name}' already exists.")
    
            supported_models = self.detect_models(base_url,api_key)

            if default_model is None:
                if supported_models:
                    default_model = supported_models[0]
                else:
                    default_model = "None"

            provider : dict[str, any] = {
                "api_key": api_key,
                "base_url": base_url,
                "all_unvisual_models":supported_models,
                "models": [],
                "default_model": default_model
            }
            self.providers.append({provider_name: provider})

    def add_model_to_provider(self, provider_name: str, model_name: str):
    
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found.")
    
            if model_name not in self.providers[provider_name]["all_unvisual_models"]:
                self.list_single_provider_models(provider_name)
                raise ValueError(f"提供商 '{provider_name}' 不支持模型 '{model_name}'。")
            
            self.providers[provider_name]["models"].append(model_name)


    

#   删

    #provider
    def delete_model_from_provider(self, provider_name: str, model_name: str):
    
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found.")
    
            if model_name not in self.providers[provider_name]["models"]:
                raise ValueError(f"Model '{model_name}' not found for provider '{provider_name}'.")
    
            self.providers[provider_name]["models"].remove(model_name)


    def delete_provider(self,provider_name:str) :
         self.providers.remove(provider_name)

#   改/更新

    #provider
    def change_all_provider_avalible_models(self):
        for provider , provider_info in self.providers.items():
            url = provider_info["base_url"]
            api_key = provider_info["api_key"]
            supported_models = self.detect_models(url,api_key)
            provider_info["all_unvisual_models"] = self.detect_models(url,api_key)

    
    
    def change_default_model(self, provider_name: str, default_model: str):
            if provider_name not in self.providers:
                raise ValueError(f"Provider '{provider_name}' not found.")

            if default_model not in self.providers[provider_name]["all_unvisual_models"]:
                raise ValueError(f"提供商 '{provider_name}' 不支持模型 '{default_model}'。 请手动探测可用模型之后重试")
            
            self.providers[provider_name]["default_model"] = default_model

    


#   查

    #provider
    def get_single_provider_models(self, provider_name: str) -> list[str]:
            models = self.providers[provider_name]["models"]
    
            if not models:
                models.append("None")
            return models
    
    def get_all_providers(self) -> list[str] :
            return list(self.providers.keys())
    
    def get_single_provider_avalible_models(self,provider_name:str) -> list[str] :
        return self.providers[provider_name]["all_unvisual_models"]


    def get_all_avalible_models(self) -> list[str] :
        all_models :list[str] = []

        for provider, provider_info in self.providers.items():
            for model in provider_info["all_unvisual_models"]:
                all_models.append( provider +":" + model )

        if not all_models:
            all_models.append("None")
        
        return all_models

#O   O

    #provider
    def list_all_avalible_models(self) -> list[str] :
            all_models :list[str] = self.get_all_avalible_models()
    
            if not all_models:
                print("No available models found.")
                return None
    
            for p in all_models:
                print(p)

    def list_single_provider_models(self, provider_name: str) -> list[str]:
            models = self.get_single_provider_models(provider_name)
    
            if not models:
                print(f"No models found for provider '{provider_name}'.")
                return None
            for model in models:
                print(model)

    def list_all_providers_info(self) :
        for provider, provider_info in self.providers.items():
            print(f"Provider: {provider} url: {provider_info['base_url']} default_model: {provider_info['default_model']}")

    def list_single_provider_info(self, provider_name: str) :
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found.")
        provider_info = self.providers[provider_name]
        print(f"Provider: {provider_name} url: {provider_info['base_url']} default_model: {provider_info['default_model']}")
#    I




        

    
    

    

    

