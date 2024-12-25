import os
import requests
import json
from typing import List


class EmbeddingClient:
    """
    A client for interacting with the embedding service API.

    This class handles sending text data to the embedding API and retrieving the generated embeddings.
    It uses the URL from the environment variable `EMBEDDING_URL` for the API endpoint.

    Attributes:
        url (str): The base URL of the embedding service API.
        headers (dict): The headers required by the API for the request.
    """

    def __init__(self, embedding_url: str = None):
        """
        Initializes the EmbeddingClient with the URL from the environment variable or provided as an argument.

        Args:
            embedding_url (str): The embedding API URL. Defaults to None, in which case the URL is retrieved from the environment variable 'EMBEDDING_URL'.
        """
        self.url = embedding_url or os.getenv("EMBEDDING_URL")
        if not self.url:
            raise ValueError(
                "Embedding URL must be provided either as an argument or via environment variable 'EMBEDDING_URL'."
            )

        # Set headers for the HTTP request
        self.headers = {
            "Content-Type": "application/json",
            "Cookie": "acw_tc=0a5cc91217346004928187547ede31ab8900fa93156591e93b26d29b8a9da9",  # Cookie should be secure and configurable
        }

    def get_embeddings(
        self,
        text_list: List[str],
        model: str = "m3e",
        version: str = "m3e",
        unique_id: str = "test",
    ) -> dict:
        """
        Sends the text list to the embedding API and retrieves the embeddings.

        Args:
            text_list (List[str]): A list of text strings to be embedded.
            model (str): The model to use for embedding generation (default: "m3e").
            version (str): The version of the model to use (default: "m3e").
            unique_id (str): A unique identifier for the request (default: "test").

        Returns:
            dict: The response from the API containing the generated embeddings or an error message.

        Raises:
            ValueError: If the API response indicates an error.
        """
        payload = {
            "model": model,
            "textList": text_list,
            "version": version,
            "uniqueId": unique_id,
        }

        # 发送请求并获取响应
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()  # Check if the request was successful
            response_data = response.json()

            # 检查API返回的状态是否正确
            if "error" in response_data:
                raise ValueError(f"API Error: {response_data['error']}")

            return response_data
        except requests.RequestException as e:
            # 捕捉请求中的任何异常，打印日志信息
            raise ValueError(f"Request failed: {e}")

    def print_embeddings(self, text_list: List[str]) -> None:
        """
        Retrieves and prints the embeddings for a given list of texts.

        Args:
            text_list (List[str]): The list of text to get embeddings for.
        """
        try:
            embeddings = self.get_embeddings(text_list)
            result_list = embeddings.get("data", {}).get("resultList", [])
            print(f"Embeddings for the provided text list:")
            for text, embedding in zip(text_list, result_list):
                print(f"Text: {text}")
                print(f"Embedding: {embedding}")
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Set API URL
    TEST_URL = "https://ai-platform-cloud-proxy.polymas.com/ai/common/kb-get-embedding"

    # Input text to embed
    text_to_embed = [
        "人机界面设计问题\n1人机界面设计问题\n2人机界面设计过程\n3人机界面设计指南\n(1)系统响应时间\n(2)用户帮助设施\n(3)出错信息处理\n(4)命令交互\n本节课主要探讨人机界面设计的相关问题、设计过程及设计指南。首先,人机界面设计问题中,系统响应时间被定义为用户完成特定控制动作(如按回车键或点击鼠标)至软件给出预期响应之间的时间间隔。现代软件通常提供在线帮助功能,使用户能够在不离开界面的情况下解决问题。此外,出错信息的处理是用户交互中重要的一环,系统通过出错信息和警告信息向用户传达潜在问题。最后,命令交互仍然受到许多高级用户的青睐,他们可以选择通过菜单或键盘命令序列来调用软件功能,这种灵活性增强了用户的操作体验"
    ]

    # Instantiate client and print embeddings
    embedding_client = EmbeddingClient(embedding_url=TEST_URL)
    try:
        embedding_client.print_embeddings(text_to_embed)
    except Exception as e:
        print(f"An error occurred: {e}")
