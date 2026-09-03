from types import SimpleNamespace

from agents.llm import LLM


def bare_llm(client=None):
    llm = LLM.__new__(LLM)
    llm.client = client
    llm.model = "default-model"
    llm.max_retries = 3
    return llm


def test_complete_forwards_validated_arguments():
    recorded = {}

    class Completions:
        def create(self, **kwargs):
            recorded.update(kwargs)
            return {"choices": [{"message": {"content": "done"}}]}

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm = bare_llm(client)

    response = llm.complete(
        [{"role": "user", "content": "hello"}],
        model="override-model",
        temperature=0.2,
        timeout=5,
        stream=False,
        tools=[],
    )

    assert response["choices"][0]["message"]["content"] == "done"
    assert recorded["model"] == "override-model"
    assert recorded["tools"] == []


def test_think_collects_and_closes_a_stream(capsys):
    class Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield {"choices": [{"delta": {"content": "hel"}}]}
            yield {"choices": [{"delta": {"content": "lo"}}]}

        def close(self):
            self.closed = True

    stream = Stream()
    llm = bare_llm()
    llm.complete = lambda *args, **kwargs: stream

    result = llm.think(
        [{"role": "user", "content": "hello"}], stream_response_bool=True
    )

    assert result == "hello"
    assert stream.closed
    assert capsys.readouterr().out == "hello"


def test_think_reads_a_non_streaming_mapping_response():
    llm = bare_llm()
    llm.complete = lambda *args, **kwargs: {
        "choices": [{"message": {"content": "done"}}]
    }

    assert (
        llm.think(
            [{"role": "user", "content": "hello"}],
            stream_response_bool=False,
        )
        == "done"
    )
