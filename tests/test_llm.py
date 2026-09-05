import json
from types import SimpleNamespace

from agents.llm import LLM, _assemble_streaming_response
from agents.llm import _FinalAnswerEchoer


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


def test_streaming_assembly_joins_content_deltas():
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": None}, "finish_reason": "stop"}]},
    ]
    response = _assemble_streaming_response(iter(chunks))
    message = response["choices"][0]["message"]
    assert message["content"] == "Hello"
    assert message["role"] == "assistant"
    assert "tool_calls" not in message
    assert response["choices"][0]["finish_reason"] == "stop"


def test_streaming_assembly_rebuilds_fragmented_tool_calls():
    arguments = '{"value": 7}'
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "test__agent_echo",
                                    "arguments": "",
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": None,
                                "function": {
                                    "name": None,
                                    "arguments": arguments[:5],
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": None,
                                "function": {
                                    "name": None,
                                    "arguments": arguments[5:],
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"delta": {"content": None}, "finish_reason": "tool_calls"}]},
    ]
    response = _assemble_streaming_response(iter(chunks))
    message = response["choices"][0]["message"]
    assert message["content"] is None
    call = message["tool_calls"][0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "test__agent_echo"
    assert call["function"]["arguments"] == arguments


def test_streaming_assembly_closes_stream_on_mid_stream_error():
    class BrokenStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
            raise RuntimeError("connection reset")

        def close(self):
            self.closed = True

    stream = BrokenStream()
    try:
        _assemble_streaming_response(stream)
    except RuntimeError:
        pass
    else:
        raise AssertionError("stream error should propagate")
    assert stream.closed


def test_complete_streaming_forwards_options_and_streams():
    recorded = {}

    class Completions:
        def create(self, **kwargs):
            recorded.update(kwargs)

            def gen():
                yield {
                    "choices": [
                        {"delta": {"content": "ok"}, "finish_reason": None}
                    ]
                }

            return gen()

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm = bare_llm(client)

    response = llm.complete_streaming(
        [{"role": "user", "content": "hi"}],
        temperature=0.3,
        timeout=9,
        prompt_cache_key="cache-1",
    )

    assert recorded["stream"] is True
    assert recorded["timeout"] == 9
    assert recorded["prompt_cache_key"] == "cache-1"
    assert response["choices"][0]["message"]["content"] == "ok"


def _chunk(delta: dict) -> dict:
    return {"choices": [{"delta": delta}]}


def test_react_final_echo_stays_silent_until_marker_then_streams():
    pieces: list[str] = []
    echoer = _FinalAnswerEchoer("react_final", pieces.append, prefix="")

    echoer.feed("Thought: I need time\nAction: system.current_time\n")
    assert pieces == []  # tool-round text must not echo

    echoer.feed("Action Input: {}\n")
    assert pieces == []

    echoer.feed("Final Answer: It is ")
    echoer.feed("12:00.")
    echoer.flush()

    assert "".join(pieces) == " It is 12:00.\n"
    assert echoer.echoed


def test_react_final_echo_supports_chinese_marker_and_fullwidth_colon():
    pieces: list[str] = []
    echoer = _FinalAnswerEchoer("react_final", pieces.append, prefix="")

    echoer.feed("思考：需要时间\n行动：system.current_time\n")
    echoer.feed("最终答案：现在是")
    echoer.feed("12点")
    echoer.flush()

    assert "".join(pieces) == "现在是12点\n"


def test_content_echo_mode_forwards_every_delta_immediately():
    pieces: list[str] = []
    echoer = _FinalAnswerEchoer("content", pieces.append, prefix="")

    echoer.feed("hel")
    echoer.feed("lo")
    echoer.flush()

    assert pieces == ["hel", "lo", "\n"]


def test_silent_round_leaves_terminal_untouched():
    pieces: list[str] = []
    echoer = _FinalAnswerEchoer("react_final", pieces.append)

    echoer.feed("Thought: call the tool\nAction: web.search\n")
    echoer.feed('Action Input: {"query": "python"}')
    echoer.flush()

    assert pieces == []
    assert echoer.echoed is False


def test_streaming_response_echoes_content_during_assembly():
    captured: list[str] = []
    stream = iter(
        [
            _chunk({"content": "Thought: t\nAction: x\n"}),
            _chunk({"content": "Final Answer: done"}),
        ]
    )
    response = _assemble_streaming_response(
        stream, echo_mode="react_final", echo_write=captured.append
    )

    assert response["choices"][0]["message"]["content"] == (
        "Thought: t\nAction: x\nFinal Answer: done"
    )
    assert response["stream_echoed"] is True
    # default prefix is emitted exactly once before the tail
    assert "".join(captured) == "\nAI： done\n"


def test_streaming_response_reports_unechoed_final_answers():
    captured: list[str] = []
    stream = iter(
        [
            _chunk({"content": "plain answer without marker"}),
        ]
    )
    response = _assemble_streaming_response(
        stream, echo_mode="react_final", echo_write=captured.append
    )

    assert response["stream_echoed"] is False
    assert captured == []


def test_complete_streaming_passes_echo_mode_to_assembler():
    recorded = {}

    class Completions:
        def create(self, **kwargs):
            recorded.update(kwargs)

            class Chunk:
                def __init__(self, delta):
                    self.choices = [SimpleNamespace(delta=delta)]

            return iter(
                [
                    Chunk(SimpleNamespace(content="Final Answer: hi")),
                ]
            )

    llm = LLM.__new__(LLM)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm.model = "m"
    pieces: list[str] = []

    response = llm.complete_streaming(
        [{"role": "user", "content": "hi"}],
        echo_mode="react_final",
        echo_write=pieces.append,
    )

    assert response["choices"][0]["message"]["content"] == "Final Answer: hi"
    assert "".join(pieces) == "\nAI： hi\n"
