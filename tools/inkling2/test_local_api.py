import json
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from local_api import CompletionDecoder, normalize_request


class LocalApiTests(unittest.TestCase):
    def setUp(self):
        self.call = '{"name":"lookup","args":{"key":"日本語"}}'
        self.unknown_call = '{"name":"unknown","args":{}}'
        self.tokens = [
            "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
            "<|content_invoke_tool_json|>", "<|end_message|>", "<|content_model_end_sampling|>",
            "lookup", "think", "answer", self.call, self.unknown_call,
        ]
        self.tokenizer = Tokenizer(WordLevel({token: i for i, token in enumerate(self.tokens)}))

    def decode(self, decoder, *tokens):
        return [delta for token in tokens if (delta := decoder.step(self.tokenizer.token_to_id(token)))]

    def test_tool_call_and_segment_boundaries(self):
        decoder = CompletionDecoder(self.tokenizer, {"lookup"}, True)
        deltas = self.decode(
            decoder, "<|message_model|>", "<|content_thinking|>", "think", "<|end_message|>",
            "<|message_model|>", "lookup", "<|content_invoke_tool_json|>", self.call, "<|end_message|>",
            "<|message_model|>", "<|content_text|>", "answer", "<|end_message|>",
        )
        self.assertEqual(deltas[0], {"reasoning_content": "think"})
        call = deltas[1]["tool_calls"][0]
        self.assertEqual(call["index"], 0)
        self.assertEqual(call["function"]["name"], "lookup")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"key": "日本語"})
        self.assertEqual(deltas[2:], [{"content": "answer"}])

    def test_unknown_and_truncated_calls_are_never_emitted(self):
        decoder = CompletionDecoder(self.tokenizer, {"lookup"}, True)
        self.assertEqual(self.decode(
            decoder, "<|content_invoke_tool_json|>", self.unknown_call, "<|end_message|>",
            "<|content_invoke_tool_json|>", self.call, "<|content_model_end_sampling|>",
        ), [])

    def test_tool_result_history_with_null_assistant_content(self):
        assistant = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":1}'},
        }]}
        result = {"role": "tool", "tool_call_id": "call_1", "content": "value"}
        messages, tools = normalize_request({"messages": [assistant, result], "tool_choice": "none"})
        self.assertEqual(tools, [])
        self.assertEqual(messages[0]["content"], "")
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], {"key": 1})
        self.assertEqual(messages[1], result)


if __name__ == "__main__":
    unittest.main()
