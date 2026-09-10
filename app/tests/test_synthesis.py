"""Offline regressions; native ML libraries are replaced at their boundary."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_codes import build_prompt, parse_output, split_codes


def frame(values=range(7)):
    return [128266 + i * 4096 + value for i, value in enumerate(values)]


class ProtocolTests(unittest.TestCase):
    def test_prompt_includes_full_assistant_audio_prefix(self):
        model = MagicMock()
        model.tokenize.return_value = [128000, 42]
        self.assertEqual(build_prompt(model, "Bonjour", "Pierre"),
                         [128259, 128000, 42, 128009, 128260, 128261, 128257])
        model.tokenize.assert_called_once_with(b"Pierre: Bonjour", add_bos=True, special=False)

    def test_frames_and_layer_order(self):
        codes = parse_output(frame() + frame(range(10, 17)))
        self.assertEqual(split_codes(codes), [[0, 10], [1, 4, 11, 14], [2, 3, 5, 6, 12, 13, 15, 16]])

    def test_stop_and_incomplete_tail(self):
        self.assertEqual(parse_output([128257] + frame() + [128258, 123]), list(range(7)))
        self.assertEqual(parse_output(frame() + frame()[:3]), list(range(7)))

    def test_empty_and_malformed_audio_is_an_error(self):
        for tokens in ([], [128258], frame()[:6], [1] + frame(), frame([4096] * 7)):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                parse_output(tokens)

    def test_codebook_boundaries(self):
        self.assertEqual(parse_output(frame([4095] * 7)), [4095] * 7)
        for codes in ([], [0], [-1] * 7, [4096] * 7):
            with self.assertRaises(ValueError):
                split_codes(codes)


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        deps = {name: MagicMock() for name in
                ("gradio", "torch", "numpy", "soundfile", "snac", "huggingface_hub", "llama_cpp")}
        spec = importlib.util.spec_from_file_location("orpheus_test_app", Path(__file__).resolve().parents[1] / "app.py")
        self.app = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, deps):
            spec.loader.exec_module(self.app)
        self.model = MagicMock()
        self.model.tokenize.return_value = [128000, 42]
        self.model.n_ctx.return_value = 4096
        self.model.token_eos.return_value = 128009
        self.app.LOADED_MODELS.update(orpheus_model=self.model, current_model_type="english")
        self.decode = self.app.redistribute_codes
        self.app.redistribute_codes = MagicMock(return_value=[0.1, -0.1])
        self.output_patch = patch("builtins.print")
        self.traceback_patch = patch("traceback.print_exc")
        self.output_patch.start()
        self.traceback_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.addCleanup(self.traceback_patch.stop)

    def synthesize(self, **overrides):
        args = dict(text="Hello", voice="tara", model_type="english", temperature=0.6,
                    top_p=0.8, repetition_penalty=1.3, max_new_tokens=100)
        args.update(overrides)
        return self.app.synthesize(**args)

    def test_success_stops_at_audio_end_and_closes_generator(self):
        closed = []
        def stream():
            try:
                yield from frame() + [128258, 0]
            finally:
                closed.append(True)
        self.model.generate.return_value = stream()
        path, status = self.synthesize()
        self.assertEqual(status, "Done!")
        self.assertEqual(Path(path).parent, self.app.APP_DIR / "outputs")
        self.app.redistribute_codes.assert_called_once_with(list(range(7)), None)
        self.assertEqual(closed, [True])

    def test_invalid_audio_never_writes_a_successful_wav(self):
        self.model.generate.return_value = iter_generator([128258])
        path, status = self.synthesize()
        self.assertIsNone(path)
        self.assertIn("complete audio frame", status)
        self.app.sf.write.assert_not_called()

    def test_native_context_budget_bounds_generation(self):
        self.model.n_ctx.return_value = 14  # seven prompt IDs, seven remaining
        seen = []
        def stream():
            for token in frame() * 20:
                seen.append(token)
                yield token
        self.model.generate.return_value = stream()
        self.assertEqual(self.synthesize()[1], "Done!")
        self.assertEqual(len(seen), 7)

    def test_long_prompt_is_rejected_before_generation(self):
        self.model.n_ctx.return_value = 13
        self.assertIn("too long", self.synthesize()[1])
        self.model.generate.assert_not_called()

    def test_bad_api_inputs_do_not_load_or_generate(self):
        for args in ({"model_type": "../bad"}, {"max_new_tokens": 0},
                     {"max_new_tokens": 100.5}, {"temperature": float("nan")}, {"top_p": 2}):
            with self.subTest(args=args):
                self.assertIsNone(self.synthesize(**args)[0])
        self.model.generate.assert_not_called()

    def test_unknown_model_does_not_unload_existing_model(self):
        with self.assertRaises(ValueError):
            self.app.load_models("unknown")
        self.model.close.assert_not_called()

    def test_decoder_disables_autograd_and_propagates_failures(self):
        snac = MagicMock()
        snac.parameters.return_value = iter([MagicMock(device="cpu")])
        self.decode(list(range(7)), snac)
        self.app.torch.inference_mode.return_value.__enter__.assert_called_once()
        snac.parameters.return_value = iter([MagicMock(device="cpu")])
        snac.decode.side_effect = RuntimeError("decoder failed")
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            self.decode(list(range(7)), snac)


def iter_generator(tokens):
    yield from tokens


if __name__ == "__main__":
    unittest.main()
