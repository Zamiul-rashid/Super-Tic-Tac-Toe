import json
import sys
import tempfile
import unittest
from pathlib import Path

from sttt.engine_registry import create_configured_engine, load_engine_registry


class EngineRegistryTests(unittest.TestCase):
    def test_loads_relative_command_and_creates_strict_external_bot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'registry.json'
            config.write_text(json.dumps({'engines': {'mock': {
                'command': [sys.executable, str(Path(__file__).parent / 'mock_codingame_bot.py')], 'cwd': '.',
                'protocol': 'codingame', 'timeout': 2, 'fallback': 'raise'}}}))
            registry = load_engine_registry(config)
            self.assertEqual(registry['mock']['cwd'], str(root))
            bot = create_configured_engine('mock', registry)
            self.assertEqual(bot.name, 'mock')
            self.assertEqual(bot.protocol, 'codingame')
            self.assertEqual(bot.fallback, 'raise')
            bot.close()

    def test_rejects_missing_command_and_unknown_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'registry.json'
            for spec in ({'command': ''}, {'command': ['x'], 'protocol': 'bogus'}):
                path.write_text(json.dumps({'engines': {'bad': spec}}))
                with self.assertRaises(ValueError):
                    load_engine_registry(path)

    def test_unknown_name_stays_for_builtin_factory(self):
        self.assertIsNone(create_configured_engine('alphabeta', {}))


if __name__ == '__main__':
    unittest.main()
