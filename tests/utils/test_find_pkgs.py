import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_find_pkgs():
    path = Path(__file__).parents[2] / 'deep_ep' / 'utils' / 'find_pkgs.py'
    spec = importlib.util.spec_from_file_location('find_pkgs_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NCCLVersionTests(unittest.TestCase):
    def setUp(self):
        self.find_pkgs = load_find_pkgs()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / 'include').mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def write_header(self, contents):
        (self.root / 'include' / 'nccl.h').write_text(contents)

    def test_reads_version_from_nccl_header(self):
        self.write_header('''
#define NCCL_MAJOR 2
#define NCCL_MINOR 30
#define NCCL_PATCH 4
''')
        self.assertEqual(self.find_pkgs.get_nccl_version(str(self.root)), '2.30.4')
        self.assertEqual(self.find_pkgs.check_nccl_version(str(self.root), '2.30.4'), '2.30.4')

    def test_rejects_runtime_version_different_from_build(self):
        self.write_header('''
#define NCCL_MAJOR 2
#define NCCL_MINOR 28
#define NCCL_PATCH 9
''')
        with self.assertRaisesRegex(RuntimeError, 'built against 2.30.4.*provides 2.28.9'):
            self.find_pkgs.check_nccl_version(str(self.root), '2.30.4')

    def test_rejects_incomplete_version_header(self):
        self.write_header('#define NCCL_MAJOR 2\n#define NCCL_MINOR 30\n')
        with self.assertRaisesRegex(RuntimeError, 'NCCL_PATCH is missing'):
            self.find_pkgs.get_nccl_version(str(self.root))


if __name__ == '__main__':
    unittest.main()
