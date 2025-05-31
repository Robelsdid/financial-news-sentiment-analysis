import unittest

class TestPlaceholder(unittest.TestCase):
    def test_sample(self):
        self.assertEqual(1, 1)

if __name__ == "__main__":
    unittest.main()
# //This gives GitHub Actions something to run while I am still building the project.
#//It’ll confirm my .github/workflows/unittests.yml setup is working perfectly.