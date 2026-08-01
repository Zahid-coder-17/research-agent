import sys
import unittest

def main():
    """Runs all unit tests in the repository."""
    print("\n=======================================================")
    print("  RUNNING RESEARCH AGENT UNIT TEST SUITE")
    print("=======================================================\n")
    
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n=======================================================")
    if result.wasSuccessful():
        print("  ALL UNIT TESTS PASSED (100% SUCCESS)")
        print("=======================================================\n")
        sys.exit(0)
    else:
        print("  SOME UNIT TESTS FAILED")
        print("=======================================================\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
