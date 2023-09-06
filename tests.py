from simca.tests import test_cassisystem_initialization, test_functions_acquisition
import unittest

if __name__ == '__main__':
    unittest.main(test_cassisystem_initialization, exit=False)

    unittest.main(test_functions_acquisition)