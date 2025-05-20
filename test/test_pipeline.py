# tests/test_pipeline.py
# tests/test_pipeline.py
import unittest
from unittest.mock import patch
import sys
import os
import numpy as np  # Import numpy for array assertions

# Add the parent directory to the Python path for importing 'main'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import main  # Now it should find main.py in the parent directory
# print(f"Imported main from: {main.__file__}") # Uncomment this for debugging

class TestPipeline(unittest.TestCase):

    @patch('main.load_data')
    @patch('main.preprocess_data')
    @patch('main.train_model')
    @patch('main.evaluate_model')
    def test_full_pipeline_execution(self, mock_evaluate, mock_train, mock_preprocess, mock_load):
        """Tests if the main pipeline functions are called."""
        mock_load.return_value = (np.array([[1, 2], [3, 4]]), np.array([0, 1])) # Changed to numerical NumPy arrays
        mock_preprocess.return_value = (np.array([[2, 4], [6, 8]]), np.array([0, 1])) # Changed to numerical NumPy arrays
        mock_train.return_value = 'trained_model'
        mock_evaluate.return_value = {'accuracy': 0.95}

        result = main.main()

        mock_load.assert_called_once()
        # Use np.array_equal to compare NumPy arrays element-wise for mock_preprocess
        mock_preprocess.assert_called_once()
        args_preprocess, _ = mock_preprocess.call_args
        self.assertTrue(np.array_equal(args_preprocess[0], np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.array_equal(args_preprocess[1], np.array([0, 1])))

        # Use np.array_equal to compare NumPy arrays element-wise for mock_train
        mock_train.assert_called_once()
        args_train, _ = mock_train.call_args
        self.assertTrue(np.array_equal(args_train[0], np.array([[2, 4], [6, 8]])))
        self.assertTrue(np.array_equal(args_train[1], np.array([0, 1])))

        # Use np.array_equal to compare NumPy arrays element-wise for mock_evaluate
        mock_evaluate.assert_called_once()
        args_evaluate, _ = mock_evaluate.call_args
        self.assertEqual(args_evaluate[0], 'trained_model')
        self.assertTrue(np.array_equal(args_evaluate[1], np.array([[2, 4], [6, 8]])))
        self.assertTrue(np.array_equal(args_evaluate[2], np.array([0, 1])))

        self.assertEqual(result, {'accuracy': 0.95})

    def test_load_data(self):
        """Example test for the data loading function (adapt to your actual loading)."""
        # Assuming your load_data function returns data and labels as NumPy arrays
        data, labels = main.load_data()
        self.assertIsNotNone(data)
        self.assertIsNotNone(labels)
        self.assertIsInstance(data, np.ndarray)  # Changed assertion to check for NumPy array
        self.assertIsInstance(labels, np.ndarray) # Changed assertion to check for NumPy array
        self.assertEqual(len(data), len(labels))

    def test_preprocess_data(self):
        """Example test for the data preprocessing function (adapt to your preprocessing)."""
        # Provide NumPy arrays with numerical data
        sample_data = np.array([1.0, 2.5, 3.0])
        sample_labels = np.array([0, 1, 0])
        processed_data, processed_labels = main.preprocess_data(sample_data, sample_labels)
        self.assertIsNotNone(processed_data)
        self.assertIsNotNone(processed_labels)
        self.assertEqual(len(processed_data), len(sample_data))
        self.assertEqual(len(processed_labels), len(sample_labels))
        # Add assertions to check if the scaling was applied
        self.assertTrue(np.all(processed_data == np.array([2.0, 5.0, 6.0])))

    # Add more test methods for other functions in your pipeline
    # For example: test_train_model, test_evaluate_model, etc.

if __name__ == '__main__':
    unittest.main()