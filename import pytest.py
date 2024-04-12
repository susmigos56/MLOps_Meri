import unittest
import pandas as pd
from custom_preprocessor import CustomPreprocessor

class TestCustomPreprocessor(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            'booking_id': [0, 1],
            'hotel': ['Resort Hotel', 'City Hotel'],
            'is_canceled': [0, 1],
            'lead_time': [342, 737],
            'arrival_date_year': [2015, 2016],
            'children': [0, 1],  # Adding a column
            # Add other columns here as needed
        })

    def test_preprocessing(self):
        # Initialize the preprocessor
        preprocessor = CustomPreprocessor()

        # Apply the preprocessing steps
        preprocessed_data = preprocessor.transform(self.sample_data)

        # Expected columns after preprocessing
        expected_columns = [
            'hotel', 'is_canceled', 'lead_time', 'stays_in_weekend_nights',
            'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
            'market_segment', 'distribution_channel', 'is_repeated_guest',
            'previous_cancellations', 'previous_bookings_not_canceled',
            'reserved_room_type', 'booking_changes', 'deposit_type', 'agent',
            'customer_type', 'adr', 'required_car_parking_spaces',
            'total_of_special_requests', 'year', 'month', 'day'  # Added by preprocessing
        ]

        # Assert that all expected columns are present in the preprocessed data
        self.assertCountEqual(preprocessed_data.columns.tolist(), expected_columns)

if __name__ == '__main__':
    unittest.main()
