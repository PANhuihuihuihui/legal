from .crime_prediction import get_crime_datasets

def build_dataset(args):
    return get_crime_datasets(args)