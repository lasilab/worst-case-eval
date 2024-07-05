'''
for each state, downloads/caches the Folktables data for that state
'''

from utils import states
from folktables import ACSDataSource
from tqdm import tqdm


data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

if __name__ == "__main__":
    for i, state in tqdm(list(enumerate(states))):
        acs_data = data_source.get_data(states=[state], download=True)