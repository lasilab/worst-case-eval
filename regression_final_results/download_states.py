'''
for each state, downloads/caches the Folktables data for that state
'''

from folktables2.folktables import ACSDataSource
from tqdm import tqdm

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"][:5]

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

if __name__ == "__main__":
        for i, state in tqdm(list(enumerate(states))):
                    acs_data = data_source.get_data(states=[state], download=True)
