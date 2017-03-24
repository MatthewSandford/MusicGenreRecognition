import os
import json

def load_json(file_name):

    if os.stat(file_name).st_size != 0:
        
        with open(file_name) as f:
              
            return json.load(f)
        
    else:

        return {}

def save_to_json(data,file_name):

    with open(file_name, "w") as fp:

        json.dump(data, fp)