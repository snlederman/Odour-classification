"""
pickle best model
"""

# packages
import os
import pickle

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def pickle_model(model, model_file):
    pickle.dump(model, open(model_file, 'wb'))

if __name__ == "__main__":
    model_file = os.path.join(PROJECT_DIR, "src", "best_model.pkl")    
    # pickle_model(model, model_file)
