import json
import os

def save_result_json(result_json: dict, model_name: str, loc_name: str, stat_type: str):
    json_file = os.path.join("Result", "STATS", f"{model_name}", stat_type, f"{model_name}_{loc_name}.json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as json_file:
        json.dump(result_json, json_file, indent=4)