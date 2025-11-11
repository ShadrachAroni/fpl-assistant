# scripts/run_recommend.py
import argparse
import requests
import json

parser = argparse.ArgumentParser()
parser.add_argument("--manager-id", type=int, required=True)
parser.add_argument("--host", default="http://127.0.0.1:8000")
args = parser.parse_args()

resp = requests.get(f"{args.host}/recommend/{args.manager_id}")
print(json.dumps(resp.json(), indent=2))
