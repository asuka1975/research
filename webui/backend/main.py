from flask import Flask, abort, jsonify, request
from flask_cors import CORS
import logging

import os
import json

import redis

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.logger.setLevel(logging.INFO)
log_handler = logging.FileHandler("logs/server.log")
log_handler.setLevel(logging.INFO)
app.logger.addHandler(log_handler)

pool = redis.ConnectionPool(host="redis", port=6379, db=0)
rds = redis.StrictRedis(connection_pool=pool)

@app.route("/", methods=["GET"])
def get_settings():
    filter = request.args.get("filter")
    if not filter:
        filter = ""
    settings = [file.replace(".json", "") for file in os.listdir("/opt/app/unuse_setting") if filter in file]
    return jsonify({
        "items" : settings
    })

@app.route("/content", methods=["GET"])
def get_content():
    name = request.args.get("name")
    filepath = f"/opt/app/unuse_setting/{name}.json"
    if not os.path.exists(filepath):
        abort(404)
    with open(filepath, "r") as f:
        return f.read()


@app.route("/", methods=["POST"])
def create_setting():
    data = json.loads(request.get_data().decode("utf-8"))
    filepath = f"/opt/app/unuse_setting/{data['name']}.json"
    if os.path.exists(filepath):
        abort(403)
    with open(filepath, "w") as f:
        f.write(json.dumps(data["data"], indent=4))
    return ""

@app.route("/schedule", methods=["GET"])
def get_schedule():
    global rds
    return json.dumps([json.loads(v) for v in rds.lrange("schedule", 0, rds.llen("schedule"))])

@app.route("/schedule", methods=["POST"])
def set_schedule():
    global rds
    data = json.loads(request.get_data().decode("utf-8"))
    return ""

@app.route("/monitor", methods=["GET"])
def get_monitor():
    global rds
    rds.hgetall("monitor")
    return json.dumps([
        { "title" : setting.decode("utf-8"), "value" : v.decode("utf-8"), "unit" : "%" } for setting, v in rds.hgetall("monitor").items()
    ])
    

CORS(app, origin=["http://localhost:51224", "http://localhost:3000"])

if __name__ == "__main__":
    app.run(port=51223)