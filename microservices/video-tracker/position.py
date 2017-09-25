"""
Monitor the redis database and print the position when it changes

Usage: python3 track.py
"""

import redis
import time
import json
from config import *

db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

position = None

while True:
	new_position = db.get('position')
	if new_position and new_position!=position:
		print(json.loads(new_position))
	# Prepare for next loop
	position = new_position
	time.sleep(0.01)
