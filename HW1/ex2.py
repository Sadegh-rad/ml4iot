from time import time
from time import sleep
import uuid
import redis
import psutil
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--host', type=str, default='redis-19466.c293.eu-central-1-1.ec2.cloud.redislabs.com')
parser.add_argument('--port', type=int, default=19466)
parser.add_argument('--user', type=str, default='default')
parser.add_argument('--password', type=str, default='iIQjppRdJ7f3kKU4Ng5IDkgv6TvaPnrN')

args = parser.parse_args()

# Getting redis acount info and check if it's connected.
REDIS_HOST = args.host
REDIS_PORT = args.port
REDIS_USERNAME = args.user
REDIS_PASSWORD = args.password

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

mac_address = hex(uuid.getnode())
DAY = 24 * 60 * 60 * 1000
mac_battery = mac_address + ':battery'
mac_power = mac_address + ':power'
mac_plugged = mac_address + ':plugged_seconds'

# Creating keys with retention time which we calculated base on the memory size
try:
    redis_client.ts().create(mac_battery, chunk_size=128, retention_msecs=3276800 * 1000)
    redis_client.ts().create(mac_power, chunk_size=128, retention_msecs=3276800 * 1000)
    redis_client.ts().create(mac_plugged, chunk_size=128, retention_msecs=56623104000 * 1000)

    # Creating a rule to get sum of mac-power every day.
    redis_client.ts().createrule(mac_power, mac_plugged, 'sum', bucket_size_msec=DAY)

except redis.ResponseError:
    pass

# Get the status of battery and power every one second and add them to their keys.
while True:
    timestamp = time()
    battery_level = psutil.sensors_battery().percent
    power_plugged = int(psutil.sensors_battery().power_plugged)

    timestamp_ms = int(time() * 1000)  # Redis TS requires the timestamp in ms

    redis_client.ts().add(mac_battery, timestamp_ms, battery_level)
    redis_client.ts().add(mac_power, timestamp_ms, power_plugged)

    print(f'power-plugged status is {power_plugged}')
    sleep(1)
