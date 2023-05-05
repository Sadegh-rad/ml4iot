import psutil
import uuid
import time
import json
import paho.mqtt.client as mqtt

MAC_ADDRESS = hex(uuid.getnode())
CLIENT_ID = 's291462'


def on_connect(client, userdata, flags, rc):
    # rc is the result code to check connection status
    print(f'Connected with result code {str(rc)}')


client = mqtt.Client()
client.on_connect = on_connect
client.connect('mqtt.eclipseprojects.io', 1883)

while True:
    timestamp = int(time.time() * 1000)
    battery_level = psutil.sensors_battery().percent
    power_plugged = psutil.sensors_battery().power_plugged
    bat_dict = {
        'mac_address': MAC_ADDRESS,
        'timestamp': timestamp,
        'battery_level': battery_level,
        'power_plugged': power_plugged
    }
    print(json.dumps(bat_dict))
    client.publish(CLIENT_ID, json.dumps(bat_dict))
    time.sleep(1)
