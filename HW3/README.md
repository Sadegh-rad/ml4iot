# Homework 3 - REST API and MQTT Communication

## Description
This homework implements REST API services and MQTT communication protocols for IoT applications, including system monitoring and data exchange.

## Files
- `publisher.py` - MQTT publisher for system monitoring data
- `subscriber.ipynb` - MQTT subscriber implementation (Jupyter notebook)
- `rest_server.ipynb` - REST API server implementation (Jupyter notebook)
- `rest_client.ipynb` - REST API client implementation (Jupyter notebook)
- `mpl/` - Additional implementations directory
  - `rest_client.ipynb` - Alternative REST client implementation
  - `ReadMe.txt` - Additional documentation
- `ML4IoT-HW3_Problem_description.pdf` - Problem statement and requirements
- `Team18_Homework3.pdf` - Final report and solutions

## Key Features
- MQTT publisher/subscriber pattern implementation
- REST API server and client development
- System monitoring (CPU, memory, etc.)
- Real-time data communication
- IoT device simulation and monitoring

## Dependencies
- paho-mqtt
- psutil
- uuid
- json
- Flask/FastAPI (for REST API)
- requests (for REST client)

## Usage

### MQTT Publisher
```bash
python publisher.py
```

### REST Server and Client
Open the respective Jupyter notebooks:
- `rest_server.ipynb` - Start the REST API server
- `rest_client.ipynb` - Test the REST API endpoints
- `subscriber.ipynb` - Subscribe to MQTT messages

## Implementation Details
- MQTT communication using Eclipse Mosquitto broker
- System metrics collection and publishing
- RESTful API design and implementation
- Client-server communication patterns
- Real-time data streaming and monitoring
