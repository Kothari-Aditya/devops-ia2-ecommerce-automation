# Gunicorn configuration file
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# Worker processes
workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "delivery_risk_api"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = None
certfile = None