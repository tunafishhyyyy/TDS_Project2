#!/bin/bash
# Usage: ./server_8001.sh start|stop|status

SERVER_CMD="/work/tanmay/TDS/TDS_Project2/Project2/.venv/bin/python chains/main_app.py"
LOG_FILE="main_app_server.log"
PID_FILE="server_8001.pid"
PORT=8001

start() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Server already running on port $PORT (PID $(cat $PID_FILE))"
        exit 0
    fi
    echo "Starting server on port $PORT..."
    nohup $SERVER_CMD > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    echo "Started (PID $(cat $PID_FILE)). Log: $LOG_FILE"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 $PID 2>/dev/null; then
            echo "Stopping server (PID $PID)..."
            kill $PID && rm -f "$PID_FILE"
            echo "Stopped."
        else
            echo "No running process found for PID $PID. Removing stale PID file."
            rm -f "$PID_FILE"
        fi
    else
        echo "No PID file found. Server may not be running."
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Server is running on port $PORT (PID $(cat $PID_FILE))"
    else
        echo "Server is not running on port $PORT."
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 start|stop|status"
        exit 1
        ;;
esac
