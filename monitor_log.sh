#!/bin/bash

# Directory where log files are stored
LOG_DIR="./runs/DREAM"

# Check if a job ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

JOB_ID=$1

# Search for the log file with the given job ID
LOG_FILE=$(ls $LOG_DIR | grep "${JOB_ID}" | head -n 1)
LOG_FILE="${LOG_FILE}.log"

# Check if the log file exists
if [ -z "$LOG_FILE" ]; then
    echo "No log file found for job ID: $JOB_ID"
    exit 1
fi

# Display the log file in real time
echo "Displaying log file: $LOG_DIR/$LOG_FILE"
tail -f "$LOG_DIR/$LOG_FILE"