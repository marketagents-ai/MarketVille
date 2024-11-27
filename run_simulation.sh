#!/bin/bash

# Script to run market simulation with parallel orchestration, dashboard, and time tracking

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# File names for the Python scripts
ORCHESTRATOR_SCRIPT="market_agents/orchestrators/meta_orchestrator.py"
DASHBOARD_SCRIPT="market_agents/agents/db/dashboard/dashboard.py"
DASHBOARD_PORT=8000

# Check if the Python scripts exist
if [ ! -f "$ORCHESTRATOR_SCRIPT" ]; then
    echo "Error: $ORCHESTRATOR_SCRIPT not found!"
    exit 1
fi

if [ ! -f "$DASHBOARD_SCRIPT" ]; then
    echo "Error: $DASHBOARD_SCRIPT not found!"
    exit 1
fi

# Check if dashboard is already running
if check_port $DASHBOARD_PORT; then
    echo "Dashboard is already running at http://localhost:$DASHBOARD_PORT"
    DASHBOARD_STARTED=false
else
    # Start the dashboard in the background
    echo "Starting dashboard..."
    python3 "$DASHBOARD_SCRIPT" &
    DASHBOARD_PID=$!
    DASHBOARD_STARTED=true

    # Give the dashboard a moment to start up
    sleep 2
    echo "Dashboard is running. Access it at http://localhost:$DASHBOARD_PORT"
fi

# Get the start time
start_time=$(date +%s)

# Run the orchestrator script and print its output in real-time
echo "Starting market simulation with parallel orchestration..."
python3 "$ORCHESTRATOR_SCRIPT" --environments group_chat auction 2>&1 | tee simulation_output.log
#python3 "$ORCHESTRATOR_SCRIPT" --environments auction 2>&1 | tee simulation_output.log

orchestrator_exit_code=${PIPESTATUS[0]}

# Get the end time for the orchestrator
end_time=$(date +%s)

# Calculate the duration
duration=$((end_time - start_time))

# Print the results
echo "----------------------------------------"
echo "Simulation completed with exit code: $orchestrator_exit_code"
echo "Total execution time: $duration seconds"
echo "----------------------------------------"
echo "Full output has been saved to simulation_output.log"
echo "----------------------------------------"

# Check if the orchestrator script ran successfully
if [ $orchestrator_exit_code -ne 0 ]; then
    echo "Error: Orchestrator script failed with exit code $orchestrator_exit_code"
else
    echo "Simulation completed successfully."
fi

# Only prompt to stop dashboard if we started it in this session
if [ "$DASHBOARD_STARTED" = true ]; then
    echo "Dashboard was started by this script. Press Enter to stop the dashboard and exit."
    read
    kill $DASHBOARD_PID
else
    echo "Dashboard was already running and will continue running."
fi

exit $orchestrator_exit_code