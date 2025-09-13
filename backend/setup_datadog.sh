#!/bin/bash
# setup_datadog.sh - Script to set up Datadog environment variables for the HackMIT 2025 backend

# Print ASCII art banner
cat << "EOF"
 _   _            _    __  __ _____ _____   ____   ___ ____  ____  
| | | | __ _  ___| | _|  \/  |_   _|_   _| |___ \ / _ \___ \| ___| 
| |_| |/ _` |/ __| |/ / |\/| | | |   | |     __) | | | |__) |___ \ 
|  _  | (_| | (__|   <| |  | | | |   | |    / __/| |_| / __/ ___) |
|_| |_|\__,_|\___|_|\_\_|  |_| |_|   |_|   |_____|\___/_____|____/ 
                                                                   
 ____        _        _                 
|  _ \  __ _| |_ __ _| |__   __ _ _ __  
| | | |/ _` | __/ _` | '_ \ / _` | '_ \ 
| |_| | (_| | || (_| | |_) | (_| | | | |
|____/ \__,_|\__\__,_|_.__/ \__, |_| |_|
                            |___/       
EOF

echo "Setting up Datadog environment variables for monitoring and profiling..."

# Set default values
DEFAULT_ENV="development"
DEFAULT_SERVICE="hackmit-backend"
DEFAULT_VERSION="0.1.0"
DEFAULT_AGENT_HOST="localhost"
DEFAULT_TRACE_AGENT_PORT="8126"
DEFAULT_PROFILING_ENABLED="true"
DEFAULT_TRACE_SAMPLE_RATE="1.0"
DEFAULT_DEMO_SLOW_PROCESSING="false"

# Check if we're in an interactive shell
if [ -t 0 ]; then
    # Ask for environment values interactively
    read -p "Environment [${DEFAULT_ENV}]: " DD_ENV
    read -p "Service name [${DEFAULT_SERVICE}]: " DD_SERVICE
    read -p "Service version [${DEFAULT_VERSION}]: " DD_VERSION
    read -p "Datadog agent host [${DEFAULT_AGENT_HOST}]: " DD_AGENT_HOST
    read -p "Trace agent port [${DEFAULT_TRACE_AGENT_PORT}]: " DD_TRACE_AGENT_PORT
    read -p "Enable profiling (true/false) [${DEFAULT_PROFILING_ENABLED}]: " DD_PROFILING_ENABLED
    read -p "Trace sample rate (0.0-1.0) [${DEFAULT_TRACE_SAMPLE_RATE}]: " DD_TRACE_SAMPLE_RATE
    read -p "Demo slow processing (true/false) [${DEFAULT_DEMO_SLOW_PROCESSING}]: " DEMO_SLOW_PROCESSING
    
    # Use defaults for empty values
    DD_ENV=${DD_ENV:-$DEFAULT_ENV}
    DD_SERVICE=${DD_SERVICE:-$DEFAULT_SERVICE}
    DD_VERSION=${DD_VERSION:-$DEFAULT_VERSION}
    DD_AGENT_HOST=${DD_AGENT_HOST:-$DEFAULT_AGENT_HOST}
    DD_TRACE_AGENT_PORT=${DD_TRACE_AGENT_PORT:-$DEFAULT_TRACE_AGENT_PORT}
    DD_PROFILING_ENABLED=${DD_PROFILING_ENABLED:-$DEFAULT_PROFILING_ENABLED}
    DD_TRACE_SAMPLE_RATE=${DD_TRACE_SAMPLE_RATE:-$DEFAULT_TRACE_SAMPLE_RATE}
    DEMO_SLOW_PROCESSING=${DEMO_SLOW_PROCESSING:-$DEFAULT_DEMO_SLOW_PROCESSING}
else
    # Use defaults for non-interactive mode
    DD_ENV=$DEFAULT_ENV
    DD_SERVICE=$DEFAULT_SERVICE
    DD_VERSION=$DEFAULT_VERSION
    DD_AGENT_HOST=$DEFAULT_AGENT_HOST
    DD_TRACE_AGENT_PORT=$DEFAULT_TRACE_AGENT_PORT
    DD_PROFILING_ENABLED=$DEFAULT_PROFILING_ENABLED
    DD_TRACE_SAMPLE_RATE=$DEFAULT_TRACE_SAMPLE_RATE
    DEMO_SLOW_PROCESSING=$DEFAULT_DEMO_SLOW_PROCESSING
fi

# Create the output file for sourcing
cat > datadog_env.sh << EOF
#!/bin/bash
# Datadog environment variables for HackMIT 2025 backend
# Generated on $(date)
# Source this file before running the application:
# source datadog_env.sh

# Required variables
export DD_ENV="${DD_ENV}"
export DD_SERVICE="${DD_SERVICE}"
export DD_VERSION="${DD_VERSION}"

# Agent configuration
export DD_AGENT_HOST="${DD_AGENT_HOST}"
export DD_TRACE_AGENT_PORT="${DD_TRACE_AGENT_PORT}"

# Profiling and tracing configuration
export DD_PROFILING_ENABLED="${DD_PROFILING_ENABLED}"
export DD_TRACE_SAMPLE_RATE="${DD_TRACE_SAMPLE_RATE}"

# Demo configuration
export DEMO_SLOW_PROCESSING="${DEMO_SLOW_PROCESSING}"

# Additional recommended settings
export DD_LOGS_INJECTION="true"
export DD_RUNTIME_METRICS_ENABLED="true"

echo "Datadog environment variables have been set:"
echo "  - Environment: \${DD_ENV}"
echo "  - Service: \${DD_SERVICE}"
echo "  - Agent host: \${DD_AGENT_HOST}"
echo "  - Profiling enabled: \${DD_PROFILING_ENABLED}"
EOF

# Make the file executable
chmod +x datadog_env.sh

# Provide instructions
echo -e "\n\033[1;32mSetup complete!\033[0m"
echo "A file named 'datadog_env.sh' has been created in the current directory."
echo ""
echo "To activate these settings, run:"
echo "  source datadog_env.sh"
echo ""
echo "To run the application with Datadog monitoring, make sure the Datadog agent"
echo "is installed and running, then source the environment file before starting"
echo "the application."
echo ""
echo "To install the Datadog agent, visit:"
echo "  https://docs.datadoghq.com/agent/basic_agent_usage/"
echo ""
echo "For more information on monitoring FastAPI applications with Datadog, visit:"
echo "  https://docs.datadoghq.com/tracing/setup_overview/setup/python/"