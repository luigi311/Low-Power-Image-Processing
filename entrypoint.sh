#!/usr/bin/env bash

help() {
    echo "Usage: ./entrypoint.sh [COMMAND] [FLAGS]"
    echo "COMMAND:"
    echo "  --help, -h              Print this help message"
    echo "  all_in_one              Use all_in_one"
    echo "FLAGS: Flags to pass to command called"
}

COMMAND="$1"
shift # short for shift 1

for i; do
    # Save all arguments in a variable FLAGS
    FLAGS="$FLAGS $i"
done

if [ "$COMMAND" = "--help" ] || [ "$COMMAND" = "-h" ]; then
    help
elif [ "$COMMAND" = "all_in_one" ]; then
    python all_in_one.py $FLAGS
else
    echo "Unknown command: $COMMAND calling directly"
    $COMMAND $FLAGS
fi
