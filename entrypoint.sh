#!/bin/bash

help() {
    echo "Usage: ./entrypoint.sh [COMMAND] [FLAGS]"
    echo "COMMAND:"
    echo "  --help, -h              Print this help message"
    echo "  auto_stack              Stacks all images in a folder for better quality"
    echo "  opencv_super_resolution Super resolution the image"
    echo "  ffdnet                  Denoise using ffdnet"
    echo "  ircnn                   Denoise using ircnn"
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
    python all_in_one/all_in_one.py $FLAGS
elif [ "$COMMAND" = "auto_stack" ]; then
    python stacking/auto_stack/auto_stack.py $FLAGS
elif [ "$COMMAND" = "opencv_super_resolution" ]; then
    python super_resolution/opencv_super_resolution/opencv_super_resolution.py $FLAGS
elif [ "$COMMAND" = "ffdnet" ]; then
    python denoise/ffdnet/ffdnet.py $FLAGS
elif [ "$COMMAND" = "ircnn" ]; then
    python denoise/ircnn/ircnn.py $FLAGS
else
    echo "Unknown command: $COMMAND calling directly"
    $COMMAND $FLAGS
fi
