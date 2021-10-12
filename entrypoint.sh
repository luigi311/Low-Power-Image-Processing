#!/bin/bash

help () {
    echo "Usage: ./entrypoint.sh [COMMANDS] [FLAGS]"
    echo "COMMANDS:"
    echo "  --help, -h: Print this help message"
    echo "  auto_stack: Stacks all images in a folder for better quality"
    echo "  opencv_super_resolution: Super resolution the image"
    echo "FLAGS: Flags to pass to command called"
}

COMMANDS="$1"
shift # short for shift 1

for i
do 
    # Save all arguments in a variable FLAGS
    FLAGS="$FLAGS $i"
done

# if COMMANDS is "--help" or "-h"
if [ "$COMMANDS" = "--help" ] || [ "$COMMANDS" = "-h" ]
then
    help
# else if COMMANDS is "auto_stack"
elif [ "$COMMANDS" = "auto_stack" ]
then
    python stacking/auto_stack/auto_stack.py $FLAGS
# else if COMMANDS is "opencv_super_resolution"
elif [ "$COMMANDS" = "opencv_super_resolution" ]
then
    python super_resolution/opencv_super_resolution/opencv_super_resolution.py $FLAGS
# else
else
    echo "Unknown command: $COMMANDS"
    help
fi
