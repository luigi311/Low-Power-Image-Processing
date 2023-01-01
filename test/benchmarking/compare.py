import sys

# check for correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python percentage_change.py file1 file2")
    sys.exit()

# read in the two files
try:
    with open(sys.argv[1], "r") as f1:
        num1 = float(f1.read())
    with open(sys.argv[2], "r") as f2:
        num2 = float(f2.read())
except ValueError:
    print("Error: Input files must contain a single number")
    sys.exit()

# calculate and print the percentage change
percent_change = (num2 - num1) / num1 * 100
print(f"Percentage change: {percent_change:.4f}%")