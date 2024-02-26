import re, sys

regex = sys.argv[1]

#for every line passed into the script
for line in sys.stdin:
    if re.search(regex, line):            #first comes regex then line
        sys.stdout.write(line)