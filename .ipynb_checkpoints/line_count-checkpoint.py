#line_count.py

import sys

count = 0
for line in sys.stdin:
    count += 1

print(count)                    #by default print goes to sys.stdout, therefore it writes in console