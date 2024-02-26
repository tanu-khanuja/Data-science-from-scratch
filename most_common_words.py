from collections import Counter
import sys

num = sys.argv[1]

try:
    num = int(sys.argv[1])

except:
    print("usage: python most_common_words.py num")
    sys.exit(1)               # nonzero exit code indicates error

c = Counter(word.lower() 
            for line in sys.stdin 
            for word in line.strip().replace('.', ' ').split()
            if word)         
                             #strips whitespaces/newlines/tabs characters from start-end of line first
                             #replace full stop with whitespace
                             #splits words at space 

for word, count in c.most_common(num):      #c is dictionary of words and its counts
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")
    
           


