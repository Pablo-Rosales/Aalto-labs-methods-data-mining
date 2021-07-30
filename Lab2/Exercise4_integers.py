# Pablo Rosales Rodríguez, Student ID 914769
# Exercise 4

import pandas as pd
import numpy as np

fp = open('worlddiscr.names') 
fo = open('worlddiscr_int.names', 'w')



words= [word.strip() for line in fp.readlines() for word in line.split(',') if word.strip()]
different_words = list(dict.fromkeys(words))

fp.close() 
fp = open('worlddiscr.names')
it = 0
for line in fp.readlines():
    for word in line.split(','):
        if word.strip():
            index = different_words.index(word.strip())
            fo.write('%d' %index)
            fo.write(' ')
    if it > 0:     
        fo.write("\n")
    it = it+1
            

        
fo.close()
fp.close()

# In order to study the obtained rules, the integers associated to:
# exColony, mostCorrupted, leastCorrupted, compulsoryEducation<9y, 
# compulsoryEducation<7y and oil must be found

exColony_int = different_words.index("exColony")
mostCorrupted_int = different_words.index("mostCorrupted")
leastCorrupted_int = different_words.index("leastCorrupted")
compulsoryEducation9_int = different_words.index("compulsoryEducation<9y")
compulsoryEducation7_int = different_words.index("compulsoryEducation<7y")
oil_int = different_words.index("oil")

print("\nexColony: ", exColony_int)
print("\nmostCorrupted: ", mostCorrupted_int)
print("\nleastCorrupted: ", leastCorrupted_int)
print("\ncompulsoryEducation<9y: ", compulsoryEducation9_int)
print("\ncompulsoryEducation<7y: ", compulsoryEducation7_int)
print("\noil: ", oil_int)

print("\n\n")
print(different_words[19])