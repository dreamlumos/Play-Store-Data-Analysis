import numpy as np

def createDictionary(textArray, minWordLength, minOccurrences, extraStopWords=[], dropS=False, verbose=False):
    """
    Creates a dictionary containing the most common words (excluding English stop words).
    ndarray * int * int * boolean -> dict
    """
    
    # Making a dictionary of all words
    wordDict = dict()
    for text in textArray:
        words = text.split()
        for word in words:
            word = ''.join(filter(str.isalpha, word)).lower()
            #if len(word) > 3:
                #if word[-1] == 's':
                    #word = word[:-1]
            if len(word) >= minWordLength:
                count = wordDict.get(word, 0)
                wordDict.update({word: count+1})
    if verbose:
        print("Total number of unique words that fit requirements:", len(wordDict))

    # Removing words that appear less than minOccurrences times
    wordDict = {key:val for key, val in wordDict.items() if val >= minOccurrences}
    # Removing stop words
    stopWords = ['the', 'and', 'for', 'of', 'or', 'to']
    stopWords += extraStopWords
    for word in stopWords:
        wordDict.pop(word, 0)

    if verbose:
        print("Number of words that occur at least", minOccurrences, "times:", len(wordDict))
    # print(dict(sorted(wordDict.items(), key=lambda item: item[1])))

    # Reducing size of dictionary by taking most significant words 
    # Hypothesis : app creators choose buzzwords for their app names
    cleanWordDict = dict()
    for text in textArray:
        words = text.split()
        words = np.unique(words)
        for i in range(len(words)):
            words[i] = ''.join(filter(str.isalpha, words[i])).lower() # lowercase and alpha

        occurrences = [wordDict.get(word, 0) for word in words]
        k = 3 # pick the k most significant words in the text

        if len(occurrences) <= k:
            for i in range(len(occurrences)):
                if occurrences[i] > 0:
                    count = cleanWordDict.get(words[i], 0)
                    cleanWordDict.update({words[i]: count+1})
        else:
            indices = np.argpartition(occurrences, k)
            indices = indices[:k]
            for i in indices:
                if occurrences[i] > 0:
                    count = cleanWordDict.get(words[i], 0)
                    cleanWordDict.update({words[i]: count+1})

    cleanWordDict = {key:val for key, val in cleanWordDict.items() if val >= minOccurrences}
    if verbose:
        print("Length of final dictionary:", len(cleanWordDict))
    return cleanWordDict

def convertText(textArray, dictionary, dropS=False):
    """
    Converts an array of texts to a matrix indicating the usage of words that are in the dictionary.
    Each row represents a text, each column represents a word in the dictionary.
    """

    rows = len(textArray)
    columns = len(dictionary)
    convertedText = np.zeros((rows, columns))

    indexedDict = dict()
    keys = dictionary.keys()
    i = 0
    for word in keys:
        indexedDict.update({word: i})
        i += 1

    i = 0
    for i in range(len(textArray)):
        words = textArray[i].split()
        for word in words:
            word = ''.join(filter(str.isalpha, word)).lower()
            #if len(word) > 3:
                #if word[-1] == 's':
                    #word = word[:-1]
            j = indexedDict.get(word, -1)
            if j >= 0:
                convertedText[i,j] = 1
                    
    return convertedText