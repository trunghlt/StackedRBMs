import numpy as np
from itertools import izip
from sklearn.neighbors import BallTree

class Embeddings(object):

    def __init__(self, embeddings, unknown):
        """ ``embeddings`` is a dictionary whose keys are words and values are
        numerical vectors """
        self._map = embeddings
        self.unknown = unknown
        self.words = embeddings.keys()

    def vector(self, word):
        return self._map[word] if word in self._map\
               else self._map[self.unknown]

    def build_balltree(self):
        T = []
        for word in self.words: 
            T.append(self._map[word])
        T = np.asarray(T)
        self.balltree = BallTree(T, leaf_size=2)

    def knn(self, word, K=20):
        dist, ids = self.balltree.query(self.vector(word), k=K)
        return [self.words[id] for id in ids[0]]

class CollbertWestonEmbeddings(Embeddings):

    def __init__(self, word_list_path, embeddings_path, unknown="UNKNOWN"):
        self.unknown = unknown
        self._map = {}
        MAX = float("-inf")
        fw, fe = open(word_list_path, "r"), open(embeddings_path, "r")
        for word, embeddings in izip(fw, fe):
            word = word.strip()
            self._map[word] = np.array([float(e) for e in embeddings.split(" ")])
            _max = np.max(np.abs(self._map[word]))
            if _max > MAX: MAX = _max
        
        # Normalisation
        for word in self._map:
            self._map[word] /= MAX 
        self.words = self._map.keys()

class GoogleEmbeddings(Embeddings):

    def __init__(self, vector_file, unknown="unknown"):
        self.unknown = unknown
        self._map = {}
        f = open(vector_file, "rb")
        first = f.readline()
        N, size = map(int, first.split())
        dt = np.dtype("(200,)f4")
        for i in xrange(N):
            w = ""
            while True:
                c = f.read(1)
                if c == " ": break
                w += c
            vec = np.fromfile(f, dt, count=1)
            self._map[w.strip()] = vec
        self.words = self._map.keys()
        f.close()
        
def main(args, options):
    print "Loading embeddings file..."
    embeddings = Embeddings(options.word_path, options.embedding_path)
    print "Buildiing ball tree..."
    embeddings.build_balltree()
    if args[0] == "neighbours":
        for word in args[1:]:
            print word, ":", embeddings.knn(word)
        

if __name__ == "__main__":
    # Parse command options
    from optparse import OptionParser
    parser = OptionParser(usage="Usage: %prog [options] param1 param2")    
    # Add more options here
    parser.add_option("-w", "--word-path", dest="word_path",
                default="Sentiment140/words.lst", action="store",
                help="Word list file path")
    parser.add_option("-e", "--embedding-path", dest="embedding_path",
                default="Sentiment140/embeddings/embeddings.txt", 
                action="store",
                help="Embeddings file path")
    options, args = parser.parse_args()
    main(args, options)

