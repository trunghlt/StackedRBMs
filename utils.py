#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
""" 

"""
import logging
from numpy import *
from scipy import *
from matplotlib.pyplot import *
import nltk

log = logging.getLogger(__name__)

def stdnorm(x):    
    return (x - x.mean())/x.std()

def parse_tree(tree):
    result = ""
    for e in tree:
        if type(e) is nltk.tree.Tree:
            result += parse_tree(e)
        elif e[1] == "NNP":
            result += " " + by
        else:
            result += " " + e[0]
    return result
        
def replace_nnp(sent, by="entity"):
    words = nltk.word_tokenize(sent)
    pos = nltk.pos_tag(words)
    tree = nltk.ne_chunk(pos, True)    
    return parse_tree(tree)

def padded(words, length=3):
      return ['PADDING']*length + words + ['PADDING']*length

def tiles(height, width, images, imwidth, imheight, welastic=False, helastic=False, figsize=(10, 10)):
    f = figure(figsize=figsize)
    ax = f.add_subplot(111)
    for i in xrange(height):
        for j in xrange(width):
            id = (i - 1)*width + j
            im = images[id]
            _imwidth = im.size/imheight if welastic else imwidth
            _imheight = im.size/imwidth if helastic else imheight
            assert _imwidth*_imheight==im.size, "%d %d != %d" % (_imwidth, _imheight, im.size)
            im = im.reshape(_imheight, _imwidth)
            ax.imshow(im, 
                      extent = array([j*imwidth, j*imwidth +_imwidth, 
                                      i*imheight, i*imheight + _imheight]), 
                      cmap=gray())

    xlim(0,imwidth * width)
    ylim(0,imheight * height)
