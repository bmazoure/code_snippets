from flask import Flask
from flask import request
from flask import render_template
import numpy as np
from lark import Lark, InlineTransformer
from numpy import add, subtract as sub, multiply as mul, divide as div, negative as neg
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def prepare():
    words_index = {}
    i_index = []
    embedding_matrix = []
    with open('embeddings/glove.6B.100d.txt',"r",encoding='utf8') as f:
        for i,line in enumerate(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix.append(coefs)
            words_index[word]=i
            i_index.append(word)
        f.close()
    embedding_matrix=np.array(embedding_matrix)
    return embedding_matrix,words_index, i_index
def prepareParser():
    grammar ='''?sum: product
                     | sum "+" product   -> add
                     | sum "-" product   -> sub

                 ?product: item
                     | product "*" item  -> mul
                     | product "/" item  -> div

                 ?item: WORD             -> word
                      | "-" item         -> neg
                      | "(" sum ")"

                 %import common.WORD
                 %import common.WS
                 %ignore WS
         '''
    return Lark(grammar, start='sum', ambiguity='explicit')
def parseInput(expr,parser):
    vec=parseExpr(parser.parse(expr))
    sims=cosine_similarity(vec,mat)
    return  idx2w[np.argmax(sims)]
def parseExpr(expr):
    if expr.data=="add":
        return add(parseExpr(expr.children[0]),parseExpr(expr.children[1]))
    elif expr.data=="sub":
        return sub(parseExpr(expr.children[0]),parseExpr(expr.children[1]))
    elif expr.data=="mul":
        return mul(parseExpr(expr.children[0]),parseExpr(expr.children[1]))
    elif expr.data=="div":
        return div(parseExpr(expr.children[0]),parseExpr(expr.children[1]))
    elif expr.data=="neg":
        return neg(parseExpr(expr.children[0]))
    elif expr.data=="word":
        word=expr.children[0].value.lower()
        if word in w2idx.keys():
            return mat[w2idx[word]]
        else:
            return np.zeros(100)
            
mat,w2idx,idx2w=prepare()
parser=prepareParser()

@app.route('/')
def my_form():
    return render_template("main.html")
    
@app.route('/suggestions')
def my_form_post():
    text = request.args.get('jsdata')
    return render_template('suggestions.html', suggestion=parseInput(text,parser))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
