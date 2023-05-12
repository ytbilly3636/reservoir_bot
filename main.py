import os
import pickle
import cupy

import discord
import MeCab
import gensim
from reservoir import Reservoir
from reservoir import ridge_regression


# preference
debug = True                # True -> print is available
R_SIZE = 100                # size of reservoir
PARAM_PATH = 'params.bin'   # filename of parameters


# discord client
client = discord.Client()


# mecab for morphological analysis in Japanese
m_wakati = MeCab.Tagger("-Owakati")

# word2vec model
wv_model = gensim.models.Word2Vec.load('word2vec.gensim.model').wv

# function converting message to input vectors for reservoir
def message2vec(message):
    words = m_wakati.parse(message).split()
    
    if debug:
        print('MeCab result:', words)
    
    vecs = []
    for word in words:
        if word in wv_model:
            vecs.append(wv_model[word])
        else:
            vecs.append(wv_model['-'])
            
    return vecs


# reservoir
res = Reservoir(i_size=wv_model.vector_size, r_size=200)
w_o = None


# training data
#   1. if the client receives a message, it stores the message id, reservoir reaction to the message, and label (=0)
#   2. if the client receives a reaction, it rewrite stored label of the message
#   3. optimized weight connection of readout is computed using stored state and label
buffer_id = []
buffer_state = []
buffer_label = []


# save parameters
def save(path):
    params = {
        'w_i': res.w_i,
        'w_r': res.w_r, 
        'w_o': w_o,
        'leak': res.leak,
        'buf_i': buffer_id, 
        'buf_s': buffer_state,
        'buf_l': buffer_label
    }

    with open(path, 'wb') as p:
        pickle.dump(params, p)
    
# load parameters
def load(path):
    with open(path, 'rb') as p:
        params = pickle.load(p)
        
    global w_o
    global buffer_id
    global buffer_state
    global buffer_label
    res.w_i = params['w_i']
    res.w_r = params['w_r']
    res.leak = params['leak']
    w_o = params['w_o']
    buffer_id = params['buf_i']
    buffer_state = params['buf_s']
    buffer_label = params['buf_l']


# login
@client.event
async def on_ready():
    if os.path.exists(PARAM_PATH):
        if debug:
            print('load params')
        load(PARAM_PATH)

    print('Login!')

# receive message
@client.event
async def on_message(message):
    if message.author.bot:
        return

    # message -> vectors
    vecs = message2vec(message.content)

    # feeding
    res.reset(batch=1)
    xs = []
    for vec in vecs:
        x = res(cupy.array(vec.reshape(1, -1)))
        xs.append(x)
    xs = cupy.concatenate(xs, axis=0)
    
    # storing
    global buffer_id
    global buffer_state
    global buffer_label
    buffer_id.append(message.id)
    buffer_state.append(xs)
    buffer_label.append(0)

    # if not trained yet -> finish
    global w_o
    if w_o is None:
        return

    # prediction and pin the message
    y = xs.dot(w_o.T)
    label_pred = cupy.argmax(cupy.sum(y, axis=0))
    if debug:
        print('Prediction:', label_pred, cupy.sum(y, axis=0))

    if label_pred == 1:
        if debug:
            print('Try to pin')
            
        try:
            await message.pin()
        except:
            print('cannot pin')

# receive reaction
@client.event
async def on_reaction_add(reaction, user):
    if reaction.message.author.bot:
        return

    if debug:
        print('Training starts')

    # change label of reactioned message
    global buffer_id
    global buffer_state
    global buffer_label
    id = buffer_id.index(reaction.message.id)
    buffer_label[id] = 1

    # training
    ts = []
    for state, label in zip(buffer_state, buffer_label):
        t = cupy.zeros((state.shape[0], 2), dtype=cupy.float32)
        t[:, label] = 1.0
        ts.append(t)

    global w_o
    w_o = ridge_regression(cupy.concatenate(buffer_state, axis=0), cupy.concatenate(ts, axis=0))

    # save parameters
    if debug:
        print('Training is done')
        print('save params')
    save(PARAM_PATH)


# start bot
with open('token.txt') as f:
    TOKEN = f.read()

client.run(TOKEN)
