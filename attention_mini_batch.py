
# coding: utf-8

# In[2]:

import _gdynet as dy
#dy.init()
dyparams = dy.DynetParams()
# Fetch the command line arguments (optional)
# Set some parameters manualy (see the command line arguments documentation)
dyparams.set_mem(10000)
dyparams.init()


# In[3]:

import random
import codecs
import numpy as np


# In[4]:

import nltk


# In[5]:

TRAIN_TOK_EN = '../data/en-de/train.en-de.low.filt.en'
TRAIN_TOK_DE = '../data/en-de/train.en-de.low.filt.de'


# In[6]:

TEST_TOK_EN = '../data/en-de/valid.en-de.low.en'
TEST_TOK_DE = '../data/en-de/valid.en-de.low.de'


# In[7]:

additional_stuff = ['<UNK>']

class Data(object):
    def __init__(self, train_file):
        self.train_file = train_file
        with codecs.open(self.train_file, 'r') as open_file:
            data = open_file.readlines()
        
        data = [x.strip() for x in data]
        self.sent_data = data
        self.max_len = 0
        
    def get_data_dict(self):      
        words = []
        for i, sent in enumerate(self.sent_data):
            split_sent = sent.split()
            words.extend(split_sent)
            
        words = list(set(words))
        words.extend(additional_stuff)
        words = sorted(words)
        words.insert(0, '<s>')
        words.insert(0, '<EOS>')
        
        id2word = {v: k for v, k in enumerate(words)}
        word2id = {k: v for v, k in enumerate(words)}
        return id2word, word2id
        
    def get_training_data(self):
        id2word, word2id = self.get_data_dict()
        
        X = []
        for i, sent in enumerate(self.sent_data):
            words = sent.split() 
            words.insert(0, '<s>')
            words.append('<EOS>')
            #print sent
            sent_indices = [word2id[x] for x in words]
            X.append(sent_indices)
        self.X = X
        return id2word, word2id, X
    
#     def get_mini_batched_data(self):
#         n = self.minibatch_size
        
#         minibatched_X = [self.X[i:i+n] for i in range(0, len(self.X), n)]
        
#         for i, mini_batch in enumerate(minibatched_X):
#             #print mini_batch
#             max_len = max([len(x) for x in mini_batch])
#             for j, sent in enumerate(mini_batch):
#                 minibatched_X[i][j].extend([word2id_en['<EOS>']]* (max_len - len(sent)))
        
#         return minibatched_X


# In[8]:
file_eng_obj = Data(TRAIN_TOK_EN)
id2word_en, word2id_en, X_en = file_eng_obj.get_training_data()


# In[9]:
file_ger_obj = Data(TRAIN_TOK_DE)
id2word_de, word2id_de, X_de = file_ger_obj.get_training_data()


# In[10]:

## HYPERPARAMETERS 

VOCAB_SIZE_EN = len(word2id_en.keys())
VOCAB_SIZE_DE = len(word2id_de.keys())

LSTM_NUM_OF_LAYERS = 2
EMBEDDINGS_SIZE = 512
STATE_SIZE = 512
ATTENTION_SIZE = 256
BATCH_SIZE = 32
DROPOUT = 0.0



# In[11]:

model = dy.Model()

encoder_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
encoder_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

input_lookup = model.add_lookup_parameters((VOCAB_SIZE_DE, EMBEDDINGS_SIZE))

decoder_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
attention_v = model.add_parameters( (1, ATTENTION_SIZE))
decoder_w = model.add_parameters( (VOCAB_SIZE_EN, 3*STATE_SIZE))
decoder_b = model.add_parameters( (VOCAB_SIZE_EN))
output_lookup = model.add_lookup_parameters((VOCAB_SIZE_EN, EMBEDDINGS_SIZE))

trainer = dy.SimpleSGDTrainer(model)


# In[12]:

def minibatched_data(X_de, X_en, n):
    train_data = []
    for i, sent_de in enumerate(X_de):
        train_data.append([sent_de, X_en[i]])
    
    train_data.sort(key=lambda item: (-len(item[0]), item))
    minibatched_X = [train_data[i:i+n] for i in range(0, len(train_data), n)]
    return minibatched_X


# def generate_validation_set(file_name, lang):
#     if lang == 'en':
#         word2id, id2word = word2id_en, id2word_en
#     else:
#         word2id, id2word = word2id_de, id2word_de

#     with codecs.open(self.train_file, 'r') as open_file:
#         data = open_file.readlines()
        
#     data = [x.strip() for x in data]
#     val_sents = []
#     for i, sent in enumerate(data):
#         words = sent.split()
#         id_sent = [word2id[word] for word in words]



# In[ ]:

EOS = 0

class LSTMAttention(object): 
    
    def __init__(self):
        pass
        
    def run_lstm(self, init_state, input_vecs):
        s = init_state
        out_vectors = s.transduce(input_vecs)
        return out_vectors
    
#     def reverse_sentence_batch(self, sentence_batch):
#         rev_sent_batch = []
#         for sent in sentence_batch:
#             print sent
#             rev_sent_batch.append(list(reversed(sent)))
#         return rev_sent_batch
    
    def create_sentence_batch(self, sentence_batch):
        S = 0
        sents = sentence_batch
        wids, masks = [], []
        for i in range(len(sentence_batch[0])):
            #print "INSIDE CREATE BATCH"
            wids.append([(sent[i] if len(sent)>i else S) for sent in sents])
            mask = [(1 if len(sent)>i else 0) for sent in sents]
            masks.append(mask)
        return wids, masks
   
    def embed_sentence_batch(self, sentence_batch):
        global input_lookup
        padded_sentence_batch, _ = self.create_sentence_batch(sentence_batch)
        batched_lookup = [dy.lookup_batch(input_lookup, wids) for wids in padded_sentence_batch]
        return batched_lookup

    def encode_sentence_batch(self, enc_fwd_lstm, enc_bwd_lstm, sentence_batch):
        sentence_rev_batch = list(reversed(sentence_batch)) #self.reverse_sentence_batch(sentence_batch)
        fwd_vectors = self.run_lstm(enc_fwd_lstm.initial_state(), sentence_batch)
        bwd_vectors = self.run_lstm(enc_bwd_lstm.initial_state(), sentence_rev_batch)
        bwd_vectors = list(reversed(bwd_vectors))
        
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors
    
    def attend(self, input_mat, state, w1dt, input_len, batch_size):
        global attention_w2
        global attention_v
        w2 = dy.parameter(attention_w2)
        v = dy.parameter(attention_v)
        w2dt = w2*dy.concatenate(list(state.s()))
        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        unnormalized = dy.reshape(unnormalized, (input_len, ), batch_size)
        att_weights = dy.softmax(unnormalized)
        
        context = input_mat * att_weights
        return context, att_weights
    
    def get_loss(self, input_sentence_batch, output_sentence_batch, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
        embedded = self.embed_sentence_batch(input_sentence_batch)
        encoded = self.encode_sentence_batch(encoder_fwd_lstm, encoder_bwd_lstm, embedded)
        return self.decode(dec_lstm, encoded, output_sentence_batch)
        
   
    def decode(self, dec_lstm, vectors, output_batch):
        # output = [EOS] + list(output) + [EOS]
        # output = [char2int[c] for c in output]

        w = dy.parameter(decoder_w)
        b = dy.parameter(decoder_b)
        w1 = dy.parameter(attention_w1)
        input_mat = dy.concatenate_cols(vectors)
        input_len = len(vectors)

        w1dt = w1 * input_mat

        curr_bsize = len(output_batch)
        
        #n_batch_start = [1] * curr_bsize
        #last_output_embeddings = dy.lookup_batch(output_lookup, n_batch_start)
        
        s = dec_lstm.initial_state()
        c_t_previous = dy.vecInput(STATE_SIZE*2)
        
        loss = []

        output_batch, masks = self.create_sentence_batch(output_batch)

        for i, (word_batch, mask_word) in enumerate(zip(output_batch[1:], masks[1:]), start=1):
            last_output_embeddings = dy.lookup_batch(output_lookup, output_batch[i-1])
            vector = dy.concatenate([c_t_previous, last_output_embeddings])
            s = s.add_input(vector)
            h_t = s.output()
            c_t, alpha_t = self.attend(input_mat, s, w1dt, input_len, curr_bsize)
            
            h_c_concat = dy.concatenate([h_t, c_t])
            out_vector = dy.affine_transform([b, w, h_c_concat])
            
            #if DROPOUT > 0.0:
            #    out_vector = dy.dropout(out_vector, DROPOUT)

            loss_current = dy.pickneglogsoftmax_batch(out_vector, output_batch[i])

            if 0 in mask_word:
                mask_vals = dy.inputVector(mask_word)
                mask_vals = dy.reshape(mask_vals, (1,), curr_bsize)
                loss_current = loss_current * mask_vals
                
            loss.append(loss_current)
            c_t_previous = c_t
 
        loss = dy.esum(loss)
        loss = dy.sum_batches(loss)/curr_bsize
        #perplexity = loss.value() * curr_bsize / float(sum([x.count(1) for x in masks[1:]]))
        return loss
    
    def get_validation_perplexity():
        pass


    def generate(self, in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
        embedded = self.embed_sentence(in_seq)
        encoded = self.encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

        w = dy.parameter(decoder_w)
        b = dy.parameter(decoder_b)
        w1 = dy.parameter(attention_w1)
        input_mat = dy.concatenate_cols(encoded)
        
        w1dt = w1 * input_mat
        
        last_output_embeddings = output_lookup[EOS]
        
        s = dec_lstm.initial_state()
        c_t_previous = dy.vecInput(STATE_SIZE*2)
        
        out = ''
        count_EOS = 0
        
        for i in range(len(in_seq)*2):
            if count_EOS == 2: break
            vector = dy.concatenate([last_output_embeddings, c_t_previous])
            s = s.add_input(vector)
            h_t = s.output()
            c_t, alpha_t = self.attend(input_mat, s, w1dt)
            
            h_c_concat = dy.concatenate([h_t, c_t])
            out_vector = dy.affine_transform([b, w, h_c_concat])
            
            probs = dy.softmax(out_vector).vec_value()
            
            next_char = probs.index(max(probs))
            last_output_embeddings = output_lookup[next_char]
            c_t_previous = c_t
            
            if next_char == EOS:
            
                count_EOS += 1
                continue

            out += " " + id2word_en[next_char]
            
        return out
            
     
    def train(self, train_set_de, train_set_en, n_epochs):
        batched_X_Y = minibatched_data(train_set_de, train_set_en, BATCH_SIZE)
        for i in range(n_epochs):
            random.shuffle(batched_X_Y)
            j = 0
            for batch in batched_X_Y:
                j += BATCH_SIZE
                batched_X = [x[0] for x in batch]
                batched_Y = [x[1] for x in batch]
                #print CURR_BATCH_SIZE, batched_X
                dy.renew_cg()
                enc_sentence, dec_sentence = batched_X, batched_Y
                loss = self.get_loss(enc_sentence, dec_sentence, encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm)
                loss_value = loss.value()
                loss.backward()
                trainer.update()
                perplexity = np.exp(loss.value() * len(batched_Y) / float(sum([len(sent) for sent in batched_Y])) )
                if j % 1000 == 0:
                    with codecs.open('models_batched/output_loss_samples.txt', 'a+') as open_file:
                        open_file.write("Epoch: " + str(i) + "   Samples:" + str(j) + "   LOSS: " + str(loss_value) + "   Perplexity: " + str(perplexity) + "\n")
                        print ("Epoch: " + str(i) + "   Samples:" + str(j) + "   LOSS: " + str(loss_value) + "   Perplexity: " + str(perplexity) )
                        #dy.renew_cg()
                        #open_file.write(self.generate(enc_sentence, encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm) + "\n")
                        #open_file.write("BLEU SCORE : " + str(self.calculate_bleu_score()) + "\n")
                if j % 20000 == 0:
                    model.save('models_batched/model_'+str(i) + '_' + str(j), [encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm, input_lookup, output_lookup,
                            attention_w1, attention_w2, attention_v, decoder_w, decoder_b])

            #print "Epoch : ", i


# In[ ]:

translation_model = LSTMAttention()
translation_model.train(X_de, X_en, 30)


