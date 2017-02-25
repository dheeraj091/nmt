
# coding: utf-8

# In[1]:

import _gdynet as dy
#dy.init()
dyparams = dy.DynetParams()
# Fetch the command line arguments (optional)
# Set some parameters manualy (see the command line arguments documentation)
dyparams.set_mem(10000)
dyparams.init()


# In[2]:

import random
import codecs
import numpy as np


# In[3]:

import nltk


# In[4]:

TRAIN_TOK_EN = '../data/en-de/train.en-de.tok.filt.en'
TRAIN_TOK_DE = '../data/en-de/train.en-de.tok.filt.de'


# In[5]:

TEST_TOK_EN = '../data/en-de/test.en-de.low.en'
TEST_TOK_DE = '../data/en-de/test.en-de.low.de'


# In[6]:

additional_stuff = ['<UNK>']

class Data(object):
    def __init__(self, train_file, minibatch_size):
        self.train_file = train_file
        with codecs.open(self.train_file, 'r') as open_file:
            data = open_file.readlines()
        
        data = [x.strip().lower() for x in data]
        self.sent_data = data
        
        self.minibatch_size = minibatch_size
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
    
    
    def get_mini_batched_data(self):
        n = self.minibatch_size
        minibatched_X = [self.X[i:i+n] for i in range(0, len(self.X), n)]
        
        for i, mini_batch in enumerate(minibatched_X):
            #print mini_batch
            max_len = max([len(x) for x in mini_batch])
            for j, sent in enumerate(mini_batch):
                minibatched_X[i][j].extend([word2id_en['<PAD>']]* (max_len - len(sent)))
        
        return minibatched_X


# In[7]:

file_eng_obj = Data(TRAIN_TOK_EN, minibatch_size=10)
id2word_en, word2id_en, X_en = file_eng_obj.get_training_data()
#minibatched_eng_data = file_eng_obj.get_mini_batched_data()


# In[8]:

file_ger_obj = Data(TRAIN_TOK_DE, minibatch_size=10)
id2word_de, word2id_de, X_de = file_ger_obj.get_training_data()
#minibatched_ger_data = file_ger_obj.get_mini_batched_data()


# In[9]:

## HYPERPARAMETERS 

VOCAB_SIZE_EN = len(word2id_en.keys())
VOCAB_SIZE_DE = len(word2id_de.keys())

LSTM_NUM_OF_LAYERS = 2
EMBEDDINGS_SIZE = 512
STATE_SIZE = 512
ATTENTION_SIZE = 256


# In[10]:

model = dy.Model()

#encoder_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
#encoder_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

#input_lookup = model.add_lookup_parameters((VOCAB_SIZE_DE, EMBEDDINGS_SIZE))

#decoder_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

#attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
#attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
#attention_v = model.add_parameters( (1, ATTENTION_SIZE))
#decoder_w = model.add_parameters( (VOCAB_SIZE_EN, 3*STATE_SIZE))
#decoder_b = model.add_parameters( (VOCAB_SIZE_EN))
#output_lookup = model.add_lookup_parameters((VOCAB_SIZE_EN, EMBEDDINGS_SIZE))

[encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm, input_lookup, output_lookup, attention_w1, attention_w2, attention_v, decoder_w, decoder_b] = model.load('models_latest_1/model_12_40000')

trainer = dy.SimpleSGDTrainer(model)

def BLEU_score(reference, hypothesis):
    bleu = nltk.translate.bleu_score.corpus_bleu(reference, hypothesis)
    return bleu
    
def file_to_words(file_name):
    with codecs.open(file_name, 'r') as open_file:
        lines = [x.strip().split() for x in open_file.readlines()]
    return lines

def get_ids_from_sentences(list_of_sentences, dict_lang):
    ids = []
    
    for sent in list_of_sentences:
        wids = []
        for word in sent:
            if word in dict_lang:
                wids.append(dict_lang[word])
            else:
                wids.append(dict_lang['<UNK>'])
        ids.append(wids)
    
    return ids


def get_test_sentence_pairs(encoder='de'):
    if encoder == 'en':
        encoder_test_file, decoder_test_file = TEST_TOK_DE, TEST_TOK_EN
        encoder_dict, decoder_dict = word2id_en, word2id_de
    else:
        encoder_test_file, decoder_test_file = TEST_TOK_DE, TEST_TOK_EN
        encoder_dict, decoder_dict = word2id_de, word2id_en
        
    encoder_sentences = file_to_words(encoder_test_file)
    decoder_sentences = file_to_words(decoder_test_file)
    
    encoder_wids = get_ids_from_sentences(encoder_sentences, encoder_dict)
    decoder_wids = get_ids_from_sentences(decoder_sentences, decoder_dict)
    
    return encoder_sentences, encoder_wids, decoder_sentences, decoder_wids
    
# In[11]:

def randomize(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    return a, b


# In[12]:

EOS = 0

class LSTMAttention(object): 
    
    def __init__(self):
        pass
        
    def run_lstm(self, init_state, input_vecs):
        s = init_state
        out_vectors = []
        #print input_vecs, input_vecs.shape
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors
   
    def embed_sentence(self, sentence):
        #sentence = [word2id_en[word] for word in sentence]
        global input_lookup
        return [input_lookup[word] for word in sentence]

    def encode_sentence(self, enc_fwd_lstm, enc_bwd_lstm, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = self.run_lstm(enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = self.run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors
    
    def attend(self, input_mat, state, w1dt):
        global attention_w2
        global attention_v
        w2 = dy.parameter(attention_w2)
        v = dy.parameter(attention_v)

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = w2*dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context, att_weights
    
    def get_loss(self, input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
        dy.renew_cg()
        embedded = self.embed_sentence(input_sentence)
        encoded = self.encode_sentence(encoder_fwd_lstm, encoder_bwd_lstm, embedded)
        return self.decode(dec_lstm, encoded, output_sentence)
        
   
    def decode(self, dec_lstm, vectors, output):
        # output = [EOS] + list(output) + [EOS]
        # output = [char2int[c] for c in output]

        w = dy.parameter(decoder_w)
        b = dy.parameter(decoder_b)
        w1 = dy.parameter(attention_w1)
        input_mat = dy.concatenate_cols(vectors)
        
        w1dt = w1 * input_mat
        
        last_output_embeddings = output_lookup[EOS]
        
        s = dec_lstm.initial_state()
        c_t_previous = dy.vecInput(STATE_SIZE*2)
        
        loss = []

        for word in output:
            vector = dy.concatenate([last_output_embeddings, c_t_previous])
            s = s.add_input(vector)
            h_t = s.output()
            c_t, alpha_t = self.attend(input_mat, s, w1dt)
            
            h_c_concat = dy.concatenate([h_t, c_t])
            out_vector = dy.affine_transform([b, w, h_c_concat])
            
            loss_current = dy.pickneglogsoftmax(out_vector, word)
            last_output_embeddings = output_lookup[word]
            loss.append(loss_current)
            c_t_previous = c_t
            
        loss = dy.esum(loss)
        return loss
    
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
    
    def calculate_bleu_score(self):
        enc_sentences, enc_ids, dec_sentences, _ = get_test_sentence_pairs()
        
        generated_decoded_set = []
        for i, sent in enumerate(enc_ids):
            decoded_test_sent = self.generate(sent, encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm)
            generated_decoded_set.append(decoded_test_sent)
        
        return BLEU_score(dec_sentences, generated_decoded_set)
            
     
    def train(self, train_set_de, train_set_en, n_epochs):
        for i in range(13, n_epochs):
            a, b = randomize(train_set_de, train_set_en)
            train_set = zip(a,b)
            for j, train_sample in enumerate(train_set):
                enc_sentence, dec_sentence = train_sample[0], train_sample[1]
                loss = self.get_loss(enc_sentence, dec_sentence, encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm)
                loss_value = loss.value()
                loss.backward()
                trainer.update()
                if j % 1000 == 0:
                    with codecs.open('models_latest_1/output_loss_samples.txt', 'a+') as open_file:
                        open_file.write("Epoch: " + str(i) + "   Samples:" + str(j) + "   LOSS: " + str(loss_value) + "\n")
                        open_file.write(self.generate(enc_sentence, encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm) + "\n")
                        #open_file.write("BLEU SCORE : " + str(self.calculate_bleu_score()) + "\n")
                if j % 20000 == 0:
                    model.save('models_latest_1/model_'+str(i) + '_' + str(j), [encoder_fwd_lstm, encoder_bwd_lstm, decoder_lstm, input_lookup, output_lookup,
                            attention_w1, attention_w2, attention_v, decoder_w, decoder_b])

            #print "Epoch : ", i


# In[ ]:

translation_model = LSTMAttention()
translation_model.train(X_de, X_en, 20)


# In[ ]:



