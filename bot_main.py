# Copyright 2017 Aniruddh Chandratre. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

from telegram.ext import *

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.
    
gConfig = {}
logs_path = os.getcwd()


def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def create_model(session, forward_only):

  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'], gConfig['dec_vocab_size'], _buckets, gConfig['layer_size'], gConfig['num_layers'], gConfig['max_gradient_norm'], gConfig['batch_size'], gConfig['learning_rate'], gConfig['learning_rate_decay_factor'], forward_only=forward_only)

  if 'pretrained_model' in gConfig:
      model.saver.restore(session,gConfig['pretrained_model'])
      return model

  ckpt = tf.train.get_checkpoint_state(gConfig['working_directory'])
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def decode(sentence):
    print(sentence)
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(_buckets))
                     if _buckets[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                    target_weights, bucket_id, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    response = 'I am sorry, but I guess I am not capable enough to answer that yet'
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # Print out French sentence corresponding to outputs.
        response = ' ' .join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
        if " ' " in response:
            response = response.replace(" ' ", "'")
        if " . "  in response:
            response = response.replace(" . ", ".")
        if " ." in response:
            response = response.replace(" .", ".")
        print(response)
    return response


def start(bot, update):
    update.messsage.reply_text('Hi! Message me so that we can chat!')

def hello(bot, update):
    update.message.reply_text('Hello {}'.format(update.message.from_user.first_name))

def handle(bot, update):
    text = 'Message from %s : %s' % (update.message.from_user.first_name,update.message.text)
    print(text)
    if 'What' in update.message.text and 'your' in update.message.text and 'name' in upadte.message.text:
        print('Name')
        bot.sendMessage(update.message.chat_id, reply_to_message_id=update.message.message_id, text='My name is Nova!')
    else:
        response = decode(update.message.text)
        bot.sendMessage(update.message.chat_id, reply_to_message_id=update.message.message_id, text=response)

def init_session(sess, conf='seq2seq.ini'):
    global gConfig
    gConfig = get_config(conf)
 
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    return sess, model, enc_vocab, rev_dec_vocab

sess = tf.Session()
[sess, model, enc_vocab, rev_dec_vocab] = init_session(sess)

handle_handler = MessageHandler(Filters.text, handle)

updater = Updater('425777490:AAHKcQd73guagyH5a-W3nKgLUrR7iyNqzZA')

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('hello', hello))
updater.dispatcher.add_handler(handle_handler)
updater.start_polling()
updater.idle()

if __name__ == '__main__':
    if len(sys.argv) - 1:
        gConfig = get_config(sys.argv[1])
    else:
        # get configuration from seq2seq.ini
        gConfig = get_config()

    print('Bot started')
