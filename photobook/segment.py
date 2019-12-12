import os
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np

from .tokenizer import UtteranceTokenizer

# Loads a PhotoBook segment data set object from file
class SegmentDataset(Dataset):
    def __init__(self, data_dir, segment_file, vectors_file, split='train'):

        self.data_dir = data_dir
        self.split = split
        self.segment_file = self.split + '_' + segment_file

        # Load a PhotoBook dialogue segment data set
        with open(os.path.join(self.data_dir, self.segment_file), 'r') as file:
            self.temp_dialogue_segments = json.load(file)

        self.dialogue_segments = []
        for d in self.temp_dialogue_segments:

            if d['segment'] == []:
                d['segment'] = [0] #pad empty segment
                d['length'] = 1


            self.dialogue_segments.append(d)

        # Load pre-defined image features
        with open(os.path.join(data_dir, vectors_file), 'r') as file:
            self.image_features = json.load(file)

    # Returns the length of the data set
    def __len__(self):
        return len(self.dialogue_segments)

    # Returns a PhotoBook Segment object at the given index
    def __getitem__(self, index):
        return self.dialogue_segments[index]

    @staticmethod
    def get_collate_fn(device):

        def collate_fn(data):

            #print('collate',data)
            max_src_length = max(d['length'] for d in data)
            max_target_images = max(len(d['targets']) for d in data)
            max_num_images = max([len(d['image_set']) for d in data])

            #print(max_src_length, max_target_images, max_num_images)

            batch = defaultdict(list)

            for sample in data:
                for key in data[0].keys():

                    if key == 'segment':
                        padded = sample['segment'] \
                            + [0] * (max_src_length-sample['length'])
                        #print('seg', padded)

                    elif key == 'image_set':

                        padded = [int(img) for img in sample['image_set']]

                        padded = padded \
                            + [0] * (max_num_images-len(sample['image_set']))
                        #print('img', padded)

                    elif key == 'targets':

                        #print(sample['targets'])
                        padded = np.zeros(max_num_images)
                        padded[sample['targets']] = 1

                        #print('tar', padded)

                    else:
                        #length of segment in number of words
                        padded = sample[key]

                    batch[key].append(padded)

            for key in batch.keys():
                #print(key, batch[key])
                batch[key] = torch.Tensor(batch[key]).long().to(device)

            return batch

        return collate_fn

class SegmentBuilder():
    def __init__(self, ):
        self.tokenizer = UtteranceTokenizer()

    def build(self, messages, targets, image_set, vocab, method="word2index", tokenization="word_tokenize", speaker_lables=True, lowercase=True, splitting=True):
        dialogue_segment = {}

        dialogue_segment["segment"] = self.flatten_and_encode(messages, method, vocab, tokenization, speaker_lables, lowercase, splitting)
        dialogue_segment["image_set"] = [str(target.split('_')[-1].split('.')[0].lstrip('0')) for target in image_set]
        target_ids = [str(target.split('_')[-1].split('.')[0].lstrip('0')) for target in targets]
        dialogue_segment["targets"] = [dialogue_segment["image_set"].index(target) for target in target_ids]
        dialogue_segment["length"] = len(dialogue_segment["segment"])

        return dialogue_segment

    def flatten_and_encode(self, messages, method="word2index", vocab=None, tokenization="word_tokenize", speaker_lables=True, lowercase=True, splitting=True):
        """
        Concatenates the utterances of a dialogue segment and encodes them in the specified manner
        :param messages: list. List of Message objects
        :param method: String. Specifies the desired word encoding scheme
        :param vocab: Vocabulary object. Vocabulary for word encoding
        :param speaker_lables: bool. Set to False to disable adding speaker labels in the output string
        :return: list. A vector representation of the encoded dialogue segment
        """
        assert vocab, print("Warning: No vocabulary given!")

        last_speaker = None
        segment = []
        for message in messages:
            if message.type == "text":
                speaker = message.speaker
                if speaker_lables and last_speaker != speaker:
                    if tokenization == "word_tokenize":
                        segment.extend(vocab.encode(["-" + speaker + "-"]))
                    else:
                        segment.extend(vocab.encode(["<" + speaker + ">"]))
                    last_speaker = speaker
                segment.extend(vocab.encode(self.tokenizer.tokenize_utterance(message.text, tokenization, lowercase, splitting)))
        if method == "word2index":
            pass
        else:
            print("Warning: Encoding method not implemented.")
        return segment



