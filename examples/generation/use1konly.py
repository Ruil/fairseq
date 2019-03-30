from nltk.tokenize import sent_tokenize, word_tokenize
import os

data_path = '/home/liurui/data/writingPrompts/'
data = ["toy.train", "toy.test"] # train
out_path = './generation/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for name in data:
  with open(data_path + name + ".wp_target") as f:
    stories = f.readlines()
    stories = [" ".join(i.split()[0:1000]) for i in stories]
    tgt_stories = []
    src_stories = []
    for story in stories:
      sentences = sent_tokenize(story)
      print('sentences')
      print(len(sentences))
      #break
      tgt_stories = tgt_stories + sentences
      for idx, sentence in enumerate(sentences):
        cp_sentences = sentences[:]
        cp_sentences[idx] = '|||'
        cur = ' '.join(cp_sentences).strip()
        #print(cur)
        src_stories.append(cur)  
      
    #break
    with open(out_path + name + ".tags.tgt", "w") as o:
      for line in tgt_stories:
        o.write(line.strip() + "\n")

    with open(out_path + name + ".tags.src", "w") as o:
      for line in src_stories:
        o.write(line.strip() + "\n")
