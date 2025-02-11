import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

embeds = nn.Embedding(2, 5)
words_to_index = {'hello': 0, 'world': 1}
lookup_tensor = torch.tensor([words_to_index['hello']], dtype=torch.long)
embed = embeds(lookup_tensor)
# print(embed)

test_sentence = """Once upon a time, in a small village nestled between lush green hills, there lived a young girl named Elara. She had a wild imagination and often dreamed of exploring the world beyond her village. Every day after finishing her chores, Elara would sit by the riverbank, sketching the adventures she wished to have.
One sunny afternoon, while drawing a magnificent dragon soaring through the clouds, Elara noticed something sparkling in the water. Curiosity piqued, she leaned closer and discovered a beautiful, shimmering stone. It glowed with an ethereal light, and as she picked it up, she felt a surge of energy coursing through her.
That night, Elara placed the stone under her pillow, and as she drifted off to sleep, she was whisked away to a magical realm filled with vibrant colors and fantastical creatures. In this world, she met a wise old owl who spoke in riddles and a mischievous fox who guided her through enchanted forests.
Together, they embarked on a quest to find the Heart of the Forest, a mythical jewel said to grant one wish to those pure of heart. Along the way, they faced challenges, like crossing a bridge guarded by a grumpy troll and solving puzzles set by the cunning owl.
After a series of adventures, Elara and her friends reached the Heart of the Forest, a radiant gem surrounded by glowing flowers. As she held it in her hands, Elara realized her wish was not for riches or fame but for the courage to share her stories and inspire others.
With her newfound courage, Elara returned to her village, where she began telling tales of her adventures to anyone who would listen. The villagers were captivated, and soon, they gathered every evening to hear her stories. Elara’s imagination sparked a fire in their hearts, and the village became a place of creativity and joy.
From that day on, Elara not only explored new worlds in her dreams but also created adventures in the hearts of her friends, reminding everyone that the greatest journeys often start with a single story. And so, the girl who once dreamed by the riverbank became a beacon of inspiration for all, proving that magic exists wherever there is imagination.""".split()

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i+2]) for i in range(len(test_sentence) - 2)]
# print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
# print(word_to_ix)

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), 10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(1000):
    total_loss = 0
    for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        # 会累加梯度，所以每次训练前需要清零
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        # 反向传播更新梯度
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)
