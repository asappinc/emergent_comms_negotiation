import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F


class NumberSequenceEncoder(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(11, embedding_size)
        self.lstm = nn.LSTMCell(
            input_size=embedding_size,
            hidden_size=embedding_size)
        self.zero_state = None

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        x = x.transpose(0, 1)
        x = self.embedding(x)
        type_constr = torch.cuda if x.is_cuda else torch
        state = (
                Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0)),
                Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0))
            )
        for s in range(seq_len):
            state = self.lstm(x[s], state)
        return state[0]


class CombinedNet(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.embedding_size = embedding_size
        self.h1 = nn.Linear(embedding_size * 3, embedding_size)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        return x


class TermPolicy(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()
        self.h1 = nn.Linear(embedding_size, 1)

    def forward(self, x, eps=1e-8):
        x = self.h1(x)
        x = F.sigmoid(x)
        out_node = torch.bernoulli(x)
        x = x + eps
        entropy = - (x * x.log()).sum(1).sum()
        return out_node, out_node.data, entropy


class UtterancePolicy(nn.Module):
    def __init__(self, embedding_size=100, num_tokens=10, max_len=6):
        super().__init__()
        # use this to make onehot
        self.embedding_size = embedding_size
        self.onehot = torch.eye(num_tokens)
        self.num_tokens = num_tokens
        self.max_len = max_len
        self.lstm = nn.LSTM(
            input_size=num_tokens,
            hidden_size=embedding_size,
            num_layers=1
        )
        self.h1 = nn.Linear(embedding_size, num_tokens)

    def forward(self, h_t):
        batch_size = h_t.size()[0]

        state = (
            h_t.view(1, batch_size, self.embedding_size),
            Variable(torch.zeros(1, batch_size, self.embedding_size))
        )

        # use first token as the initial dummy token
        last_token = torch.zeros(batch_size).long()
        utterance_nodes = []
        while len(tokens) < self.max_len:
            token_onehot = self.onehot[last_token]
            token_onehot = token_onehot.view(1, batch_size, self.num_tokens)
            out, state = self.lstm(Variable(token_onehot), state)
            out = self.h1(out)
            out = F.softmax(out)
            token_node = torch.multinomial(out.view(batch_size, self.num_tokens))
            utterance_nodes.append(token_node)
            last_token = token_node.data.view(batch_size)

        type_constr = torch.cuda if h_t.is_cuda else torch
        utterance = type_constr.LongTensor(batch_size, self.max_len).fill_(0)
        for i in range(6):
            utterance[:, i] = utterance_nodes[i].data

        entropy = 0  # placeholder
        return utterance_nodes, utterance, entropy


class ProposalPolicy(nn.Module):
    def __init__(self, embedding_size=100, num_counts=6, num_items=3):
        super().__init__()
        self.num_counts = num_counts
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.fcs = []
        for i in range(num_items):
            fc = nn.Linear(embedding_size, num_counts)
            self.fcs.append(fc)
            self.__setattr__('h1_%s' % i, fc)

    def forward(self, x, eps=1e-8):
        batch_size = x.size()[0]
        nodes = []
        entropy = 0
        type_constr = torch.cuda if x.is_cuda else torch
        proposal = type_constr.LongTensor(batch_size, self.num_items).fill_(0)
        for i in range(self.num_items):
            x1 = self.fcs[i](x)
            x2 = F.softmax(x1)
            node = torch.multinomial(x2)
            nodes.append(node)
            x2 = x2 + eps
            entropy += (- x2 * x2.log()).sum(1).sum()
            proposal[:, i] = node.data

        return nodes, proposal, entropy


class AgentModel(nn.Module):
    def __init__(
            self, enable_comms, enable_proposal,
            term_entropy_reg,
            proposal_entropy_reg,
            embedding_size=100):
        super().__init__()
        self.term_entropy_reg = term_entropy_reg
        self.proposal_entropy_reg = proposal_entropy_reg
        self.embedding_size = embedding_size
        self.enable_comms = enable_comms
        self.enable_proposal = enable_proposal
        self.context_net = NumberSequenceEncoder()
        self.utterance_net = NumberSequenceEncoder()
        self.proposal_net = NumberSequenceEncoder()
        self.proposal_net.embedding = self.context_net.embedding

        self.combined_net = CombinedNet()

        self.term_policy = TermPolicy()
        self.utterance_policy = UtterancePolicy()
        self.proposal_policy = ProposalPolicy()

    def forward(self, context, m_prev, prev_proposal):
        batch_size = context.size()[0]
        c_h = self.context_net(context)
        type_constr = torch.cuda if context.is_cuda else torch
        if self.enable_comms:
            m_h = self.utterance_net(m_prev)
        else:
            m_h = Variable(type_constr.FloatTensor(batch_size, self.embedding_size).fill_(0))
        p_h = self.proposal_net(prev_proposal)

        h_t = torch.cat([c_h, m_h, p_h], -1)
        h_t = self.combined_net(h_t)

        entropy_loss = 0

        term_node, term_a, entropy = self.term_policy(h_t)
        entropy_loss -= entropy * self.term_entropy_reg

        utterance_nodes = []
        utterance = None
        if self.enable_comms:
            utterance_nodes, utterance, utterance_entropy = self.utterance_policy(h_t)
            # entropy_loss -= self.itterance_entropy_reg * utterance_entropy
        else:
            utterance = type_constr.LongTensor(batch_size, 6).zero_()  # hard-coding 6 here is a bit hacky...

        proposal_nodes, proposal, proposal_entropy = self.proposal_policy(h_t)
        entropy_loss -= self.proposal_entropy_reg * proposal_entropy

        return term_node, term_a, utterance_nodes, utterance, proposal_nodes, proposal, entropy_loss