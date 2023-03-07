import numpy as np

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    x: 
    """
    # minus the max value to avoid overflow, avoid numerical instability due to large values in x.
    exp_x = np.exp(x-np.max(x))
    # axis=1 means that we sum over the rows
    # keepdims=True means that we keep the dimensions of the array
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class MultiheadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads
        np.random.seed(0)
        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.Wo = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        # make x into the shape of (batch_size, seq_len, n_heads, depth)
        x = np.reshape(x, (x.shape[0], -1, self.n_heads, self.depth))
        # transpose the axes to make it (batch_size, n_heads, seq_len, depth)
        return np.transpose(x, (0, 2, 1, 3))
    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_model)
        Q = np.dot(query, self.Wq)
        K = np.dot(key, self.Wk)
        V = np.dot(value, self.Wv)
        # split the heads
        # Q, K, V: (batch_size, n_heads, seq_len, depth)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
    
        score = np.matmul(Q, np.transpose(K, (0, 1, 3, 2)))/np.sqrt(self.depth)
        if mask is not None:
            # mask is a boolean tensor with dimension (batch_size, seq_len)
            score = np.where(mask, score, np.full_like(score, -np.inf))
        attention_weights = softmax(score)
        # attention_weights: (batch_size, n_heads, seq_len_q, seq_len_k)
        # V: (batch_size, n_heads, seq_len_v, depth)
        context = np.matmul(attention_weights, V)
        context = np.transpose(context, (0, 2, 1, 3))
        context = np.reshape(context, (context.shape[0], -1, self.d_model))
        out = np.dot(context, self.Wo)

        return out, attention_weights
class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pos_encoding = self.get_pos_encoding()

    def get_angles(self, pos, i):
        # this will get us the angles for any index labeled as i, both positive and negative
        angle_rate = 1/np.power(10000, (2*(i//2))/np.float32(self.d_model))
        return pos*angle_rate
    def get_pos_encoding(self):
        # pos_encoding = np.zeros((self.max_len, self.d_model))
        # for pos in range(self.max_len):
        #     for i in range(0, self.d_model, 2):
        #         angles = self.get_angles(pos, i)
        #         pos_encoding[pos, i] = np.sin(angles)
        #         pos_encoding[pos, i+1] = np.cos(angles)
        angle_rads = self.get_angles(np.arrange(self.max_len)[:, np.newaxis], np.arrange(self.d_model)[np.newaxis, :])
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return pos_encoding
    def forward(self, x):
        x = x + self.pos_encoding[:, :x.shape[1], :]
        return x
    
# no bias
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
    def forward(self, x):
        x = np.dot(x, self.W1)
        #relu, everything negative is set to 0
        x = np.maximum(0, x)
        output = np.dot(x, self.W2)
        return output
    
class EncoderLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

        # add & norm
        # layer normalization
        self.norm1 = np.random.randn((1, d_model))
        self.norm2 = np.random.randn((1, d_model))

    def forward(self, x, mask):
        
        ## multiheadAttention
        x_hat, _ = self.self_attention.forward(x, x, x, mask)
        ## layer normalization 1
        x = x + x_hat
        x = x / np.sqrt(self.d_model)
        x = x*self.norm1
        
        ## feed forward
        x_hat = self.ff.forward(x)
        ## layer normalization 2
        x = x + x_hat
        x = x*self.norm2
        return x

class DecodeLayer:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff

        self.self_attention = MultiheadAttention(d_model, n_heads)
        self.encorder_decoder_attention = MultiheadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = np.random.randn((1, d_model))
        self.norm2 = np.random.randn((1, d_model))
        self.norm3 = np.random.randn((1, d_model))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        src_mask: encoder mask
        tgt_mask: label mask
        """
        # Masked multihead attention
        x_hat, _ = self.self_attention.forward(x, x, x, tgt_mask)
        # layer normalization 1
        x = x + x_hat
        x = x / np.sqrt(self.d_model)
        x = x*self.norm1
        
        # encoder-decoder attention
        # K, V: encoder output
        # Q: decoder output
        x_hat, attention_weight = self.encorder_decoder_attention.forward(x, encoder_output, encoder_output, src_mask)
        # layer normalization 2
        x = x + x_hat
        x = x  / np.sqrt(self.d_model)
        x = x * self.norm2
        
        # feed forward
        x_hat = self.ff.forward(x)
        # layer normalization 3
        x = x + x_hat
        x = x * self.norm3

        return x, attention_weight


        
