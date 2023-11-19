require 'byebug'
require 'matrix'

checkpoint_path = "stories15M.bin"#ARGV[0
puts "Loading model checkpoint from #{checkpoint_path}..."

checkpoint = File.open(checkpoint_path)

# checkpoint is a binary pre-trained model file
# format:
# - 28 bytes of config
# - token embedding table (vocab_size x dim)
# - rms_att_weight (n_layers * dim)
# - wq (n_layers * dim * (n_heads * head_size))
# - wk (n_layers * dim * (n_kv_heads * head_size))
# - wv (n_layers * dim * (n_kv_heads * head_size))
# - wo (n_layers * dim * (n_heads * head_size))
# - rms_ffn_weight (n_layers * dim)
# - w1 (n_layers * dim * hidden_dim)
# - w2 (n_layers * dim * hidden_dim)
# - w3 (n_layers * dim * hidden_dim)
# - rms_final_weight (dim + 2*(seq_len * head_size/2))
# - wcls (equiv to token_embedding_table for stories15M (shared weights))

Config = Struct.new(:dim,
    :hidden_dim,
    :n_layers,
    :n_heads,
    :n_kv_heads,
    :vocab_size,
    :seq_len)

# config struct size is 28 (7 32 bit unsigned little-endian ints)
values = checkpoint.read(28).unpack("L<*")
config = Config.new(*values)

# Now let's read the Transformer weights out of the model checkpoint

TransformerWeights = Struct.new(
    :token_embedding_table,
    :rms_att_weight,
    :wq,
    :wk,
    :wv,
    :wo,
    :rms_ffn_weight,
    :w1,
    :w2,
    :w3,
    :rms_final_weight,
    :wcls)


head_size = config.dim / config.n_heads

# these are all 32-bit little endian floats. Most of this is just figuring out the format of the model file
# the magic number 4 here is sizeof(float)
# NB unlike llama.c, we're explicitly partitioning these into 2d arrays
token_embedding_table = checkpoint.read(config.vocab_size * config.dim * 4).unpack("e*").each_slice(config.dim).to_a
rms_att_weight = checkpoint.read(config.n_layers * config.dim * 4).unpack("e*").each_slice(config.dim).to_a
wq = checkpoint.read(config.n_layers * config.dim * (config.n_heads * head_size) * 4).unpack("e*").each_slice(config.dim ** 2).to_a.map { |x| x.each_slice(config.dim).to_a }
wk = checkpoint.read(config.n_layers * config.dim * (config.n_kv_heads * head_size) * 4).unpack("e*").each_slice(config.dim ** 2).to_a.map { |x| x.each_slice(config.dim).to_a }
wv = checkpoint.read(config.n_layers * config.dim * (config.n_kv_heads * head_size) * 4).unpack("e*").each_slice(config.dim ** 2).to_a.map { |x| x.each_slice(config.dim).to_a }
wo = checkpoint.read(config.n_layers * config.dim * (config.n_heads * head_size) * 4).unpack("e*").each_slice(config.dim ** 2).to_a.map { |x| x.each_slice(config.dim).to_a }
rms_ffn_weight = checkpoint.read(config.n_layers * config.dim * 4).unpack("e*").each_slice(config.dim).to_a
w1 = checkpoint.read(config.n_layers * config.dim * config.hidden_dim * 4).unpack("e*").each_slice(config.dim * config.hidden_dim).to_a.map { |x| x.each_slice(config.dim).to_a }
w2 = checkpoint.read(config.n_layers * config.dim * config.hidden_dim * 4).unpack("e*").each_slice(config.dim * config.hidden_dim).to_a.map { |x| x.each_slice(config.dim).to_a }
w3 = checkpoint.read(config.n_layers * config.dim * config.hidden_dim * 4).unpack("e*").each_slice(config.dim * config.hidden_dim).to_a.map { |x| x.each_slice(config.dim).to_a }
rms_final_weight = checkpoint.read((config.dim + 2*(config.seq_len * head_size/2)) * 4).unpack("e*")
wcls = token_embedding_table#.flatten.each_slice(config.vocab_size).to_a # re-partition this?? I think?

weights = TransformerWeights.new(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls)

puts "Finished reading model checkpoint. #{File.size(checkpoint_path) - checkpoint.pos} bytes left in file"


# OK, now let's build the tokenizer

Tokenizer = Struct.new(
    :vocab,
    :vocab_scores,
    :sorted_vocab,
    :vocab_size,
    :max_token_length,
    :byte_pieces
)
tokenizer_path = "tokenizer.bin"

# The format here is something like:
#   [4 bytes specifying max token length]
#   Then, for each token:
#     [4 bytes specifying token score] [4 bytes specifying length of token] [token string]

vocab_size = config.vocab_size # tied to the vocab size of the model
tokenizer_bin = File.open(tokenizer_path)
max_token_length = tokenizer_bin.read(4).unpack("L<")[0] # sizeof int

# now we harvest the tokens
vocab_scores = []
vocab = []
vocab_size.times do |i|
    score = tokenizer_bin.read(4) # (sizeof float)
    vocab_scores << score.unpack("e")[0]
    len = tokenizer_bin.read(4).unpack("L<")[0] # read length of segment

    # NB: here llama.c appends "\x0" to the end of every vocab entry, but we don't need to do that
    vocab << tokenizer_bin.read(len).unpack("A#{len}")[0] # read out that size segment
    
end

# llama.c uses sorted_vocab to look up entries in the vocab more quickly. we're skipping this for readability
sorted_vocab = []
byte_pieces = nil # I have no idea why llama.c needs this
tokenizer = Tokenizer.new(vocab, vocab_scores, sorted_vocab, vocab_size, max_token_length, byte_pieces)

# here llama.c builds the sampler, but for now we're just going to sample in the simplest way possible
# Time to generate!

# Let's start by encoding a prompt
# here llama.c generates sorted_vocab. I think it's an optimization for lookups later on

# llama.c uses this method to tokenize:
# We go through prompt by UTF codepoints. For each codepoint:
#  - if it's in vocab, we add its id to tokens
#  - otherwise we add the actual byte to tokens
# Then we go through tokens and try to merge as much as possible (e.g. "I" and "n" into "In")
# We merge if the vocab score of the merged token is higher than the score of the unmerged tokens
# When we can't merge any more, we return the list of tokens


prompt = "Once up"
prompt_tokens = []

prompt.chars.each do |char|
  char = "<0x20>" if char == " " # work around .chars a bit
  prompt_tokens << tokenizer.vocab.index(char)
end

50.times do # llama.c is a bit more clever and ends early when it can't do better - for now we'll just do 50 loops
    prompt_tokens.each.with_index do |a, i|
        next_token = prompt_tokens[i+1]
        next unless next_token

        merged_token = tokenizer.vocab[a] + tokenizer.vocab[next_token]
        merged_token_index = tokenizer.vocab.index(merged_token)
        next unless merged_token_index

        if tokenizer.vocab_scores[merged_token_index] > tokenizer.vocab_scores[a]
            prompt_tokens[i] = merged_token_index
            prompt_tokens.delete_at(i+1)
        end
    end
end

# now we have prompt_tokens, which is a list of token ids
# e.g. for "In the beginning", we get:
# (byebug) prompt_tokens.map { |t| tokenizer.vocab[t] }
# ["In", "<0x20>", "the", "<0x20>", "be", "gi", "nn", "in", "g", "<0x20>"]

prompt_tokens = [1] + prompt_tokens # prepend the ?BOS token? I think?

steps = 10 # this is the `-n` param in llama.c
pos = 0
num_prompt_tokens = prompt_tokens.length
token = prompt_tokens.first
head_size = config.dim / config.n_heads

RunState = Struct.new(
    :x, # activation at current time stamp
    :xb, # same but in residual branch
    :xb2, # convenience buffer
    :hb, # buffer for hiddem dim in ffn
    :hb2, # same
    :q,
    :k,
    :v,
    :att,
    :logits,
    :key_cache,
    :value_cache,

    )
run_state = RunState.new(Array.new(config.dim){ |a| 0 }, 0,nil, nil, nil, nil, nil, nil, nil, nil, 0,0)
run_state.att = Array.new(config.n_heads) { |r| Array.new(config.seq_len) { |c| 0 } }
run_state.key_cache = [Array.new(config.n_heads) { |c| Array.new(head_size) { |d| 0 } }] * 100 # hack - make room for 100 time steps
run_state.value_cache = [Array.new(config.n_heads) { |c| Array.new(head_size) { |d| 0 } }] * 100

kv_dim = (config.dim * config.n_kv_heads) / config.n_heads
kv_mul = config.n_heads / config.n_kv_heads

def rmsnorm(x, weight)
    # Calculate sum of squares
    ss = x.reduce(0.0) { |sum, xi| sum + xi**2 } / x.size.to_f
    ss += 1e-5
    ss = 1.0 / Math.sqrt(ss)
  
    # Normalize and scale
    o = x.map.with_index { |xi, j| weight[j] * (ss * xi) }
  
    o
end
  

def softmax(array)
    max = array.max  # To improve numerical stability
    exps = array.map { |x| Math.exp(x - max) }
    sum_of_exps = exps.sum
    exps.map { |exp| exp / sum_of_exps }
end

# I fully gave up on making this idiomatic ruby
def matmul(x, w)
    # Determine the dimensions
    d = w.size
  
    # Initialize the output array
    xout = Array.new(d, 0.0)
  
    # Perform the matrix multiplication
    (0...(d-1)).each do |i|
      val = 0.0
      (0...(x.size)).each do |j|
        val += w[i][j].to_f * x[j].to_f
      end

      xout[i] = val

    end
  
    xout
end

while (pos < steps) do
    # now we do the actual generation
    # start with a forward pass

    # copy the token embedding into x
    run_state.x = weights.token_embedding_table[token]


    # config.n_layers.times do |i| # forward all the layers
    2.times do |i| # forward all the layers

        run_state.xb = rmsnorm(run_state.x, weights.rms_att_weight[i]) # attention rmsnorm
        puts "layer #{i}, xb0 #{run_state.xb[0]}, x #{run_state.x[0]}"

        # qkv matmuls
        run_state.q = matmul(run_state.xb,  weights.wq[i])
        run_state.k = matmul(run_state.xb,  weights.wk[i])
        run_state.v = matmul(run_state.xb,  weights.wv[i])

        # RoPE relative positional encoding: complex-valued rotate q and k in each head
        (config.dim / 2).times do |j|
            counter = j * 2
            head_dim = counter % head_size
            freq = 1.0 / (10000.0 ** (head_dim / head_size.to_f))
            val = pos * freq
            fcr = Math.cos(val)
            fci = Math.sin(val)
            rotn = counter < kv_dim ? 2 : 1; # how many vectors? 2 = q & k, 1 = q only
            rotn.times do |r|
                vec = r == 0 ? run_state.q : run_state.k # the vec to rotate - query or key
                v0 = vec[counter]
                v1 = vec[counter+1]

                vec[counter] = v0 * fcr - v1 * fci
                vec[counter+1] = v0 * fci + v1 * fcr
                if r == 0 
                    run_state.q = vec
                else
                    run_state.k = vec
                end
            end
        end

        # NB llama.c does this way above (though some ports like llama2.py do it here)
        # We skip the llama.c kv cache layer offset since we're using actual arrays
        # llama.c also sets .k and .v here. we reverse the order, since the point is to
        # make sure key/val caches are populated
        run_state.key_cache[pos] = run_state.k
        run_state.value_cache[pos] = run_state.v

        # multihead attention. iterate over all heads
        config.n_heads.times do |head_index|
            q = run_state.q[(head_index * head_size)..((head_index * head_size) + head_size - 1)] # get the query vec for this head
            # att = run_state.att[head_index] # get the attention scores

            # Iterate over all timesteps, including the current one
            # TODO: split key_cache etc by head size so we aren't doing this awkward
            # 2d in 1d C-style trick. Will need changes to the RoPE above
            (pos + 1).times do |t|
                k = run_state.key_cache[t][(head_index * head_size)..((head_index * head_size) + head_size - 1)] # get the key vec for this head and this timestamp
                # calculate the attention score as the dot product of q and k
                # score = q.zip(k).map { |x,y| x * y }.sum
                score = 0.0
                q.length.times do |c|
                    score += q[c] * k[c]
                end
                score /= Math.sqrt(head_size)
                run_state.att[head_index][t] = score # save score to attention buffer
            end
            # we only softmax the number of elements we've worked on so far (e.g. pos)
            run_state.att[head_index][0..pos] = softmax(run_state.att[head_index][0..pos])

            # weighted sum of the values, store back into xb
            run_state.xb[(head_index * head_size)..((head_index * head_size) + head_size - 1)] = Array.new(48) { 0 } # zero out the xb slice for this head
            (pos + 1).times do |t|
                v = run_state.value_cache[t][(head_index * head_size)..((head_index * head_size) + head_size - 1)] # value vec for this head/timestamp
                attention = run_state.att[head_index][t] # get the attention weight at this timestamp

                # accumulate the weighted value into xb
                head_size.times do |j|
                    run_state.xb[(head_index * head_size) + j] += attention * v[j]
                end

            end
        end
        # final matmul to get the output of the attention
        run_state.xb2 = matmul(run_state.xb, weights.wo[i])

        # residual connection back into x
        config.dim.times do |j|
            run_state.x[j] += run_state.xb2[j]
        end

        run_state.xb = rmsnorm(run_state.x, weights.rms_ffn_weight[i]) # rmsnorm for ffn

        run_state.hb = matmul(run_state.xb, weights.w1[i])
        run_state.hb2 = matmul(run_state.xb, weights.w3[i])

        # SwiGLU non-linearity
        config.hidden_dim.times do |j|
            val = run_state.hb[j]
            val *= (1.0 / (1.0 + Math.exp(-val))) # silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= run_state.hb2[j]
            run_state.hb[j] = val
        end

        # final matmul to get ffn output
        run_state.xb = matmul(run_state.hb, weights.w2[i]) # not sure why I have to transpose here...

        # residual connection again?
        config.dim.times do |j|
            run_state.x[j] += run_state.xb2[j]
        end

    end

    # final rmsnorm
    run_state.x = rmsnorm(run_state.x, weights.rms_final_weight)
    # classifier into logits, by far the slowest op
    run_state.logits = matmul(run_state.x, weights.wcls)

    if pos < num_prompt_tokens - 1
        puts "Still in prompt"

        token = prompt_tokens[pos+1]
    else
        puts "Options: " + run_state.logits.max(3).map { |l| tokenizer.vocab[run_state.logits.index(l)] }.join(", ")
    
        # Ultra simple sample: just take one of the top 3
        token = run_state.logits.index(run_state.logits.max(3).sample)
    end
   


    piece = tokenizer.vocab[token]

    puts "Piece: #{piece}"

    pos += 1
end





