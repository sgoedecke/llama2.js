- tokenizing and decoding is not so bad
- parsing the .bin llama2.c format is not so bad

- by far the hardest part of this is porting the work from C-style "everything is a *float" into actual 1-D and 2-D arrays
    - all of the ports punt on this and just use a 1-D array of floats (edit: not the cpp port)

- ugh, ok, I think I've cracked that but my values are drifting like crazy. it's hard to figure out what's just floating point precision differences and what isn't. and maybe the FP stuff is enough to sink this idea by itself. 

Once more, this time in node


oh god damn it, the CPP port is the only one to actually factor out the tensors: https://github.com/leloykun/llama2.cpp/blob/master/run.cpp

OK, ported. Went through line by line and fixed a typo (xb instead of xb2). Now I notice it's not tokenizing properly.

OK, tokenizing fine. After some fiddling to make sure I prefix the prompt with an empty character, etc, it now generates the same as llama.c2 for the first four or so new tokens. Then it doesn't. Could be something to do with the different RoPE implementation? It feels like it might be attention-related, since it only appears past five tokens or so

Note: I'm still doing temp 0 both on run.c and my impl, so comparison is easier.

generated tokens:
(`make run && ./run stories15M.bin -n 10 -t 0 -i "In the park, there was"`)
```
token 1
token 512
token 278
token 14089
token 29892
token 727
token 471
token 263
token 2217 <-- here's where they begin to differ, at pos=8
token 7826
```

Yep, breaking at pos 7 reveals that runstate.att is all uniform in my code, but variable in run.c.

OK! Two problems: I was setting att[pos] instead of att[t], and more importantly I had runState.att too low: it was only a single head's worth of attention. TODO: make it an actual 2d array so we don't need to mess around with subarrays, and/or subarray the qkv stuff instead of opsing it.

It works! TODOS: proper topp sampling, dynamic temperature, visualization of attention

Attention viz was trivial, nice!

---

Forward pass breakdown

For each layer, (Attention mechanism -> feed-forward network (FFN)), then rmsnorm and turn state.x into logits for the next token

---

Initialize state.x with the default weights for the most recent token
Think of state.x as a _token embedding_: it starts a the embedding vec of the last token, and then a series of transformations are applied to it to produce the next token (or its logits, anyway)

For each layer:
    Attention mechanism:
        Populate state.xb with RMSnorm of state.x
        Initialize state.q/k/v by doing state.xb * weights.wq[layer]
        RoPE: for each head, mess with state.q/k by adding the positional encoding
        Populate state.keyCache and state.valueCache for the current layer/position from state.k and state.v
        For each head:
            Go through each previous token and populate each head's segment of state.att
            The value should be state.q (for the current token) * state.keyCache (for each previous token)
            The idea: given the attention tensor for the current token, how much does each head now care about each previous token
            Softmax att
            Nil out state.xb and set each value to the sum of attentions over all positions for each previous (whatever headSize corresponds to) 
        Populate xb2 with xb * weights.wo[layer]
        Sum x with state.xb (residual connection)

    FFN:
        Populate xb with RMSnorm of state.x
        Linear layer 1 (w1) with state.x into state.hb
        Linear layer 2 (w2) with state.x into state.hb2
        SwiGLU non-linearity on state.hb
        Elementwise multiply hb by hb2
        Activation function on hb into xb
        Finally sum xb with state.x (residual connection)

Do one last rmsnorm on x
Popuulate logits with state.x * weights.wcls


---

The token embedding table is 32000 rows (vocabSize) and 288 columns (dim). The entire transformer process populates state.x with a 288-element array that represents the activations at the current time stamp. That gets multiplied into the token embedding table to produce at 32000-element array of logits.

So we basically start with the most context-free state of the world (what was the last token) and inject it with context from state.x and state.keyCache/state.valueCache, which represent previous positions' values of state.k and state.v. **Nothing else is retained between forward passes.**
    Is this what's typically meant by transformers being able to operate on tokens in parallel? If you get a long prompt, you can calculate the key/value caches all independently before actually proceeding to the FFN step at all? (edit: no.)
        GPT4 seems to agree, though I had to clarify what I meant a bunch, so I'm suspicious
        I tried confirming from llama.cpp but god damn it's not easy to read
        Trying it out: NO, this is NOT right. We need `state.x` after the FFN step to compute attention values for each successive layer after the first, even when we're still in the prompt
    We CAN skip the final logits classifier when we're still in the prompt, of course


---

What's RMSnorm for? RMSnorm normalizes (i.e. shrinks big values and increases small ones) a vector without changing the mean value. Often you also pass a `weight` vector to it, which scales each element of the vector differently after normalization. I guess the benefit is to give you the benefits of rmsnorming (stability, consistency across layers) while also having a knob to turn when you want to emphasize some parts of the vector much more than others.

What's softmax for? Softmax takes an array of values and turns them into a probability distribution (i.e. makes all values between 0 and 1, and makes it so summing over the array is equal to 1). This happens in an exponential way: small values get smaller, large values get larger. In our model we only use softmax to turn our attention scores into a distribution over previous tokens.

---

You don't have to use exactly the same model for training and inference, although it does have to be very similar. Typically the differences are performance-related: you train in PyTorch in a highly parallelizable way, but you deploy your inference model in C in a way that's optimized for inference speed only. Alternatively, you train using 32-bit floats, but you perform inference using 4-bit ints to save memory and speed up arithmetic.

---

OK, that's enough learning. Back to experimenting. I'm going to see what the results are if I juice the attention manually.

Yep, it does work: you seem to get a more raven-focused story if you emphasise 'raven' in the attention matrix (or 'nest', or whatever)

It works really well with the attention viz stuff - you can see the bits you juice staying juiced.

---

## RoPE

It's worth noting that the freqCisReal/Imag parts of RoPE aren't learned weights, but are instead pre-computed values of some specific (likely sinusoidal?) function that's part of the RoPE algorithm. That's why some implementations (including llama2.c itself) can skip those parts of the weights and compute it on the fly.

The 10,000ft view of RoPE is it's one form of "positional encoding": a way to include information on the relative position of tokens. Older versions of this do it in the tokenEmbeddingsTable (e.g. by embedding 512 different versions of the token "the" depending on its position). That means they need an embedding table that's 512x the size.

RoPE instead applies a rotation into the complex plane. Why then doesn't it need a 2x size token embedding table to represent the real and complex parts? Because it only cares about the size of the _rotation_, not specifically that it's a rotation on the original embeddings vector. Thus it can treat an embedding of 10 real elements as a complex embedding with five pairs, each of which represents an imaginary number. 

The consequence of this is that the dot product of state.q and state.k now also encodes positional information. This gets learned by the model during backpropagation: primarily as part of the attention weights, but also in the FFN weights (since the FFN is also conditioned on the attention weights).

General summary: the model is trained to _interpret_ positional metadata out of the complex rotation of the query/key vectors (or rather a complex version of them by taking real/imag pairs). That means that when we do inference with the model, we need to perform that same complex rotation on the query/key vectors so the model has that positional metadata to work with.

## QKV

The point of all of these is to produce a single vector of attention scores for each previous token, which represents how relevant each of these tokens is when generating the next token. For instance, if the previous tokens are [456, 1, 66], the attention vector might look like [0.01, 0.9, 0.09], meaning that the model thinks the most relevant token is 1, followed at a distance by 66.

All of these vectors are initialized by running state.x through a layer of the q/k/v weights: so they begin with the model's best guess based on the current token.

The query vector represents the current token for which attention scores are calculated.

Key vectors represent all previous tokens - in practice, this set of vectors is pulled out of state.keyCache (state.k is the key vector for the current token only, and gets saved to keyCache as we go).

Attention scores are then calculated between the query vector and each key vector (typically by taking the dot product and then normalizing with softmax). Positional encoding on the query and key vectors means that attention scores are position-aware.

Value vectors also correspond to all previous tokens (and are also pulled out of state.valueCache). The "value" here represents the actual value of the token (unlike the key vector, which represents the _attention_ that should be paid to each token). That's why the QKV section finishes by weighting the value vector for each previous token by its new attention score and summing it back into state.x. 

## Heads

OK, so what's a head? A head is basically one self-contained version of this positionally-encoded QKV attention mechanism. If a model has eight heads, that's eight separate sets of weights all with the same attention structure (i.e. its own q/k/v weights). They're combined at the end of the attention section by running them through a linear layer with the weights.wout weights.

The idea is that each head can learn to pay attention to different parts/aspects of the input data, thus encoding many very different ways parts can be related to each other (e.g. syntactic, semantic, etc).

Head processing can also be done in parallel, which is a big reason why the transformer architecture performs well. (It's worth knowing that _layers_ can't be done in parallel, because subsequent layers operate on the state.x output of the previous layer.)

## SwiGLU

What's SwiGLU do? First up, it's like RoPE: a mathematical operation that is present during training so the model incorporates it into its weights. So we have to do it during inference, or the weights won't do what they're supposed to do.

Why include it? It introduces an important _non-linear_ behaviour: instead of just applying a weight to each element the vector, it can approximate a boolean multiplication by zero. That's because when you take the sigmoid (`(1.0 / (1.0 + Math.exp(x))`) of large positive numbers you very rapidly approach 0 (while negative numbers rapidly approach 1). So the model can learn to completely zero out certain sections of `state.x` by making them very large when multiplied by the first layer of weights, or to leave them alone by making them very small.

## Residuals

The "typical" neural network is an input, a bunch of matrix multiplications across various layers, and then an output. At each stage, the input is transformed by the weights of each layer. However, in this naive approach, it can be easy for early layers to get forgotten, since each subsequent layer can dramatically reduce the valence of a previous layer's output.

"Residuals" are a way to avoid this. The idea is to add (not multiply) the output of each layer directly to `state.x`, to effectively create "direct" connections that bypass the weights of future layers. This complicates the architecture of the neural network, allowing the model to encode more sophisticated relationships. It also means that all layers can be important, instead of later layers dominating earlier ones, so we can end up stacking more layers into a model without losing efficacy.

---

OK so, stepping back further - what _is_ a NN? It's a deliberately complex apparatus with a lot of different knobs and levers, which during the training process get set to some configuration that seems to work pretty well. The architecture of a NN is often choices that make the machine more complex, or widen the number of things each knob can do, in order to increase the chance of landing on a useful configuration by trial and error. But you can't just stack complexity forever, or it'll be impossible to find any useful configuration. So you have to make tradeoffs.

---

Can you do confidence estimates by estimating difference between logits? Yes, trivially. Even a naive "difference between the first and third logit" works well, and gives the results you'd expect for prompts that are in and out of domain.

I really should implement actual sampling

---

We can divide LLM development into a few different domains:

- Innovating model architecture: e.g. adding RoPE, SwiGLU, changing head sizes/counts
- Training new models: acquiring datasets, trying lots of things, etc
- Inferencing: making changes during the forward pass step, such as my `-e emphasis` work or exposing the attention weights
- Post-processing: making changes after the logits are produced, such as my confidence estimates work or grammar-aware logit sampling (which I haven't yet done)
- Prompt engineering, agents, and chaining: engineering a system around a text-in/text-out LLM API

This is roughly in order of difficulty/effort. You can do the last one (and maybe _some_ post-processing) against the OpenAI API. You can do inferencing and post-processing by messing around with a simple local model, like me. But you can't innovate architecture or train without a lot of compute and time.