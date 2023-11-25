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