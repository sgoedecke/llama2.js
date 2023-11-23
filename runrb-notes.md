- tokenizing and decoding is not so bad
- parsing the .bin llama2.c format is not so bad

- by far the hardest part of this is porting the work from C-style "everything is a *float" into actual 1-D and 2-D arrays
    - all of the ports punt on this and just use a 1-D array of floats

- ugh, ok, I think I've cracked that but my values are drifting like crazy. it's hard to figure out what's just floating point precision differences and what isn't. and maybe the FP stuff is enough to sink this idea by itself. 

Once more in node


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

OK! Two problems: I was setting [pos] instead of [t], and more importantly I had runState.att too low: it was only a single head's worth of attention. TODO: make it an actual 2d array so we don't need to mess around with subarrays, and/or subarray the qkv stuff instead of opsing it.

It works! TODOS: proper topp sampling, dynamic temperature, visualization of attention