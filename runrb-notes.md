- tokenizing and decoding is not so bad
- parsing the .bin llama2.c format is not so bad

- by far the hardest part of this is porting the work from C-style "everything is a *float" into actual 1-D and 2-D arrays
    - all of the ports punt on this and just use a 1-D array of floats

- ugh, ok, I think I've cracked that but my values are drifting like crazy. it's hard to figure out what's just floating point precision differences and what isn't. and maybe the FP stuff is enough to sink this idea by itself. 

Once more in node


oh god damn it, the CPP port is the only one to actually factor out the tensors: https://github.com/leloykun/llama2.cpp/blob/master/run.cpp

