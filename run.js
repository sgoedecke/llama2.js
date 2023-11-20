const fs = require('fs');

const FLOAT_SIZE = 4
const checkpoint_path = "stories15M.bin"
const tokenizer_path = "tokenizer.bin"

console.log("Loading model checkpoint from", checkpoint_path)

const checkpoint = fs.readFileSync(checkpoint_path)
const configKeys  = ['dim',
'hiddenDim',
'nLayers',
'nHeads',
'nKVHeads',
'vocabSize',
'seqLen']
const config = {}
configKeys.forEach((key, i) => {
    config[key] = checkpoint.readUInt32LE(i * FLOAT_SIZE) // size of float
})

console.log(config)

offset = configKeys.length * FLOAT_SIZE

const weights = {}

const headSize = config.dim / config.nHeads;

weights.tokenEmbeddingTable = new Float32Array(checkpoint.buffer, offset, config.vocabSize * config.dim);
offset += weights.tokenEmbeddingTable.byteLength;

weights.rmsAttWeight = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim);
offset += weights.rmsAttWeight.byteLength;

weights.wq = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.nHeads * headSize);
offset += weights.wq.byteLength;

weights.wk = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.nKVHeads * headSize);
offset += weights.wk.byteLength;

weights.wv = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.nKVHeads * headSize);
offset += weights.wv.byteLength;

weights.wo = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.nHeads * headSize * config.dim);
offset += weights.wo.byteLength;

weights.rmsFFNWeight = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim);
offset += weights.rmsFFNWeight.byteLength;

weights.w1 = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.hiddenDim);
offset += weights.w1.byteLength;

weights.w2 = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.hiddenDim * config.dim );
offset += weights.w2.byteLength;

weights.w3 = new Float32Array(checkpoint.buffer, offset, config.nLayers * config.dim * config.hiddenDim);
offset += weights.w3.byteLength;

weights.rmsFinalWeight = new Float32Array(checkpoint.buffer, offset, config.dim);
offset += weights.rmsFinalWeight.byteLength;

offset += config.seqLen * headSize / 2;
offset += config.seqLen * headSize / 2;

weights.wcls = weights.tokenEmbeddingTable; // no shared weights

// We've read out all the content from the checkpoint, but we're not in C anymore and we don't have to pretend that our
// 2D arrays are represented by a 1D array. Let's fix that.

function splitFloat32Array(array, chunkSize) {
    const result = [];
    for (let i = 0; i < array.length; i += chunkSize) {
        const end = Math.min(i + chunkSize, array.length);
        result.push(array.subarray(i, end));
    }
    return result;
}

weights.tokenEmbeddingTable = splitFloat32Array(weights.tokenEmbeddingTable, config.dim);
weights.rmsAttWeight = splitFloat32Array(weights.rmsAttWeight, config.dim);
weights.wq = splitFloat32Array(weights.wq, config.dim ** 2).map((array) => splitFloat32Array(array, config.dim));
weights.wk = splitFloat32Array(weights.wk, config.dim ** 2).map((array) => splitFloat32Array(array, config.dim));
weights.wv = splitFloat32Array(weights.wv, config.dim ** 2).map((array) => splitFloat32Array(array, config.dim));
weights.wo = splitFloat32Array(weights.wo, config.dim ** 2).map((array) => splitFloat32Array(array, config.dim));
weights.rmsFFNWeight = splitFloat32Array(weights.rmsFFNWeight, config.dim);
weights.w1 = splitFloat32Array(weights.w1, config.dim * config.hiddenDim).map((array) => splitFloat32Array(array, config.dim));
// w2 has to be backwards? or something? skip for now, we'll just use lengths
// weights.w2 = splitFloat32Array(weights.w2, config.hiddenDim * config.dim).map((array) => splitFloat32Array(array, config.dim));
weights.w3 = splitFloat32Array(weights.w3, config.dim * config.hiddenDim).map((array) => splitFloat32Array(array, config.dim));
weights.wcls = weights.tokenEmbeddingTable

console.log("Constructed weights:", weights)

// OK, now let's build the tokenizer
console.log("Loading model checkpoint from", tokenizer_path)
const tokenizerBin = fs.readFileSync(tokenizer_path)

const tokenizer = {}
tokenizer.maxTokenLength = tokenizerBin.readUInt32LE(0)
offset = FLOAT_SIZE

tokenizer.vocabScores = []
tokenizer.vocab = []
for (let i = 0; i < config.vocabSize; i++) {
    score = tokenizerBin.readFloatLE(offset)
    offset += FLOAT_SIZE
    tokenizer.vocabScores.push(score)
    
    len = tokenizerBin.readUInt32LE(offset)
    offset += FLOAT_SIZE
    
    let str = "";
    for (let j = 0; j < len; j++) {
        str += String.fromCharCode(tokenizerBin[offset++]);
    }
    tokenizer.vocab[i] = str;
}

// we're skipping byte pieces and sortedVocab as unnecessary for now
// time to generate! let's start by encoding a prompt

const prompt = "Zap"
const promptTokens = []

prompt.split("").forEach((char) => {
    if (char == " ") {
        char = "<0x20>" 
    }
    promptTokens.push(tokenizer.vocab.indexOf(char))
})

const tries = 50
for (let i = 0; i < tries; i++) {
    promptTokens.forEach((token, index) => {
        const nextToken = promptTokens[index+1]
        if (nextToken == undefined) {
            return
        }
        const mergedToken = tokenizer.vocab[token] + tokenizer.vocab[nextToken]
        const mergedTokenIndex = tokenizer.vocab.indexOf(mergedToken)
        
        if (mergedTokenIndex == undefined) {
            return
        }
        
        if (tokenizer.vocabScores[mergedTokenIndex] > tokenizer.vocabScores[token]) {
            promptTokens.splice(index, 2, mergedTokenIndex)
        }
    })
}

// Now we have our prompt tokenized into a list of token ids
// Let's get started

const steps = 50 // the -n param in llama2.c
let pos = 0
const numPromptTokens = promptTokens.length
let token  = promptTokens[0]

const runState = {
    // current wave of activations
    x: new Float32Array(config.dim), // activation at current time stamp (dim,)
    xb: new Float32Array(config.dim), // same, but inside a residual branch (dim,)
    xb2: new Float32Array(config.dim), // an additional buffer just for convenience (dim,)
    hb: new Float32Array(config.hiddenDim), // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: new Float32Array(config.hiddenDim), // buffer for hidden dimension in the ffn (hidden_dim,)
    q: new Float32Array(config.dim), // query (dim,)
    k: new Float32Array(config.dim), // key (dim,)
    v: new Float32Array(config.dim), // value (dim,)
    att: new Float32Array(config.nHeads * config.seqLen), // buffer for scores/attention values (n_heads, seq_len)
    logits: new Float32Array(config.vocabSize), // output logits
    // kv cache
    keyCache: new Float32Array(config.nLayers * config.seqLen * config.dim),   // (layer, sq_len, dim)
    valueCache: new Float32Array(config.nLayers * config.seqLen * config.dim), // (layer, seq_len, dim)
};

const kvDim = (config.dim * config.nKVHeads) / config.nHeads
const kvMul = config.nHeads / config.nKVHeads

function matmul(xout, x, w, n=null, d=null) {
    if (typeof w[0] == 'object') { // if we have an arr of arrs
        // w = new Float32Array(w.reduce((acc, val) => [...acc, ...Array.from(val)], []));
    }

    n = n || x.length
    d = d || w.length

    
    for (let i = 0; i < d; i++) {
        let val = 0;
        for (let j = 0; j < n; j++) {
            val += w[i][j] * x[j];
        }
        xout[i] = val; 
    }
    if (x.length > n || w.length > (d*n)) {
        debugger
        throw "dimensions do not match"
    }
}

function rmsnorm(o, x, weight) {
    const size = x.length
    // calculate sum of squares  
    let ss = 0.0;
    for (let j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5; 
    ss = 1.0 / Math.sqrt(ss);
    
    // normalize and scale
    for (let j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

function softmax(x, size) {
    
    // const size = x.length
    
    // find max value 
    let maxVal = x[0];
    for (let i = 1; i < size; i++) {
        if (x[i] > maxVal) {
            maxVal = x[i]; 
        }
    }
    
    // exp and sum
    let sum = 0;
    for (let i = 0; i < size; i++) {
        x[i] = Math.exp(x[i] - maxVal);
        sum += x[i];
    }
    
    // normalize 
    for (let i = 0; i < size; i++) {
        x[i] /= sum;
    }
    return x
    
}

while (pos < steps) {
    // forward pass
    // copy the token embedding into x
    runState.x = weights.tokenEmbeddingTable[token]
    
    // forward each layer
    for (let l = 0; l < config.nLayers; l++) {

        if (runState.x == undefined) {
            debugger
        }

        rmsnorm(runState.xb, runState.x, weights.rmsAttWeight[l]) // attention rmsnorm

        // qkv matmuls
        matmul(runState.q, runState.xb, weights.wq[l], config.dim, config.dim)
        matmul(runState.k, runState.xb, weights.wk[l], config.dim, kvDim)
        matmul(runState.v, runState.xb, weights.wv[l], config.dim, kvDim)

        // RoPE relative pos encoding: complex-valued rotate q and k in each head
        for (let i = 0; i < config.dim; i += 2) {
            const headDim = i % headSize;
            const freq = 1 / Math.pow(10000, headDim / headSize);    
            const val = pos * freq;
            const fcr = Math.cos(val);
            const fci = Math.sin(val);
            const rotn = i < kvDim ? 2 : 1; 
            for (let v = 0; v < rotn; v++) {
                let vec = v === 0 ? runState.q : runState.k; // this NaNs them
                const v0 = vec[i];
                const v1 = vec[i+1];

                vec[i] = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }

        }

        // populate key/value caches. llama.c does this way above here
        // TODO: skip the cache layer offset and use actual arrays
        const loff = l * config.seqLen * kvDim;

        const keyCacheRow = runState.keyCache.subarray(loff + pos * kvDim, loff + (pos + 1) * kvDim);
        const valueCacheRow = runState.valueCache.subarray(loff + pos * kvDim, loff + (pos + 1) * kvDim);
        keyCacheRow.set(runState.k);
        valueCacheRow.set(runState.v);

        // multi-head attention, iterate over all heads. could parallelize this in theory.
        for (let h = 0; h < config.nHeads; h++) {
            // get q vector for this head  
            const q = runState.q.subarray(h * headSize, (h + 1) * headSize);
            // att scores for this head
            const att = runState.att.subarray(h * config.seqLen, (h + 1) * config.seqLen);
            
            for (let t = 0; t <= pos; t++) {
                // get k vector
                const k = runState.keyCache.subarray(t * kvDim + Math.floor(h / kvMul) * headSize, t * kvDim + Math.floor(h / kvMul) * headSize + headSize);
                
                // att score
                let score = 0;
                for (let i = 0; i < headSize; i++) {
                    score += q[i] * k[i];
                }
                score /= Math.sqrt(headSize);
                // save to att buffer  
                att[t] = score; 
            }
            // softmax att weights  
            softmax(att, pos + 1);

            // weighted sum of v into xb
            const xb = runState.xb.subarray(h * headSize, (h + 1) * headSize);
            xb.fill(0);
            for (let t = 0; t <= pos; t++) {
                const v = runState.valueCache.subarray(t * kvDim + Math.floor(h / kvMul) * headSize, t * kvDim + Math.floor(h / kvMul) * headSize + headSize);
                const a = att[t];
                for (let i = 0; i < headSize; i++) {
                    xb[i] += a * v[i];
                }
            }

        }

        // final matmul to get the output of the attention
        matmul(runState.xb2, runState.xb, weights.wo[l], config.dim, config.dim)

        // residual connection 
        for (let i = 0; i < config.dim; i++) {
            runState.x[i] += runState.xb2[i]; 
        }

        rmsnorm(runState.xb, runState.x, weights.rmsFFNWeight[l]) // ffn rmsnorm
        
        matmul(runState.hb, runState.xb, weights.w1[l], config.dim, config.hiddenDim)
        matmul(runState.hb2, runState.xb, weights.w3[l], config.dim, config.hiddenDim)
        
        // SwiGLU non-linearity
        for (let i = 0; i < config.hiddenDim; i++) {
            let val = runState.hb[i];
            val *= 1 / (1 + Math.exp(-val)); 
            val *= runState.hb2[i];
            runState.hb[i] = val;
        }

        // final matmul for ffn output
        // this NaNs xb without explicitly passing the lengths. TODO: figure out how to do this with arrays
        matmul(runState.xb, runState.hb, weights.w2.subarray(l * config.hiddenDim * config.dim, (l + 1) * config.hiddenDim * config.dim), config.hiddenDim, config.dim); // dimensions don't match

        // residual connection
        for (let i = 0; i < config.dim; i++) {
            runState.x[i] += runState.xb[i];
        }
        
    }
    
    // final rmsnorm
    rmsnorm(runState.x, runState.x, weights.rmsFinalWeight);

    // classifier into logits
    matmul(runState.logits, runState.x, weights.wcls, config.dim, config.vocabSize);

    if (pos < numPromptTokens) {
        // still in prompt
        token = promptTokens[pos]
    } else {
        token = runState.logits.indexOf(Math.max(...runState.logits))
    }

    let piece = tokenizer.vocab[token]
    if (piece == undefined) {
        debugger
        piece = "<oops>"
        token = 1
    }
    process.stdout.write(piece)
    
    pos += 1
}

