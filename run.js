const fs = require('fs');

const FLOAT_SIZE = 4
const checkpoint_path = "stories42M.bin" // 42, 15
const tokenizer_path = "tokenizer.bin"

const colours = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    dim: "\x1b[2m",
    underscore: "\x1b[4m",
    blink: "\x1b[5m",
    reverse: "\x1b[7m",
    hidden: "\x1b[8m",
    
    fg: {
        black: "\x1b[30m",
        red: "\x1b[31m",
        green: "\x1b[32m",
        yellow: "\x1b[33m",
        blue: "\x1b[34m",
        magenta: "\x1b[35m",
        cyan: "\x1b[36m",
        white: "\x1b[37m",
        gray: "\x1b[90m",
        crimson: "\x1b[38m" // Scarlet
    },
    bg: {
        black: "\x1b[40m",
        red: "\x1b[41m",
        green: "\x1b[42m",
        yellow: "\x1b[43m",
        blue: "\x1b[44m",
        magenta: "\x1b[45m",
        cyan: "\x1b[46m",
        white: "\x1b[47m",
        gray: "\x1b[100m",
        crimson: "\x1b[48m"
    }
};

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

// this is temporary - seeing if the cpp style RoPE works
weights.freqCisReal = new Float32Array(checkpoint.buffer, offset, config.seqLen * headSize / 2);
offset += weights.freqCisReal.byteLength;
weights.freqCisImag = new Float32Array(checkpoint.buffer, offset, config.seqLen * headSize / 2);
offset += weights.freqCisImag.byteLength;

// offset += config.seqLen * headSize / 2; // run.cpp reads in freq_cis_real from here but run.c now skips it
// offset += config.seqLen * headSize / 2;

weights.wcls = weights.tokenEmbeddingTable; // no shared weights

// We've read out all the content from the checkpoint, but we're not in C anymore and we don't have to pretend that our
// 2D arrays are represented by a 1D array. Let's fix that.

function splitFloat32Array(array, numChunks) {
    const result = [];

    const chunkSize = array.length / numChunks
    for (let i = 0; i < array.length; i += chunkSize) {
        const end = Math.min(i + chunkSize, array.length);
        result.push(array.subarray(i, end));
    }
    return result;
}

// Let's resize the 1d weights arrays into 2d and 3d tensors
weights.tokenEmbeddingTable = splitFloat32Array(weights.tokenEmbeddingTable, config.vocabSize); // [vocabSize, dim]
weights.rmsAttWeight = splitFloat32Array(weights.rmsAttWeight, config.nLayers); // [layers, dim]
weights.rmsFFNWeight = splitFloat32Array(weights.rmsFFNWeight, config.nLayers); // [layers, dim]

// All these are attention matmuls with the 3d tensor format [layers, [dim, dim]]
weights.wq = splitFloat32Array(weights.wq, config.nLayers)
    .map((array) => splitFloat32Array(array, config.dim)) // [layers, [dim, dim]]
weights.wk = splitFloat32Array(weights.wk, config.nLayers)
    .map((array) => splitFloat32Array(array, config.dim)); // [layers, [dim, dim]]
weights.wv = splitFloat32Array(weights.wv, config.nLayers)
    .map((array) => splitFloat32Array(array, config.dim));
weights.wo = splitFloat32Array(weights.wo, config.nLayers)
    .map((array) => splitFloat32Array(array, config.dim));

// These are the FFN matmuls with the 3d tensor format [layers, [dim, hiddenDim]]
weights.w1 = splitFloat32Array(weights.w1, config.nLayers)
    .map((array) => splitFloat32Array(array, config.hiddenDim));
weights.w2 = splitFloat32Array(weights.w2, config.nLayers) // w2 is [layers, [hiddenDim, dim]]
    .map((array) => splitFloat32Array(array, config.dim));
weights.w3 = splitFloat32Array(weights.w3, config.nLayers)
    .map((array) => splitFloat32Array(array, config.hiddenDim));

weights.freqCisReal = splitFloat32Array(weights.freqCisReal, config.seqLen);
weights.freqCisImag = splitFloat32Array(weights.freqCisImag, config.seqLen);


weights.wcls = weights.tokenEmbeddingTable

console.log("Constructed weights!")

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
    tokenizer.vocabScores[i] = score
    
    len = tokenizerBin.readInt32LE(offset)
    offset += FLOAT_SIZE
    
    let str = "";
    for (let j = 0; j < len; j++) {
        str += String.fromCharCode(tokenizerBin[offset++]);
    }
    tokenizer.vocab[i] = str;
}

// we're skipping byte pieces and sortedVocab as unnecessary for now
// time to generate! let's start by encoding a prompt

// let prompt = "In the"
let prompt = process.argv[3]
if (prompt[0] != " ") { prompt = " " + prompt } // see add_dummy_prefix from llama2
const promptTokens = []

prompt.split("").forEach((char) => {
    promptTokens.push(tokenizer.vocab.indexOf(char))
})

const tries = 50
for (let i = 0; i < tries; i++) {
    let bestScore = -1e10
    let bestId = -1
    let bestIdx = -1

    for (let i = 0; i < promptTokens.length - 1; ++i) {
        const token = promptTokens[i]
        const nextToken = promptTokens[i+1]
        if (nextToken == undefined) {
            continue
        }
        const mergedToken = tokenizer.vocab[token] + tokenizer.vocab[nextToken]
        const mergedTokenIndex = tokenizer.vocab.indexOf(mergedToken)
        
        if (mergedTokenIndex == undefined || mergedTokenIndex == -1) {
            continue
        }
        
        if (tokenizer.vocabScores[mergedTokenIndex] > bestScore) {
            bestScore = tokenizer.vocabScores[mergedTokenIndex]
            bestId = mergedTokenIndex
            bestIdx = i
        }
    }

    if (bestIdx == -1) { break } // no mergeable pairs left, we're done

    promptTokens.splice(bestIdx, 2, bestId)

}

// Now we have our prompt tokenized into a list of token ids
// Let's get started

const steps = Number(process.argv[2]) // the -n param in llama2.c
let pos = 0
promptTokens.unshift(1) // 1 = BOS (beginning of string) token in Llama-2 sentence-piece
const numPromptTokens = promptTokens.length
let token  = promptTokens[0]
let next

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
    // NB: this was a bug in at least one of the ports, I think: att was initialized too small (without * nHeads)
    att: new Float32Array(config.seqLen * config.nHeads), // buffer for scores/attention values // run.js has (n_heads, seq_len) instead
    logits: new Float32Array(config.vocabSize), // output logits
    // kv cache
    keyCache: new Float32Array(config.nLayers * config.seqLen * config.dim),   // (layer, sq_len, dim)
    valueCache: new Float32Array(config.nLayers * config.seqLen * config.dim), // (layer, seq_len, dim)
};
runState.keyCache = splitFloat32Array(runState.keyCache, config.nLayers)
    .map((arr) => splitFloat32Array(arr, config.seqLen)) // 3d tensor
runState.valueCache = splitFloat32Array(runState.valueCache, config.nLayers)
    .map((arr) => splitFloat32Array(arr, config.seqLen)) // 3d tensor

function matmul(output, input, weight) {
    for (let i = 0; i < output.length; i++) {
        output[i] = 0;
        for (let j = 0; j < input.length; j++) {
            output[i] += input[j] * weight[i][j];
        }
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

function randomSample(array) {
    const index = Math.floor(Math.random() * array.length);
    return array[index];
}

const output = []

while (pos < steps) {
    // forward pass
    // copy the token embedding into x
    runState.x = new Float32Array(weights.tokenEmbeddingTable[token])

   
    // forward each layer
    for (let layer = 0; layer < config.nLayers; layer++) {

        rmsnorm(runState.xb, runState.x, weights.rmsAttWeight[layer]) // attention rmsnorm

        // attention qkv matmuls
        matmul(runState.q, runState.xb, weights.wq[layer])
        matmul(runState.k, runState.xb, weights.wk[layer])
        matmul(runState.v, runState.xb, weights.wv[layer])

        // RoPE relative pos encoding: complex-valued rotate q and k in each head
        // llama 2 doesn't use freqx from the weights, but the port I copied does
        for (let head = 0; head < config.nHeads; head++) {
            const start = head * headSize;
            for (let i = 0; i < headSize; i += 2) {
                const q0 = runState.q[start + i];  
                const q1 = runState.q[start + i + 1];
                const k0 = runState.k[start + i];
                const k1 = runState.k[start + i + 1];
                const fcr = weights.freqCisReal[pos][i / 2]
                const fci = weights.freqCisImag[pos][i / 2]

                runState.q[start + i] = q0 * fcr - q1 * fci;
                runState.q[start + i + 1] = q0 * fci + q1 * fcr;
                runState.k[start + i] = k0 * fcr - k1 * fci;
                runState.k[start + i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // populate key/value caches. llama.c does this way above here but all ports do this here
        runState.keyCache[layer][pos] = new Float32Array(runState.k)
        runState.valueCache[layer][pos] = new Float32Array(runState.v)

        // multi-head attention, iterate over all heads. could parallelize this in theory.
        for (let head = 0; head < config.nHeads; head++) {
            const att = runState.att.subarray(head * config.seqLen, (head + 1) * config.seqLen)          
            for (let t = 0; t <= pos; t++) {              
                // att score
                let score = 0;
                for (let i = 0; i < headSize; i++) {
                    // hmm
                    const qkDotProd = runState.q[head * headSize + i] * runState.keyCache[layer][t][head * headSize + i];
                    score += qkDotProd
                }

                score /= Math.sqrt(headSize);
                // save to att buffer  
                att[t] = score;
            }


            // softmax att weights  
            softmax(att, pos + 1);

            // weighted sum of v into xb
            for (let i = 0; i < headSize; i++) {
                runState.xb[head * headSize + i] = 0;
                for (let t = 0; t <= pos; t++) {
                    runState.xb[head * headSize + i] += att[t] * runState.valueCache[layer][t][head * headSize + i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(runState.xb2, runState.xb, weights.wo[layer])

        // residual connection 
        for (let i = 0; i < runState.xb2.length; i++) { // should be config.dim
            runState.x[i] += runState.xb2[i]; 
        }

        rmsnorm(runState.xb, runState.x, weights.rmsFFNWeight[layer]) // ffn rmsnorm
        
        matmul(runState.hb, runState.xb, weights.w1[layer])
        matmul(runState.hb2, runState.xb, weights.w3[layer])

        // SwiGLU non-linearity
        for (let i = 0; i < config.hiddenDim; i++) {
            runState.hb[i] = runState.hb[i] * (1.0 / (1.0 + Math.exp(-runState.hb[i])));
        }
        // elementwise mult with w3(x)
        for (let i = 0; i < config.hiddenDim; i++) {
            runState.hb[i] = runState.hb[i] * runState.hb2[i];
        }

        // final matmul for ffn output
        matmul(runState.xb, runState.hb, weights.w2[layer])

        // residual connection
        for (let i = 0; i < runState.xb.length; i++) {
            runState.x[i] += runState.xb[i];
        }
    }
    
    // final rmsnorm
    rmsnorm(runState.x, runState.x, weights.rmsFinalWeight);

    // classifier into logits
    matmul(runState.logits, runState.x, weights.wcls); // could I just reuse tokenEmbeddingTable? some ports do

    if (pos < numPromptTokens - 1) {
        // still in prompt
        next = promptTokens[pos + 1]
    } else {
        // simplest possible topp over the logits
        const numToSample = 1
        const topp = Array.from(runState.logits).sort((a, b) => b - a).slice(0, numToSample).map(x => runState.logits.indexOf(x))//.map(x => tokenizer.vocab[x])
        next = randomSample(topp)
    }

    let piece = tokenizer.vocab[next]
    if (piece == undefined) {
        piece = "<oops>"
        token = 1
    }

    // I should really decode these bytes instead, but it doesn't seem worth it
    if (piece.match(/^<0x/)) { piece = " " }

    // calculate the attention matrix at this token
    let mashedAtt = runState.att.subarray(0, config.seqLen)
    for (let head = 1; head < config.nHeads; head++) {
        const att = runState.att.subarray(head * config.seqLen, (head + 1) * config.seqLen)
        for (let i = 0; i < att.length; i++) {
            mashedAtt[i] += att[i]
        }
    }
    output.push({
        piece: piece,
        token: token,
        att: mashedAtt
    })

    console.clear()
    // console.log(output.map(x => x.piece).join(""))
    // process.stdout.write(piece)
    buf = ''
    output.forEach((chunk, i) => {
        let colour
        const currentAttention = output[output.length - 1].att[i]
        if (currentAttention > 0.7) {
            colour = colours.bright
        } else if (currentAttention > 0.1) {
            colour = colours.reset
        } else {
            colour = colours.dim
        }
        buf += colour
        buf += chunk.piece
        buf += colours.reset
    })
    process.stdout.write(buf)

    // console.log("token", token, pos)

    token = next

    // hack to pause for keypress between tokens
    // var fd = fs.openSync("/dev/stdin", "rs")
    // fs.readSync(fd, new Buffer(1), 0, 1)
    // fs.closeSync(fd)

    pos += 1
}

